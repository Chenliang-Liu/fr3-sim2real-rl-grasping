#!/usr/bin/env python3
"""Guarded FR3 policy runner using the local FR3Controller interface.

This matches the control path used by fr3_xbox_moveitservo_lin_controller.py:
small Cartesian target increments are sent through FR3Controller.move_to.

Default mode is dry-run. Add --execute to move the robot. Add --enable-gripper
to allow rule-gated gripper closing and lift after grasp.
"""

import argparse
import sys
import threading
import time
from math import atan2, asin, copysign, pi

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from stable_baselines3 import SAC
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from fr3_controller import FR3Controller
except ImportError as exc:
    raise ImportError(
        "Could not import the bundled FR3Controller. Ensure fr3_sim2real/fr3_controller.py "
        "is present, source the configured FR3 ROS 2 workspace, and use a compatible "
        "Python environment with its ROS message and NumPy dependencies. "
        "See docs/sim2real.md; the chained exception identifies the missing import."
    ) from exc


def patch_numpy_pickle_compat():
    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric


def quat_to_rpy(x, y, z, w):
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = copysign(pi / 2, sinp) if abs(sinp) >= 1 else asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(siny, cosy)
    return roll, pitch, yaw


def transform_to_pos_quat(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    pos = np.array([t.x, t.y, t.z], dtype=np.float32)
    quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
    return pos, quat


class PolicyCartesian:
    def __init__(self, args):
        self.args = args
        patch_numpy_pickle_compat()
        self.model = SAC.load(args.model)
        self.controller = FR3Controller()
        self.node = self.controller
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.executor = MultiThreadedExecutor(num_threads=8)
        self.executor.add_node(self.controller)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.motion_lock = threading.Lock()
        self.gripper_lock = threading.Lock()
        self.filtered_action = np.zeros(3, dtype=np.float32)
        self.last_print = 0.0
        self.grasping = False
        self.grasped = False
        self.lift_sent = False

    def start(self):
        self.spin_thread.start()
        mode = "EXECUTE" if self.args.execute else "DRY-RUN"
        self.node.get_logger().warn(f"Mode: {mode}")
        self.node.get_logger().info(f"Loaded model: {self.args.model}")
        if self.args.reset:
            if self.args.execute:
                self.node.get_logger().warn("Resetting robot before policy loop.")
                self.controller.reset()
                time.sleep(1.0)
            else:
                self.node.get_logger().warn("Skipping reset in dry-run mode; --execute is required.")

    def shutdown(self):
        try:
            self.controller.destroy_node()
        finally:
            self.executor.shutdown()

    def lookup(self, target_frame, source_frame):
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=self.args.tf_timeout),
        )
        return transform_to_pos_quat(transform)

    def wait_ready(self):
        deadline = time.time() + self.args.wait_tf_sec
        last_error = None
        while rclpy.ok() and time.time() < deadline:
            try:
                self.lookup(self.args.base_frame, self.args.eef_frame)
                self.lookup(self.args.base_frame, self.args.cube_frame)
                self.node.get_logger().info("Required TFs are available.")
                return True
            except TransformException as exc:
                last_error = exc
            time.sleep(0.1)
        self.node.get_logger().error(f"Timed out waiting for required TFs: {last_error}")
        return False

    def build_obs(self):
        eef_pos, eef_quat = self.lookup(self.args.base_frame, self.args.eef_frame)
        cube_pos, cube_quat = self.lookup(self.args.base_frame, self.args.cube_frame)
        rel_pos = cube_pos - eef_pos
        gripper_open = 0.0 if self.grasped else self.args.gripper_open
        obs = np.concatenate(
            [
                eef_pos,
                np.array([gripper_open], dtype=np.float32),
                cube_pos,
                cube_quat,
                rel_pos,
            ]
        ).astype(np.float32)
        return obs, eef_pos, cube_pos, rel_pos, eef_quat

    def action_to_delta(self, action, eef_pos, cube_pos):
        raw_xyz = np.clip(np.asarray(action[:3], dtype=np.float32), -1.0, 1.0)
        alpha = np.clip(self.args.action_filter_alpha, 0.0, 1.0)
        self.filtered_action = alpha * raw_xyz + (1.0 - alpha) * self.filtered_action
        delta = self.filtered_action * self.args.max_step

        target = eef_pos + delta
        lower = np.array([self.args.x_min, self.args.y_min, self.args.z_min], dtype=np.float32)
        upper = np.array([self.args.x_max, self.args.y_max, self.args.z_max], dtype=np.float32)
        for i in range(3):
            if target[i] < lower[i] and delta[i] < 0.0:
                delta[i] = 0.0
            if target[i] > upper[i] and delta[i] > 0.0:
                delta[i] = 0.0

        min_clearance_z = cube_pos[2] + self.args.min_cube_clearance
        if not self.grasped and target[2] < min_clearance_z and delta[2] < 0.0:
            delta[2] = 0.0
        return delta

    def send_motion(self, target_pos, rpy):
        if not self.args.execute:
            return
        with self.motion_lock:
            self.controller.move_to(
                float(target_pos[0]),
                float(target_pos[1]),
                float(target_pos[2]),
                float(rpy[0]),
                float(rpy[1]),
                float(rpy[2]),
                execute=True,
            )

    def send_lift(self, eef_pos, rpy):
        if not self.args.execute:
            return
        lift_target = eef_pos.copy()
        lift_target[2] = min(self.args.z_max, lift_target[2] + self.args.lift_height)
        self.node.get_logger().warn(
            f"Sending post-grasp lift to z={lift_target[2]:.3f}"
        )
        self.send_motion(lift_target, rpy)
        self.lift_sent = True

    def close_gripper_sequence(self):
        if not self.args.execute or not self.args.enable_gripper or self.grasping or self.grasped:
            return
        self.grasping = True
        try:
            if self.args.execute and self.args.enable_gripper:
                with self.gripper_lock:
                    self.node.get_logger().warn(
                        f"Closing gripper width={self.args.grasp_width:.3f}, speed={self.args.gripper_speed:.3f}"
                    )
                    self.controller.close_gripper(
                        width=self.args.grasp_width,
                        speed=self.args.gripper_speed,
                    )
                time.sleep(self.args.grasp_wait)
            self.grasped = True
            self.node.get_logger().warn("Gripper close sequence finished.")
        except Exception as exc:
            self.node.get_logger().error(f"Gripper close failed: {exc}")
        finally:
            self.grasping = False

    def maybe_trigger_grasp(self, action, rel_pos):
        if not self.args.execute or self.grasped or self.grasping or not self.args.enable_gripper:
            return
        xy_error = float(np.linalg.norm(rel_pos[:2]))
        height = float(-rel_pos[2])
        wants_close = float(action[3]) < self.args.close_action_threshold
        if wants_close and xy_error < self.args.grasp_xy_threshold and height < self.args.grasp_height:
            self.node.get_logger().warn(
                f"Triggering gripper close: xy_error={xy_error:.3f}, height={height:.3f}, action_g={action[3]:.3f}"
            )
            threading.Thread(target=self.close_gripper_sequence, daemon=True).start()

    def step(self):
        try:
            obs, eef_pos, cube_pos, rel_pos, eef_quat = self.build_obs()
        except TransformException as exc:
            self.node.get_logger().warn(f"TF lookup failed: {exc}", throttle_duration_sec=1.0)
            return

        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        delta = self.action_to_delta(action, eef_pos, cube_pos)

        rpy = quat_to_rpy(float(eef_quat[0]), float(eef_quat[1]), float(eef_quat[2]), float(eef_quat[3]))
        target_pos = eef_pos + delta

        self.maybe_trigger_grasp(action, rel_pos)

        if (
            self.args.execute
            and self.grasped
            and self.args.lift_after_grasp
            and not self.lift_sent
            and not self.motion_lock.locked()
        ):
            threading.Thread(target=self.send_lift, args=(eef_pos, rpy), daemon=True).start()
        elif (
            self.args.execute
            and not self.grasping
            and not self.grasped
            and not self.motion_lock.locked()
            and np.linalg.norm(delta) > 1e-6
        ):
            threading.Thread(target=self.send_motion, args=(target_pos, rpy), daemon=True).start()

        now = time.time()
        if now - self.last_print >= 1.0 / self.args.print_hz:
            self.last_print = now
            print(
                "mode=%s eef=%s cube=%s rel=%s action=%s delta=%s target=%s grasped=%s"
                % (
                    "EXEC" if self.args.execute else "DRY",
                    np.array2string(eef_pos, precision=4, suppress_small=True),
                    np.array2string(cube_pos, precision=4, suppress_small=True),
                    np.array2string(rel_pos, precision=4, suppress_small=True),
                    np.array2string(action, precision=4, suppress_small=True),
                    np.array2string(delta, precision=4, suppress_small=True),
                    np.array2string(target_pos, precision=4, suppress_small=True),
                    self.grasped,
                ),
                flush=True,
            )

    def run(self):
        self.start()
        if not self.wait_ready():
            return
        dt = 1.0 / max(self.args.rate, 1e-6)
        while rclpy.ok():
            start = time.time()
            self.step()
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument("--lift-after-grasp", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--eef-frame", default="fr3_hand_tcp")
    parser.add_argument("--cube-frame", default="cube")
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--tf-timeout", type=float, default=0.2)
    parser.add_argument("--wait-tf-sec", type=float, default=10.0)
    parser.add_argument("--max-step", type=float, default=0.002)
    parser.add_argument("--action-filter-alpha", type=float, default=0.25)
    parser.add_argument("--min-cube-clearance", type=float, default=0.10)
    parser.add_argument("--grasp-xy-threshold", type=float, default=0.035)
    parser.add_argument("--grasp-height", type=float, default=0.085)
    parser.add_argument("--close-action-threshold", type=float, default=-0.5)
    parser.add_argument("--grasp-width", type=float, default=0.0)
    parser.add_argument("--gripper-speed", type=float, default=0.05)
    parser.add_argument("--grasp-wait", type=float, default=0.8)
    parser.add_argument("--lift-height", type=float, default=0.075)
    parser.add_argument("--x-min", type=float, default=0.20)
    parser.add_argument("--x-max", type=float, default=0.70)
    parser.add_argument("--y-min", type=float, default=-0.30)
    parser.add_argument("--y-max", type=float, default=0.30)
    parser.add_argument("--z-min", type=float, default=0.03)
    parser.add_argument("--z-max", type=float, default=0.60)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    runner = PolicyCartesian(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        runner.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
