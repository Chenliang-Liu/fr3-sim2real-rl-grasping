from pathlib import Path
import argparse
import math
from datetime import datetime

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import robosuite as suite
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)
from robosuite.utils import transform_utils as T

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


class RoboSuiteLiftGymWrapper(gym.Env):
    """
    robosuite Lift -> Gymnasium wrapper
    Panda / FR3v2 + OSC_POSITION

    Low-dimensional observation (14 dim, expressed in robot base frame):
        0:3   robot0_eef_pos
        3:4   robot0_gripper_open_amount   (compressed from robot0_gripper_qpos)
        4:7   cube_pos
        7:11  cube_quat
        11:14 gripper_to_cube_pos

    Added:
    - cylindrical workspace constraint
    - hard clipping of desired EEF target
    - soft penalty for "out-of-bound intention"
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render: bool = False,
        offscreen_render: bool = False,
        camera_name: str = "frontview",
        camera_width: int = 1280,
        camera_height: int = 720,
        terminate_on_success: bool = True,
        robot: str = "Panda",
        table_full_size=None,
        target_lift_height: float = 0.075,
        max_lift_height: float | None = None,
        post_grasp_vertical_only: bool = True,
        post_grasp_xy_action_scale: float = 0.0,
    ):
        super().__init__()
        self.robot = robot
        self.offscreen_render = bool(offscreen_render)
        self.camera_name = camera_name
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.table_full_size = tuple(table_full_size) if table_full_size is not None else (
            (1.4, 0.9, 0.05) if self.robot == "FR3v2" else (0.8, 0.8, 0.05)
        )
        self.terminate_on_success = terminate_on_success
        self.target_lift_height = float(target_lift_height)
        self.height_success_tolerance = 0.025
        self.max_lift_height = float(max_lift_height) if max_lift_height is not None else self.target_lift_height + 0.065
        self.post_grasp_vertical_only = bool(post_grasp_vertical_only)
        self.post_grasp_xy_action_scale = float(post_grasp_xy_action_scale)
        self.max_stable_cube_speed = 0.04
        self.max_stable_eef_speed = 0.06
        self.stable_hold_steps_required = 10
        self._stable_lift_steps = 0
        self._has_grasped_cube = False
        self._grasp_anchor_cube_xy = None
        self._prev_cube_height = 0.0

        arm_controller_config = load_part_controller_config(default_controller="OSC_POSITION")
        self._osc_input_min = np.asarray(arm_controller_config["input_min"], dtype=np.float32)
        self._osc_input_max = np.asarray(arm_controller_config["input_max"], dtype=np.float32)
        self._osc_output_min = np.asarray(arm_controller_config["output_min"], dtype=np.float32)
        self._osc_output_max = np.asarray(arm_controller_config["output_max"], dtype=np.float32)

        controller_config = refactor_composite_controller_config(
            arm_controller_config,
            self.robot,
            ["right"],
        )

        self.env = suite.make(
            env_name="Lift",
            robots=self.robot,
            controller_configs=controller_config,
            has_renderer=render,
            has_offscreen_renderer=self.offscreen_render,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            table_full_size=self.table_full_size,
            control_freq=20,
            horizon=200,
        )

        low, high = self.env.action_spec
        self.action_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

        print("[INFO] robot           :", self.robot)
        print("[INFO] table_full_size :", self.table_full_size)
        print("[INFO] target_lift_height:", self.target_lift_height)
        print("[INFO] max_lift_height   :", self.max_lift_height)
        print("[INFO] post_grasp_vertical_only:", self.post_grasp_vertical_only)
        print("[INFO] post_grasp_xy_action_scale:", self.post_grasp_xy_action_scale)
        print("[INFO] action_spec low :", low)
        print("[INFO] action_spec high:", high)
        print("[INFO] action_dim      :", low.shape)

        try:
            print("\n=== robot action info ===")
            self.env.robots[0].print_action_info()
            self.env.robots[0].print_action_info_dict()
        except Exception as e:
            print(f"[WARN] Failed to print action info: {e}")

        obs = self.env.reset()
        self._last_obs_dict = obs
        flat_obs = self._flatten_obs(obs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32,
        )

        print(f"[INFO] observation_dim = {flat_obs.shape[0]}")
        print("[INFO] observation_frame = robot base site robot0_right_center")
        self._print_observation_layout()
        self._print_observation_values_info(obs)

        print("[INFO] workspace bounds:", self.get_workspace_bounds())

    def _print_observation_layout(self):
        layout = [
            ("robot0_eef_pos_base", 3),
            ("robot0_gripper_open", 1),
            ("cube_pos_base", 3),
            ("cube_quat_base", 4),
            ("gripper_to_cube_pos_base", 3),
        ]

        print("\n[INFO] 14-dim Observation Layout")
        start = 0
        for name, dim in layout:
            end = start + dim
            print(f"  {name:<24} [{start:2d}:{end:2d}]  dim={dim}")
            start = end
        print(f"[INFO] Total observation dim = {start}\n")

    def _print_observation_values_info(self, obs_dict):
        print("[INFO] Raw observation keys used for low-dim obs:")
        keys = [
            "robot0_eef_pos",
            "robot0_gripper_qpos",
            "cube_pos",
            "cube_quat",
            "gripper_to_cube_pos",
        ]
        for k in keys:
            v = np.asarray(obs_dict[k])
            print(f"  {k:<24} shape={v.shape}")
        print()

    def _robot_base_pose_in_world(self):
        base_site_id = self.env.sim.model.site_name2id("robot0_right_center")
        base_pos = np.asarray(self.env.sim.data.site_xpos[base_site_id], dtype=np.float32).ravel()
        base_rot = np.asarray(self.env.sim.data.site_xmat[base_site_id], dtype=np.float32).reshape(3, 3)
        return base_pos, base_rot

    def _world_pos_to_robot_base(self, pos_world: np.ndarray) -> np.ndarray:
        base_pos, base_rot = self._robot_base_pose_in_world()
        return (base_rot.T @ (np.asarray(pos_world, dtype=np.float32).ravel() - base_pos)).astype(np.float32)

    def _world_quat_to_robot_base(self, quat_world_xyzw: np.ndarray) -> np.ndarray:
        _, base_rot = self._robot_base_pose_in_world()
        obj_rot_world = T.quat2mat(np.asarray(quat_world_xyzw, dtype=np.float32).ravel())
        obj_rot_base = base_rot.T @ obj_rot_world
        return T.mat2quat(obj_rot_base).astype(np.float32)

    def _flatten_obs(self, obs_dict):
        """
        Build 14-dim low-dimensional observation in the robot base frame.
        """
        eef_pos_world = np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float32).ravel()      # 3
        gripper_qpos = np.asarray(obs_dict["robot0_gripper_qpos"], dtype=np.float32).ravel()  # 2
        cube_pos_world = np.asarray(obs_dict["cube_pos"], dtype=np.float32).ravel()            # 3
        cube_quat_world = np.asarray(obs_dict["cube_quat"], dtype=np.float32).ravel()          # 4

        eef_pos = self._world_pos_to_robot_base(eef_pos_world)                                # 3
        cube_pos = self._world_pos_to_robot_base(cube_pos_world)                               # 3
        cube_quat = self._world_quat_to_robot_base(cube_quat_world)                            # 4
        gripper_to_cube_pos = cube_pos - eef_pos                                               # 3

        gripper_open = np.array(
            [float(np.mean(np.abs(gripper_qpos)))],
            dtype=np.float32,
        )                                                                                      # 1

        obs = np.concatenate(
            [
                eef_pos,               # 3
                gripper_open,          # 1
                cube_pos,              # 3
                cube_quat,             # 4
                gripper_to_cube_pos,   # 3
            ],
            axis=0,
        ).astype(np.float32)

        if obs.shape != (14,):
            raise RuntimeError(f"Expected 14-dim observation, got {obs.shape}")

        return obs

    # ----------------------------
    # Workspace constraint methods
    # ----------------------------
    def get_workspace_bounds(self):
        """Conservative cylindrical workspace around the Lift table, in world meters."""
        return {
            "r_min": 0.00,
            "r_max": 0.85,
            "z_min": 0.78,
            "z_max": 1.30,
        }

    def is_pose_reachable(self, x: float, y: float, z: float) -> bool:
        b = self.get_workspace_bounds()
        r = math.hypot(x, y)
        return (b["r_min"] <= r <= b["r_max"]) and (b["z_min"] <= z <= b["z_max"])

    def clip_pose_to_workspace(self, x: float, y: float, z: float):
        """
        Project a target point back into the cylindrical workspace.
        """
        b = self.get_workspace_bounds()

        # Clip z into range
        z_clipped = min(max(z, b["z_min"]), b["z_max"])

        r = math.hypot(x, y)

        # Avoid zero division; if exactly at origin, project to min radius along +x
        if r < 1e-8:
            r_target = b["r_min"]
            return r_target, 0.0, z_clipped

        r_clipped = min(max(r, b["r_min"]), b["r_max"])
        scale = r_clipped / r

        x_clipped = x * scale
        y_clipped = y * scale

        return x_clipped, y_clipped, z_clipped

    def _action_to_eef_delta(self, action_xyz: np.ndarray) -> np.ndarray:
        """
        Convert normalized OSC_POSITION action coordinates to controller-space meters.

        robosuite exposes OSC action limits in normalized controller input units
        (usually [-1, 1]). The controller then scales those inputs to output_min /
        output_max meters before updating the EEF goal. Workspace clipping must use
        the scaled meter delta, not the raw normalized action.
        """
        action_xyz = np.asarray(action_xyz, dtype=np.float32)
        action_xyz = np.clip(action_xyz, self._osc_input_min, self._osc_input_max)

        input_span = self._osc_input_max - self._osc_input_min
        output_span = self._osc_output_max - self._osc_output_min
        return (action_xyz - self._osc_input_min) / input_span * output_span + self._osc_output_min

    def _eef_delta_to_action(self, delta_xyz: np.ndarray) -> np.ndarray:
        """Inverse of _action_to_eef_delta for the translational OSC action."""
        delta_xyz = np.asarray(delta_xyz, dtype=np.float32)
        delta_xyz = np.clip(delta_xyz, self._osc_output_min, self._osc_output_max)

        input_span = self._osc_input_max - self._osc_input_min
        output_span = self._osc_output_max - self._osc_output_min
        return (delta_xyz - self._osc_output_min) / output_span * input_span + self._osc_input_min

    def _compute_success(self) -> bool:
        check_success_fn = getattr(self.env, "_check_success", None)
        if callable(check_success_fn):
            try:
                return bool(check_success_fn())
            except Exception:
                return False
        return False

    def _table_height(self) -> float:
        return float(self.env.model.mujoco_arena.table_offset[2])

    def _cube_height_above_table(self) -> float:
        cube_pos = np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_id], dtype=np.float32)
        return float(cube_pos[2] - self._table_height())

    def _cube_pos_world(self) -> np.ndarray:
        return np.asarray(self.env.sim.data.body_xpos[self.env.cube_body_id], dtype=np.float32).ravel()

    def _is_grasping_cube(self) -> bool:
        try:
            return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cube))
        except Exception:
            return False

    def _cube_speed(self) -> float:
        try:
            cube_vel = np.asarray(self.env.sim.data.get_body_xvelp(self.env.cube.root_body), dtype=np.float32)
            return float(np.linalg.norm(cube_vel))
        except Exception:
            return 0.0

    def _eef_speed(self) -> float:
        try:
            hand_vel = self.env.robots[0]._hand_vel
            if isinstance(hand_vel, dict):
                hand_vel = hand_vel.get("right", next(iter(hand_vel.values())))
            return float(np.linalg.norm(np.asarray(hand_vel, dtype=np.float32).reshape(-1)[:3]))
        except Exception:
            return 0.0

    def _controlled_lift_metrics(self):
        cube_height = self._cube_height_above_table()
        cube_speed = self._cube_speed()
        eef_speed = self._eef_speed()

        height_error = abs(cube_height - self.target_lift_height)
        in_height_band = height_error <= self.height_success_tolerance
        not_over_lifted = cube_height <= self.max_lift_height
        stable = cube_speed <= self.max_stable_cube_speed and eef_speed <= self.max_stable_eef_speed
        controlled_lift = in_height_band and not_over_lifted and stable

        if controlled_lift:
            self._stable_lift_steps += 1
        else:
            self._stable_lift_steps = 0

        held_success = self._stable_lift_steps >= self.stable_hold_steps_required
        return {
            "cube_height": cube_height,
            "cube_speed": cube_speed,
            "eef_speed": eef_speed,
            "height_error": height_error,
            "controlled_lift": controlled_lift,
            "held_success": held_success,
            "stable_lift_steps": self._stable_lift_steps,
        }

    def _post_lift_reward(self, action: np.ndarray, metrics: dict, grasping_cube: bool) -> tuple[float, dict]:
        cube_height = metrics["cube_height"]
        height_error = metrics["height_error"]
        cube_speed = metrics["cube_speed"]
        eef_speed = metrics["eef_speed"]

        target_height_reward = 0.45 * math.exp(-40.0 * height_error)
        stable_reward = 0.20 if metrics["controlled_lift"] else 0.0

        over_lift = max(0.0, cube_height - self.max_lift_height)
        below_target = max(0.0, self.target_lift_height - cube_height)
        action_penalty = 0.006 * float(np.linalg.norm(action[:3]))
        height_penalty = 8.0 * over_lift + 1.0 * below_target

        if cube_height > 0.04:
            motion_penalty = 0.45 * cube_speed + 0.20 * eef_speed
        else:
            motion_penalty = 0.0

        vertical_lift_reward = 0.0
        xy_drift_penalty = 0.0
        xy_action_penalty = 0.0
        cube_xy_drift = 0.0
        height_progress = cube_height - self._prev_cube_height

        if grasping_cube or self._has_grasped_cube:
            vertical_lift_reward = 1.2 * max(0.0, height_progress)
            xy_action_penalty = 0.04 * float(np.linalg.norm(action[:2]))
            if self._grasp_anchor_cube_xy is not None:
                cube_xy = self._cube_pos_world()[:2]
                cube_xy_drift = float(np.linalg.norm(cube_xy - self._grasp_anchor_cube_xy))
                xy_drift_penalty = 2.0 * cube_xy_drift

        reward_delta = (
            target_height_reward
            + stable_reward
            + vertical_lift_reward
            - height_penalty
            - motion_penalty
            - action_penalty
            - xy_action_penalty
            - xy_drift_penalty
        )
        components = {
            "target_height_reward": target_height_reward,
            "stable_reward": stable_reward,
            "vertical_lift_reward": vertical_lift_reward,
            "height_penalty": height_penalty,
            "motion_penalty": motion_penalty,
            "action_penalty": action_penalty,
            "xy_action_penalty": xy_action_penalty,
            "xy_drift_penalty": xy_drift_penalty,
            "cube_xy_drift": cube_xy_drift,
            "height_progress": height_progress,
            "hold_reward_delta": reward_delta,
        }
        return reward_delta, components

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self.env.reset()
        self._last_obs_dict = obs_dict
        self._stable_lift_steps = 0
        self._has_grasped_cube = False
        self._grasp_anchor_cube_xy = None
        self._prev_cube_height = self._cube_height_above_table()
        return self._flatten_obs(obs_dict), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Current EEF position from last obs
        cur_pos = np.asarray(self._last_obs_dict["robot0_eef_pos"], dtype=np.float32).ravel()
        desired_delta = self._action_to_eef_delta(action[:3])
        grasping_before_step = self._is_grasping_cube()
        if grasping_before_step and not self._has_grasped_cube:
            self._has_grasped_cube = True
            self._grasp_anchor_cube_xy = self._cube_pos_world()[:2].copy()

        if self.post_grasp_vertical_only and self._has_grasped_cube:
            desired_delta[:2] *= self.post_grasp_xy_action_scale

        # Desired next EEF target implied by scaled OSC delta action
        desired_x = float(cur_pos[0] + desired_delta[0])
        desired_y = float(cur_pos[1] + desired_delta[1])
        desired_z = float(cur_pos[2] + desired_delta[2])

        # Hard constraint: clip desired target back into cylindrical workspace
        clipped_x, clipped_y, clipped_z = self.clip_pose_to_workspace(
            desired_x, desired_y, desired_z
        )

        # Rewrite normalized action after clipping in meter space
        clipped_delta = np.array(
            [
                clipped_x - float(cur_pos[0]),
                clipped_y - float(cur_pos[1]),
                clipped_z - float(cur_pos[2]),
            ],
            dtype=np.float32,
        )
        action[:3] = self._eef_delta_to_action(clipped_delta)

        # Step env with constrained action
        obs_dict, reward, done, info = self.env.step(action)
        self._last_obs_dict = obs_dict
        obs = self._flatten_obs(obs_dict)

        # Soft penalty: penalize out-of-bound intention
        boundary_penalty = 0.1 * float(np.linalg.norm([
            desired_x - clipped_x,
            desired_y - clipped_y,
            desired_z - clipped_z,
        ]))

        metrics = self._controlled_lift_metrics()
        grasping_cube = self._is_grasping_cube()
        if grasping_cube and not self._has_grasped_cube:
            self._has_grasped_cube = True
            self._grasp_anchor_cube_xy = self._cube_pos_world()[:2].copy()
        hold_reward_delta, hold_components = self._post_lift_reward(action, metrics, grasping_cube)
        reward = float(reward) + hold_reward_delta - boundary_penalty
        self._prev_cube_height = float(metrics["cube_height"])

        raw_lift_success = self._compute_success()
        success = bool(metrics["held_success"])
        terminated = bool(success and self.terminate_on_success)
        truncated = bool(done)

        info = dict(info) if info is not None else {}
        info["is_success"] = success
        info["raw_lift_success"] = raw_lift_success
        info["controlled_lift"] = bool(metrics["controlled_lift"])
        info["grasping_cube"] = bool(grasping_cube)
        info["has_grasped_cube"] = bool(self._has_grasped_cube)
        info["post_grasp_vertical_mode"] = bool(self.post_grasp_vertical_only and self._has_grasped_cube)
        info["stable_lift_steps"] = int(metrics["stable_lift_steps"])
        info["cube_height_above_table"] = float(metrics["cube_height"])
        info["target_lift_height"] = float(self.target_lift_height)
        info["cube_speed"] = float(metrics["cube_speed"])
        info["eef_speed"] = float(metrics["eef_speed"])
        info["boundary_penalty"] = boundary_penalty
        info["eef_pos_clipped"] = boundary_penalty > 0.0
        info["desired_eef_pos"] = np.array([desired_x, desired_y, desired_z], dtype=np.float32)
        info["clipped_eef_pos"] = np.array([clipped_x, clipped_y, clipped_z], dtype=np.float32)
        info["applied_action"] = action.copy()
        info.update(hold_components)

        return obs, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def capture_frame(self):
        frame = self.env.sim.render(
            camera_name=self.camera_name,
            width=self.camera_width,
            height=self.camera_height,
            depth=False,
        )
        return np.flipud(frame).astype(np.uint8)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def make_env(
    render: bool = False,
    offscreen_render: bool = False,
    camera_name: str = "frontview",
    camera_width: int = 1280,
    camera_height: int = 720,
    terminate_on_success: bool = True,
    robot: str = "Panda",
    table_full_size=None,
    target_lift_height: float = 0.075,
    max_lift_height: float | None = None,
    post_grasp_vertical_only: bool = True,
    post_grasp_xy_action_scale: float = 0.0,
):
    env = RoboSuiteLiftGymWrapper(
        render=render,
        offscreen_render=offscreen_render,
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        terminate_on_success=terminate_on_success,
        robot=robot,
        table_full_size=table_full_size,
        target_lift_height=target_lift_height,
        max_lift_height=max_lift_height,
        post_grasp_vertical_only=post_grasp_vertical_only,
        post_grasp_xy_action_scale=post_grasp_xy_action_scale,
    )
    env = Monitor(
        env,
        info_keywords=(
            "is_success",
            "raw_lift_success",
            "controlled_lift",
            "cube_height_above_table",
            "cube_speed",
            "eef_speed",
            "cube_xy_drift",
            "grasping_cube",
            "has_grasped_cube",
            "post_grasp_vertical_mode",
            "stable_lift_steps",
        ),
    )
    return env


def safe_reset(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    return obs, info


def safe_step(env, action):
    out = env.step(action)

    if not isinstance(out, tuple):
        raise RuntimeError(f"Unsupported step output type: {type(out)}")

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info

    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info

    raise RuntimeError(f"Unsupported step output length: {len(out)}")


def unpack_done(done):
    if isinstance(done, (list, tuple)):
        return bool(done[0]) if len(done) > 0 else False
    if isinstance(done, np.ndarray):
        if done.shape == ():
            return bool(done.item())
        return bool(done.reshape(-1)[0]) if done.size > 0 else False
    return bool(done)


def unpack_info(info):
    if isinstance(info, list):
        if len(info) > 0 and isinstance(info[0], dict):
            return info[0]
        return {}
    if isinstance(info, tuple):
        if len(info) > 0 and isinstance(info[0], dict):
            return info[0]
        return {}
    if isinstance(info, dict):
        return info
    return {}


class TrainSuccessRateCallback(BaseCallback):
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.held_successes = []
        self.raw_lift_successes = []
        self.controlled_lift_successes = []
        self.current_held_successes = []
        self.current_raw_lift_successes = []
        self.current_controlled_lift_successes = []

    @staticmethod
    def _recent_mean(values, window_size: int) -> float:
        recent = values[-window_size:]
        return float(np.mean(recent)) if recent else 0.0

    def _ensure_env_slots(self, n_envs: int) -> None:
        missing = n_envs - len(self.current_held_successes)
        if missing <= 0:
            return
        self.current_held_successes.extend([False] * missing)
        self.current_raw_lift_successes.extend([False] * missing)
        self.current_controlled_lift_successes.extend([False] * missing)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        self._ensure_env_slots(len(infos))

        for env_idx, (info, done) in enumerate(zip(infos, dones)):
            info_dict = unpack_info(info)
            if info_dict.get("is_success", False):
                self.current_held_successes[env_idx] = True
            if info_dict.get("raw_lift_success", False):
                self.current_raw_lift_successes[env_idx] = True
            if info_dict.get("controlled_lift", False):
                self.current_controlled_lift_successes[env_idx] = True

            for metric_name in (
                "cube_height_above_table",
                "cube_speed",
                "eef_speed",
                "cube_xy_drift",
                "height_progress",
                "stable_lift_steps",
                "vertical_lift_reward",
                "xy_action_penalty",
                "xy_drift_penalty",
                "grasping_cube",
                "has_grasped_cube",
                "post_grasp_vertical_mode",
                "hold_reward_delta",
            ):
                if metric_name in info_dict:
                    self.logger.record(f"rollout/{metric_name}", float(info_dict[metric_name]))

            if bool(done):
                self.held_successes.append(float(self.current_held_successes[env_idx]))
                self.raw_lift_successes.append(float(self.current_raw_lift_successes[env_idx]))
                self.controlled_lift_successes.append(float(self.current_controlled_lift_successes[env_idx]))

                self.current_held_successes[env_idx] = False
                self.current_raw_lift_successes[env_idx] = False
                self.current_controlled_lift_successes[env_idx] = False

                self.logger.record(
                    "rollout/held_success_rate",
                    self._recent_mean(self.held_successes, self.window_size),
                )
                self.logger.record(
                    "rollout/raw_lift_success_rate",
                    self._recent_mean(self.raw_lift_successes, self.window_size),
                )
                self.logger.record(
                    "rollout/controlled_lift_success_rate",
                    self._recent_mean(self.controlled_lift_successes, self.window_size),
                )
        return True


class EvalSuccessCallback(EvalCallback):
    def __init__(self, *args, n_eval_episodes_success: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_eval_episodes_success = n_eval_episodes_success

    def _log_success_rate(self):
        held_successes = []
        raw_lift_successes = []
        controlled_lift_successes = []

        for _ in range(self.n_eval_episodes_success):
            obs, _ = safe_reset(self.eval_env)
            done = False
            held_success = False
            raw_lift_success = False
            controlled_lift_success = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, _, done_raw, info_raw = safe_step(self.eval_env, action)
                done = unpack_done(done_raw)
                info = unpack_info(info_raw)
                if info.get("is_success", False):
                    held_success = True
                if info.get("raw_lift_success", False):
                    raw_lift_success = True
                if info.get("controlled_lift", False):
                    controlled_lift_success = True

            held_successes.append(float(held_success))
            raw_lift_successes.append(float(raw_lift_success))
            controlled_lift_successes.append(float(controlled_lift_success))

        success_rate = float(np.mean(held_successes)) if held_successes else 0.0
        raw_lift_success_rate = float(np.mean(raw_lift_successes)) if raw_lift_successes else 0.0
        controlled_lift_success_rate = float(np.mean(controlled_lift_successes)) if controlled_lift_successes else 0.0
        self.logger.record("eval/held_success_rate", success_rate)
        self.logger.record("eval/raw_lift_success_rate", raw_lift_success_rate)
        self.logger.record("eval/controlled_lift_success_rate", controlled_lift_success_rate)
        if self.verbose > 0:
            print(
                f"Eval held_success_rate: {success_rate:.3f}, "
                f"raw_lift_success_rate: {raw_lift_success_rate:.3f}, "
                f"controlled_lift_success_rate: {controlled_lift_success_rate:.3f}"
            )

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._log_success_rate()
        return result


def run_visualization(
    model_path: str,
    n_episodes: int = 5,
    robot: str = "Panda",
    table_full_size=None,
    target_lift_height: float = 0.075,
    max_lift_height: float | None = None,
    post_grasp_vertical_only: bool = True,
    post_grasp_xy_action_scale: float = 0.0,
    record_mp4: Path | None = None,
    record_fps: int = 20,
    camera_name: str = "frontview",
    camera_width: int = 1280,
    camera_height: int = 720,
):
    record_mp4 = Path(record_mp4) if record_mp4 is not None else None
    vis_env = make_env(
        render=record_mp4 is None,
        offscreen_render=record_mp4 is not None,
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
        robot=robot,
        table_full_size=table_full_size,
        target_lift_height=target_lift_height,
        max_lift_height=max_lift_height,
        post_grasp_vertical_only=post_grasp_vertical_only,
        post_grasp_xy_action_scale=post_grasp_xy_action_scale,
    )
    model = SAC.load(model_path)
    writer = None

    if record_mp4 is not None:
        import imageio.v2 as imageio

        if record_mp4.is_dir() or str(record_mp4).endswith("/"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_mp4 = record_mp4 / f"{Path(model_path).stem}_{robot}_vis_{timestamp}.mp4"
        record_mp4.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(record_mp4), fps=record_fps, macro_block_size=1)
        print(f"[REC] writing mp4 to: {record_mp4}")

    try:
        for ep in range(n_episodes):
            obs, _ = safe_reset(vis_env)
            if writer is not None:
                writer.append_data(vis_env.env.capture_frame())
            done = False
            episode_reward = 0.0
            episode_success = False
            clipped_count = 0
            penalty_sum = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward_raw, done_raw, info_raw = safe_step(vis_env, action)
                if writer is None:
                    vis_env.render()
                else:
                    writer.append_data(vis_env.env.capture_frame())

                reward = (
                    float(np.asarray(reward_raw).reshape(-1)[0])
                    if isinstance(reward_raw, (list, tuple, np.ndarray))
                    else float(reward_raw)
                )
                done = unpack_done(done_raw)
                info = unpack_info(info_raw)

                episode_reward += reward
                if info.get("is_success", False):
                    episode_success = True
                if info.get("eef_pos_clipped", False):
                    clipped_count += 1
                penalty_sum += float(info.get("boundary_penalty", 0.0))

            print(
                f"[VIS] episode={ep + 1}, reward={episode_reward:.3f}, "
                f"success={int(episode_success)}, clipped_steps={clipped_count}, "
                f"boundary_penalty_sum={penalty_sum:.4f}"
            )
    finally:
        if writer is not None:
            writer.close()
            print(f"[REC] saved mp4: {record_mp4}")
        vis_env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--robot", choices=["Panda", "FR3v2"], default="Panda")
    parser.add_argument("--target-lift-height", type=float, default=0.075)
    parser.add_argument(
        "--max-lift-height",
        type=float,
        default=None,
        help="Maximum allowed cube height above table. Defaults to target_lift_height + 0.065.",
    )
    parser.add_argument(
        "--table-size",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Table full size in meters. Defaults to a larger table for FR3v2.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("runs/panda_lift_sac_osc_position_lowdim_workspace"))
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--visualize", action="store_true", help="Render the best model after training.")
    parser.add_argument("--visualize-only", type=Path, default=None, help="Render an existing SAC model zip and exit.")
    parser.add_argument("--n-vis-episodes", type=int, default=5)
    parser.add_argument(
        "--record-mp4",
        type=Path,
        default=None,
        help="Save visualize-only output to an mp4 file. If a directory is passed, an auto-named mp4 is created inside it.",
    )
    parser.add_argument("--record-fps", type=int, default=20)
    parser.add_argument("--record-camera", type=str, default="frontview")
    parser.add_argument("--record-width", type=int, default=1280)
    parser.add_argument("--record-height", type=int, default=720)
    parser.add_argument("--no-terminate-on-success", action="store_true")
    parser.add_argument(
        "--disable-post-grasp-vertical-only",
        action="store_true",
        help="Disable hard x/y action suppression after the cube is grasped.",
    )
    parser.add_argument(
        "--post-grasp-xy-action-scale",
        type=float,
        default=0.0,
        help="Scale applied to x/y OSC deltas after grasp. 0 means vertical-only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.visualize_only is not None:
        run_visualization(
            str(args.visualize_only),
            n_episodes=args.n_vis_episodes,
            robot=args.robot,
            table_full_size=args.table_size,
            target_lift_height=args.target_lift_height,
            max_lift_height=args.max_lift_height,
            post_grasp_vertical_only=not args.disable_post_grasp_vertical_only,
            post_grasp_xy_action_scale=args.post_grasp_xy_action_scale,
            record_mp4=args.record_mp4,
            record_fps=args.record_fps,
            camera_name=args.record_camera,
            camera_width=args.record_width,
            camera_height=args.record_height,
        )
        raise SystemExit(0)

    total_timesteps = args.total_timesteps
    work_dir = args.work_dir
    best_model_dir = work_dir / "best_model"
    final_model_dir = work_dir / "final_model"
    eval_log_dir = work_dir / "eval_logs"
    tb_log_dir = work_dir / "tb"

    for p in [best_model_dir, final_model_dir, eval_log_dir, tb_log_dir]:
        p.mkdir(parents=True, exist_ok=True)

    terminate_on_success = not args.no_terminate_on_success
    train_env = make_env(
        render=False,
        terminate_on_success=terminate_on_success,
        robot=args.robot,
        table_full_size=args.table_size,
        target_lift_height=args.target_lift_height,
        max_lift_height=args.max_lift_height,
        post_grasp_vertical_only=not args.disable_post_grasp_vertical_only,
        post_grasp_xy_action_scale=args.post_grasp_xy_action_scale,
    )
    eval_env = make_env(
        render=False,
        terminate_on_success=terminate_on_success,
        robot=args.robot,
        table_full_size=args.table_size,
        target_lift_height=args.target_lift_height,
        max_lift_height=args.max_lift_height,
        post_grasp_vertical_only=not args.disable_post_grasp_vertical_only,
        post_grasp_xy_action_scale=args.post_grasp_xy_action_scale,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=256,
        learning_starts=5_000,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        tensorboard_log=str(tb_log_dir),
        device="auto",
    )

    train_success_callback = TrainSuccessRateCallback(window_size=100)
    eval_callback = EvalSuccessCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        n_eval_episodes_success=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callback = CallbackList([train_success_callback, eval_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    final_model_path = final_model_dir / "sac_panda_lift_final"
    model.save(str(final_model_path))
    print(f"Final model saved to: {final_model_path}")

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
    )
    print(f"Final eval reward: {mean_reward:.3f} ± {std_reward:.3f}")

    best_model_zip = best_model_dir / "best_model.zip"
    if best_model_zip.exists():
        print(f"Best model saved to: {best_model_zip}")
    else:
        print("Best model was not saved. Check EvalCallback configuration.")

    eval_env.close()
    train_env.close()

    if args.visualize and best_model_zip.exists():
        run_visualization(
            str(best_model_zip),
            n_episodes=args.n_vis_episodes,
            robot=args.robot,
            table_full_size=args.table_size,
            target_lift_height=args.target_lift_height,
            max_lift_height=args.max_lift_height,
            post_grasp_vertical_only=not args.disable_post_grasp_vertical_only,
            post_grasp_xy_action_scale=args.post_grasp_xy_action_scale,
        )
