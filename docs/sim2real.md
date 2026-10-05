# Sim2Real deployment reference

This guide is derived from the deployment scripts, the supplied FR3 controller, and the eight terminal records, archived as sanitized historical evidence in [`docs/terminal_logs`](terminal_logs/). It documents the public implementation and recorded deployment; the commands below have not been executed or validated on another robot. The public policy runner includes limited publication-time execution-gate and import fixes described below. Research originals were left unchanged. Replace the placeholders with values measured or configured for your setup.

## What is included and what is external

| Component | Evidence / requirement |
| --- | --- |
| ROS 2 | The records identify ROS 2 Humble and Python 3.10. The scripts/controller import `rclpy`, `tf2_ros`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `std_srvs`, `shape_msgs`, `moveit_msgs`, `franka_msgs`, and `cv_bridge`. |
| Robot and planning stack | A working FR3 workspace providing `franka_fr3_moveit_config`, MoveGroup, IK, robot TF, and gripper action services. These packages are external to this repository. |
| Robot controller | Supplied [`fr3_sim2real/fr3_controller.py`](../fr3_sim2real/fr3_controller.py), with no sibling Python-module imports. Its ROS dependencies and configured services remain external. |
| Camera stack | `realsense2_camera`; the recorded camera is an Intel RealSense D455. The record reports RealSense ROS 4.57.7, librealsense 2.57.7, and firmware 5.13.0.55. These are recorded versions, not a tested compatibility matrix. |
| Python perception libraries | NumPy, SciPy, and an OpenCV build exposing `cv2.aruco`; also a working ROS `cv_bridge` import in the chosen Python environment. |
| Policy library | Stable-Baselines3 with its SAC dependencies, including PyTorch, and a compatible model archive. No exact deployment dependency lockfile is provided. |
| Calibration | A measured transform from `fr3_link0` to `camera_calib_optical_frame`, for the fixed camera placement. The source record's numeric transform belongs to that setup. |
| Object | Default detector configuration: `DICT_4X4_50`, marker ID `0`, physical marker side length `0.04` m. A `cube_size` value of `0.05` m appears in the code but is not used in the position calculation. |

The scripts are ordinary Python entry points, not a complete installable ROS package. Running the public policy script by path makes its bundled `fr3_controller.py` available for import. The runner gives an actionable error if the controller or its dependencies cannot be imported. The publication change removed the original workstation-specific import-path insertion.

The supplied controller subclasses a ROS node and exposes these calls used by the policy runner:

```python
FR3Controller()
controller.move_to(x, y, z, roll, pitch, yaw, execute=True)
controller.close_gripper(width=..., speed=...)
controller.reset()
```

The source workspace's Apache 2.0 license is retained in [`LICENSE-controller-Apache-2.0.txt`](../fr3_sim2real/LICENSE-controller-Apache-2.0.txt). The controller source is copied unchanged; its ROS integration has not been run or validated here.

### Supplied controller behavior

The constructor creates action clients and subscriptions, then waits for services. `/move_action` is required within 10 seconds or construction raises an error. Missing `/compute_ik`, `/franka_gripper/move`, or `/franka_gripper/homing` produces warnings; construction can continue without them. The constructor creates a `/franka_gripper/grasp` client but does not wait for it or use it in `close_gripper()`. Source inspection shows **no motion or gripper goals sent by the constructor**.

`move_to()` waits for `/joint_states`, asserts a cylindrical reach check, creates a MoveGroup goal, and waits synchronously with `rclpy.spin_until_future_complete()`. It sets `plan_only = not execute` and returns a Boolean based on the MoveIt result. The request uses planning group `fr3_arm`, reference frame `fr3_link0`, link `fr3_hand`, and a local target-point offset `[0, 0, 0.095]` m. Confirm that this target-point convention matches the runner's `fr3_hand_tcp` TF. The request sets `planner_id="PTP"`; the actual planner/pipeline configuration is external, despite the source's introductory OMPL description. Velocity and acceleration scaling factors are `0.1`. Its cylindrical check uses radius `0.30–0.855` m and z `0.02–0.80` m; an assertion is not a complete collision or safety controller.

`close_gripper(width, speed)` delegates to `open_gripper()` with the requested width, sends a Franka **Move** action, and returns its success Boolean. Width is in meters and speed in m/s. It does not issue the force-controlled Franka Grasp action or verify object capture. `reset()` moves to `[0.4, 0.0, 0.5, pi, 0.0, 0.0]` and then homes the gripper. Even `reset(execute=False)` still calls gripper homing; the public policy runner skips reset entirely without `--execute`.

The runner does not inspect Boolean failure returns from motion or close calls. A `False` close result can still be followed by `grasped=True`, and a failed lift can still set `lift_sent=True`. Exceptions are distinct: close exceptions are logged without setting the grasp flag. The controller uses nested `spin_until_future_complete()` calls while the runner also spins a multi-thread executor; the historical `generator already executing` error makes thread/executor compatibility an unresolved integration issue. These behaviors are documented rather than changed in the preserved controller.

**Do not execute `fr3_controller.py` directly for readiness:** its `__main__` runs a sequence of physical arm motions with `execute=True`, with no preview flag. The policy runner's execution gates apply to its own calls, not standalone controller demos or direct method use.

## Observations and deployment behavior

The runner builds the following float32 observation in the robot base frame:

| Indices | Meaning |
| --- | --- |
| `0:3` | End-effector/TCP position from TF. |
| `3:4` | Synthetic gripper-open value: `--gripper-open` before the internal grasp flag, then `0.0`. No joint-state or measured finger width is read. |
| `4:7` | Published cube-frame position from TF. |
| `7:11` | Cube-frame quaternion in `x, y, z, w` order. |
| `11:14` | Cube position minus TCP position. |

SAC prediction uses `deterministic=True`. The first three action values are clipped to `[-1, 1]`, exponentially filtered, and multiplied by `--max-step` to make Cartesian increments. Each motion target retains the current TCP orientation converted to roll, pitch, and yaw. The robot runner's rate, scaling, and controller behavior differ from the simulation control loop; deployment is an adaptation of the trained policy.

The fourth learned action can request closing, but a heuristic gate also requires horizontal alignment and a height threshold. Closing runs through the supplied controller. Once that routine returns and its wait completes without an exception, the runner sets `grasped=True`, even if the controller returned `False`; it does not verify contact, object capture, or lifting success. Policy-driven approach motion then stops. When enabled, a separate heuristic sends a single vertical lift of `--lift-height`, limited by `--z-max`, while retaining the current x/y and orientation.

### Perception and coordinate frames

RGB marker detection supplies a pose in camera coordinates. A separately measured static calibration lets TF2 express that pose in the robot base frame; the detector does not estimate this calibration. Robot TCP position and a synthetic gripper value complete the policy input. The diagram shows the actual zero-offset marker convention and the learned approach versus gated closing/lifting phases.

[![RealSense RGB and camera intrinsics feed ArUco pose estimation. A separately measured static calibration transforms the published marker origin into the robot base frame. Robot TCP position and synthetic gripper opening complete the 14-value SAC input, followed by learned approach and gated close/preset lift.](../assets/figures/perception_pipeline.svg)](../assets/figures/perception_pipeline.svg)

Click the diagram to view it at full size.

The detector consumes `/camera/camera/color/image_raw` and `/camera/camera/color/camera_info`, estimates the marker pose with `solvePnP`, publishes `/cube_pose_in_camera`, and broadcasts `camera_calib_optical_frame -> cube`.

The emitted camera frame is hardcoded to `camera_calib_optical_frame`; although the node stores the `CameraInfo` frame ID, it does not use that value in publication. The static calibration must describe the same optical coordinate system used by the RGB pose estimate. Renaming a TF frame does not perform a coordinate conversion.

Comments describe offsetting a marker on the cube's top face to the cube center, but the actual `p_marker_cube` vector is `[0, 0, 0]`. Consequently the published position is the **marker origin**, with marker orientation, rather than a geometric cube center. The cube-size field currently has no effect. Account for this discrepancy when comparing simulation and real observations; correcting it changes the deployed observation convention and needs validation.

The detector has no custom command-line options or declared ROS parameters for marker ID, dimensions, topics, or output frame. To adapt those settings, edit a local working copy deliberately; preserve the experiment source as evidence.

## Startup dependencies

A practical ordering follows the dependencies below. Camera startup and robot startup can proceed independently, but all required services and TFs must be ready before the policy runner.

| Stage | Terminal record | Readiness condition |
| --- | --- | --- |
| 1. Start the robot/MoveIt stack | [`terminal_7.txt`](terminal_logs/terminal_7.txt) | Robot TF, MoveGroup, IK, and required gripper services are available. |
| 2. Start the RGB camera | [`terminal_1.txt`](terminal_logs/terminal_1.txt) | RGB image and CameraInfo topics are being published. |
| 3. Publish measured camera calibration | [`terminal_4.txt`](terminal_logs/terminal_4.txt) | `fr3_link0 -> camera_calib_optical_frame` is present. |
| 4. Start the marker detector | [`terminal_2.txt`](terminal_logs/terminal_2.txt) | Camera calibration information has arrived and the marker is visible; `camera_calib_optical_frame -> cube` is published. |
| 5. Inspect imagery and transforms | [`terminal_3.txt`](terminal_logs/terminal_3.txt), [`terminal_6.txt`](terminal_logs/terminal_6.txt), [`terminal_8.txt`](terminal_logs/terminal_8.txt) | Optional `rqt`, plus current base-to-cube and base-to-TCP checks. |
| 6. Preview policy output | Adapted [`terminal_5.txt`](terminal_logs/terminal_5.txt) | Required TFs and MoveGroup service are available; preview flags below are used. For execution, joint states and gripper services are also needed. |
| 7. Run the reviewed physical configuration | [`terminal_5.txt`](terminal_logs/terminal_5.txt) | Operator has reviewed calibration, observations, targets, workspace, controller behavior, and robot readiness. |

The numbered terminal files are references, not a sequence to execute from 1 to 8. Keep launch processes running in separate terminals. Run the environment setup in every new terminal.

## Environment and placeholder commands

These are Linux Bash commands for the recorded ROS deployment environment. They are documentation, not an installer. Use the already configured ROS/FR3 workspace and a Python environment compatible with ROS imports and the archived SAC model.

Set these values in each terminal, or place your reviewed values in your own local environment file:

```bash
REPO_DIR="/absolute/path/to/fr3-sim2real-rl-grasping"
FR3_WS="/absolute/path/to/franka_ros2_ws"
MODEL_PATH="$REPO_DIR/train_sim/fr3v2_baseframe_lift_h0075_vertical/final_model/sac_panda_lift_final.zip"
ROBOT_IP="YOUR_ROBOT_IP"
CAMERA_SERIAL="YOUR_CAMERA_SERIAL"

source /opt/ros/humble/setup.bash
source "$FR3_WS/install/setup.bash"
```

Do not assume that installing Python packages into an arbitrary environment will make ROS's compiled `cv_bridge` bindings usable. The source records show a `robosuite` environment for the policy runner, but do not provide its environment specification.

### Robot planning stack — terminal 7

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:="$ROBOT_IP"
```

This is the recorded launch entry point. The repository does not supply the package or its configuration; confirm its behavior and supported arguments in your installed FR3 workspace.

### RGB camera — terminal 1

```bash
ros2 launch realsense2_camera rs_launch.py \
  serial_no:="_${CAMERA_SERIAL}" \
  enable_color:=true \
  enable_depth:=false \
  align_depth.enable:=false \
  rgb_camera.color_profile:=1280x720x30 \
  initial_reset:=true
```

The underscore-prefixed serial follows the supplied command. Recorded readiness output includes a discovered D455, a 1280×720 color stream at 30 FPS, and `RealSense Node Is Up!`. Depth is disabled because this perception path estimates pose from a known-size RGB marker.

### Camera calibration — terminal 4

Set seven **measured** calibration components. Translation is in meters; the quaternion is in x/y/z/w order. The command deliberately fails if any component is unset.

```bash
# Supply your own calibrated transform, for example by sourcing a local file:
CALIBRATION_FILE="/absolute/path/to/your/camera-calibration.env"
source "$CALIBRATION_FILE"
# That file must set CALIB_X, CALIB_Y, CALIB_Z,
# CALIB_QX, CALIB_QY, CALIB_QZ, and CALIB_QW.

ros2 run tf2_ros static_transform_publisher \
  --x "${CALIB_X:?Set measured CALIB_X}" \
  --y "${CALIB_Y:?Set measured CALIB_Y}" \
  --z "${CALIB_Z:?Set measured CALIB_Z}" \
  --qx "${CALIB_QX:?Set measured CALIB_QX}" \
  --qy "${CALIB_QY:?Set measured CALIB_QY}" \
  --qz "${CALIB_QZ:?Set measured CALIB_QZ}" \
  --qw "${CALIB_QW:?Set measured CALIB_QW}" \
  --frame-id fr3_link0 \
  --child-frame-id camera_calib_optical_frame
```

Expected startup output states that it is publishing from `fr3_link0` to `camera_calib_optical_frame`. Recalibrate if the camera placement changes.

### ArUco perception — terminal 2

```bash
python3 "$REPO_DIR/fr3_sim2real/aruco_cube_pose_node.py"
```

Expected output announces the subscribed image/CameraInfo topics, the pose topic, and `cube in camera_calib_optical_frame: p=(...), q=(...)` when the configured marker is detected. Pose publication needs both CameraInfo and a successful marker pose estimate.

### Inspection — terminals 3, 6, and 8

```bash
# Optional image/topic inspection, in a separate terminal:
rqt
```

```bash
# Cube pose in robot base coordinates:
ros2 run tf2_ros tf2_echo fr3_link0 cube
```

```bash
# TCP pose in robot base coordinates, in a separate terminal:
ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand_tcp
```

The records show an initial missing-frame message followed by translation, quaternion, RPY, and matrix output. A persistent missing-frame message is unresolved readiness, not successful validation. Check current positions against the physical setup and verify that pose updates follow object motion.

### Policy preview

```bash
python3 "$REPO_DIR/fr3_sim2real/real_fr3_policy_cartesian.py" \
  --model "$MODEL_PATH" \
  --rate 1 \
  --print-hz 1 \
  --max-step 0.001 \
  --min-cube-clearance 0.01
```

This preview omits `--execute`, `--reset`, `--enable-gripper`, and `--lift-after-grasp`. Expected runner messages include `Mode: DRY-RUN`, `Loaded model: ...`, `Required TFs are available.`, and lines beginning `mode=DRY` with observation-derived poses, action, delta, target, and grasp state. Preview instantiates the supplied controller and waits for ROS services; its constructor sends no motion or gripper goals.

### Recorded physical policy configuration — terminal 5

After reviewing the implementation limits and your hardware configuration, the recorded command settings can be used with the included final checkpoint as follows. This checkpoint choice is an example for the public archive; it does not identify the checkpoint used in the real video or historical terminal record.

```bash
python3 "$REPO_DIR/fr3_sim2real/real_fr3_policy_cartesian.py" \
  --model "$MODEL_PATH" \
  --rate 1 \
  --print-hz 1 \
  --max-step 0.001 \
  --min-cube-clearance 0.01 \
  --enable-gripper \
  --lift-after-grasp \
  --execute
```

This command permits robot and gripper motion. Its settings reproduce the recorded invocation, not universally safe values. In particular, `0.01` m is the record's clearance setting, while the script default is `0.10` m; both are relative to the detector's published marker origin, not the table surface.

Recorded milestones include controller service connections, `Mode: EXECUTE`, required TF availability, motion goals, a threshold-triggered close, `Gripper close sequence finished.`, and `Sending post-grasp lift ...`. The record also contains a `generator already executing` gripper error followed by a later close sequence, shared-memory transport warnings, and an `rcl_shutdown already called` error after Ctrl-C. These are observed issues, not expected success criteria. The script's flags and log transitions alone do not establish grasp reliability or task success.

## Verified policy CLI

These defaults come directly from `parse_args()` in `real_fr3_policy_cartesian.py`. Distances are in meters and time values in seconds where applicable.

| Option | Default | Effect |
| --- | --- | --- |
| `--model` | Required | Stable-Baselines3 SAC archive path. |
| `--execute` | Off | Required for reset, approach motion, gripper-close calls, and lift commands sent by the public runner. |
| `--enable-gripper` | Off | Enables heuristic grasp triggering from action and relative position. |
| `--lift-after-grasp` | Off | Requests one fixed-height lift after the internal grasp transition. |
| `--reset` | Off | Calls the supplied controller's reset routine at startup only with `--execute`; otherwise logs that reset was skipped. |
| `--base-frame` | `fr3_link0` | TF reference frame. |
| `--eef-frame` | `fr3_hand_tcp` | TCP frame queried in TF. |
| `--cube-frame` | `cube` | Object frame queried in TF. |
| `--gripper-open` | `1.0` | Synthetic pre-grasp observation value. |
| `--rate` | `5.0` | Requested policy-loop frequency in Hz; not a guaranteed motion-command rate. |
| `--print-hz` | `2.0` | Status print frequency; must be positive to avoid division by zero. |
| `--tf-timeout` | `0.2` | Timeout per TF lookup. |
| `--wait-tf-sec` | `10.0` | Startup wait window for required transforms. |
| `--max-step` | `0.002` | Per-axis Cartesian increment scale. |
| `--action-filter-alpha` | `0.25` | Action filter coefficient, clipped by code to `[0, 1]`. |
| `--min-cube-clearance` | `0.10` | Suppresses downward approach increments that would cross object z plus clearance. |
| `--grasp-xy-threshold` | `0.035` | Horizontal error must be smaller than this for grasp triggering. |
| `--grasp-height` | `0.085` | TCP z minus published object z must be smaller than this; the gate has no explicit lower-height bound. |
| `--close-action-threshold` | `-0.5` | Fourth action must be smaller than this for grasp triggering. |
| `--grasp-width` | `0.0` | Width passed to `close_gripper`. |
| `--gripper-speed` | `0.05` | Speed in m/s passed to the supplied controller's Franka Move action. |
| `--grasp-wait` | `0.8` | Delay after the gripper routine returns. |
| `--lift-height` | `0.075` | Fixed vertical lift increment. |
| `--x-min`, `--x-max` | `0.20`, `0.70` | X limits used for per-axis approach suppression. |
| `--y-min`, `--y-max` | `-0.30`, `0.30` | Y limits used for per-axis approach suppression. |
| `--z-min`, `--z-max` | `0.03`, `0.60` | Z limits for approach suppression; z maximum also caps lift target. |

## Dry-run and physical motion

In the public policy runner, every reset, gripper-close, approach, and lift command requires `--execute`. A direct call to its `send_motion()` or `send_lift()` also returns without sending motion when execution is disabled. Dry-run does not advance to `grasped=True` through the gripper routine. The preview command above initializes the supplied controller and connects to ROS; source inspection confirms that its constructor sends no motion or gripper goals. This does not validate the connected ROS workspace or extend the runner's execution gate to the controller's standalone demo.

These are publication-time fixes to problems in the research source: its reset path ignored `--execute`, its dry-run gripper routine could set the internal grasp flag, and its lift branch could then command `move_to(..., execute=True)` without execution permission. The public runner also removes the hardcoded controller import directory. The learned policy, thresholds, filtering, physical execution phases, and CLI defaults were otherwise retained. The research originals and recorded terminal behavior remain historical evidence.

Hardware-free tests extract the real `PolicyCartesian` class without importing ROS or connecting to a controller. They check dry-run with all motion flags, direct motion/lift calls, a pre-existing grasp flag, background-job scheduling, execution guards at worker entry, and the retained execution-enabled approach/close/lift behavior:

```bash
python -m unittest discover -s tests -p "test_real_execution_gate.py" -v
```

These tests verify policy-runner execution guards with stubs; they do not verify ROS integration, the supplied controller, or physical safety.

The approach bounds suppress an offending axis increment; they do not implement a complete collision or workspace safety controller. `--max-step` bounds each filtered action component, not total 3D distance, robot speed, acceleration, or the fixed lift. TF lookup uses the latest available transform without checking its age. Marker loss can therefore leave previously published information in the TF buffer, and this runner does not enforce perception freshness. Orientation, calibration, gripper state, and collision response require setup-specific review. Use the robot's normal supervised operating procedure and accessible stop mechanism for physical trials.

## Scope of the evidence

The terminal files retain commands and sample outputs from the experiment. A log showing a close routine and a lift request supports that those software phases occurred; it does not supply a trial count, success rate, force/contact measurement, or proof that every recorded command completed. The controller is supplied, but the configured ROS workspace, dependency lockfile, corrected perception-origin convention, and portable camera calibration are not. This repository remains a research artifact and deployment reference rather than a turnkey hardware package.
