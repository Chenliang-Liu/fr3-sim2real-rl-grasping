# FR3 deployment scripts

This directory contains the ROS 2 perception node, SAC policy runner, and supplied FR3 MoveIt action controller. Eight sanitized historical terminal records for the Franka Research 3 deployment are archived in [`docs/terminal_logs`](../docs/terminal_logs/). The records are command-and-output references; their numbers do not prescribe startup order.

| File | Role |
| --- | --- |
| [`aruco_cube_pose_node.py`](aruco_cube_pose_node.py) | Detect an ArUco marker in RGB images and publish its pose as the `cube` TF frame. |
| [`real_fr3_policy_cartesian.py`](real_fr3_policy_cartesian.py) | Form a 14-value observation from TF, evaluate the trained SAC policy, and send Cartesian targets through the supplied `FR3Controller`. |
| [`fr3_controller.py`](fr3_controller.py) | MoveGroup action client, gripper move/homing clients, and robot-state subscriptions. Imported by the policy runner. |
| [`terminal_1.txt`](../docs/terminal_logs/terminal_1.txt) | RealSense RGB camera launch and startup output. |
| [`terminal_2.txt`](../docs/terminal_logs/terminal_2.txt) | ArUco node launch and pose output. |
| [`terminal_3.txt`](../docs/terminal_logs/terminal_3.txt) | Optional `rqt` launch. |
| [`terminal_4.txt`](../docs/terminal_logs/terminal_4.txt) | Camera-to-robot static calibration transform. |
| [`terminal_5.txt`](../docs/terminal_logs/terminal_5.txt) | Policy execution command and recorded runtime output. |
| [`terminal_6.txt`](../docs/terminal_logs/terminal_6.txt) | Base-to-cube TF check. |
| [`terminal_7.txt`](../docs/terminal_logs/terminal_7.txt) | FR3 MoveIt launch command. |
| [`terminal_8.txt`](../docs/terminal_logs/terminal_8.txt) | Base-to-tool-center-point TF check. |

See [the deployment guide](../docs/sim2real.md) for dependencies, placeholder-based commands, verified CLI options, and implementation limits.

## Before using the runner

The supplied controller has no sibling Python-module dependencies. It still needs a configured FR3 ROS workspace with MoveIt, Franka message/action interfaces, robot and gripper services, and TF. That workspace, the MoveIt configuration package, and a portable camera calibration are not included.

The initial policy preview should omit `--execute`, `--reset`, `--enable-gripper`, and `--lift-after-grasp`. The public runner requires `--execute` for every reset, gripper-close, approach, and lift command it sends. Preview initializes the supplied controller and waits for services; source inspection shows that its constructor sends no motion or gripper goals. Read the [dry-run behavior and physical-motion limits](../docs/sim2real.md#dry-run-and-physical-motion) before connecting the runner to hardware.

**Do not run `fr3_controller.py` as a readiness check:** its standalone entry point immediately runs a physical motion demo. The runner's `--execute` gate does not apply to that separate demo or direct controller method calls. In particular, the controller's `reset(execute=False)` still homes the gripper; the policy runner avoids this by skipping the entire reset call in dry-run.

For publication, the runner's reset/lift execution guards were fixed, dry-run no longer creates a synthetic successful grasp transition, and the workstation-specific controller import path was replaced with an actionable dependency error. The research originals were not modified; the terminal records are historical evidence and may reflect the original behavior. Hardware-free execution-gate tests are in [`tests/test_real_execution_gate.py`](../tests/test_real_execution_gate.py).

The real deployment combines a learned SAC approach policy with threshold-gated gripper closing and a fixed-height lift. The runner ignores controller Boolean failure returns; its `grasped=True` flag is an internal state transition, not a measurement that the object is securely grasped. The recorded gripper concurrency error and the controller's nested spinning behavior remain documented limitations.

The supplied workspace's Apache 2.0 license is retained in [`LICENSE-controller-Apache-2.0.txt`](LICENSE-controller-Apache-2.0.txt); see also [third-party notices](../THIRD_PARTY_NOTICES.md).
