# Publication notes

The repository was assembled from the supplied undergraduate research project. Original working files were left unchanged.

## Preserved experiment artifacts

The simulation training script, final model archive, evaluation NPZ, TensorBoard event files, supplied videos, final thesis PDF, and thesis source are retained. The README uses the two original thesis result figures. Derived evaluation reports, English workflow and perception diagrams, silent GIF previews, and a silent H.264 version of the real-robot recording were added for presentation and inspection. The full-duration previews keep the original playback speed.

The two videos were not accompanied by checkpoint identifiers. The repository does not assign either video to a specific checkpoint. The final model name is preserved as `sac_panda_lift_final.zip`; filenames alone do not identify robot configuration.

## Public deployment runner changes

The public `fr3_sim2real/real_fr3_policy_cartesian.py` differs from the supplied experiment script in these limited ways:

- Removed the hardcoded workstation path used to import `FR3Controller`. The controller module is included alongside the runner, and import errors explain the required ROS environment setup.
- Required `--execute` for reset, gripper closing, policy motion, and post-grasp lifting, including guards inside the motion helpers.
- Prevented a dry-run gripper request from setting an executed-grasp flag and scheduling a real lift.

These are execution-gate and portability changes for publication. They do not validate physical behavior of the controller, improve perception geometry, or constitute another physical-robot experiment. Hardware-free tests cover the affected execution paths. The original script's SHA-256 is recorded in the artifact manifest.

The separately supplied `FR3_ROS2_Workspace-master/src/my_controller/fr3_controller.py` is included unchanged, together with its workspace's Apache-2.0 license. Its constructor initializes ROS clients and waits for services without sending motion goals. Direct execution of that file starts its original motion demo; the documented deployment uses it as an imported module. The policy runner currently ignores its boolean move/gripper return values; this is retained and documented rather than treated as verified object capture.

## Historical terminal records

`docs/terminal_logs/terminal_1.txt` through `terminal_8.txt` retain the supplied command/output records with workstation usernames/hostnames, the camera serial, robot IP, and workstation-specific paths replaced by placeholders. Each file is labeled as historical evidence. Recorded errors are retained, because removing them would change the evidence.

Commands in the historical policy record can enable physical execution. The deployment guide provides a dependency-based startup procedure, separate preview and execution examples, and setup-specific placeholders. The original calibration numbers in the historical record describe the author's setup and are not a calibration for another installation.

## Research interpretation

The real system uses learned approach motion together with heuristic closing/lifting stages. Gripper state is synthetic and grasp completion is inferred from routine completion. The ArUco detector uses the marker origin as its published position; the code's cube-center comments do not match the zero offset. These are documented properties of the supplied code, rather than silently corrected experiment assumptions.

The README highlights the final scheduled training evaluation at 500,000 steps. The full NPZ evaluation history is retained, with ten episodes per checkpoint; the custom TensorBoard success callback uses another twenty-episode batch, with a delayed logger-dump step. No repeated real trial count, multi-seed result, or complete environment freeze was supplied.
