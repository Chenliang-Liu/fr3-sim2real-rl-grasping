# Eight-terminal experiment records

These text files contain the author's recorded commands and observed console output for the real-robot experiment. Workstation/usernames, local home paths, camera serial, and robot IP have been replaced with placeholders; each file has an added historical-record header. Calibration values and observed errors are retained.

| Record | Role |
| --- | --- |
| [terminal_1.txt](terminal_1.txt) | RealSense RGB camera launch |
| [terminal_2.txt](terminal_2.txt) | ArUco marker/cube-frame pose publisher |
| [terminal_3.txt](terminal_3.txt) | `rqt` inspection |
| [terminal_4.txt](terminal_4.txt) | Measured static camera calibration transform |
| [terminal_5.txt](terminal_5.txt) | SAC runner, gripper, and programmed lift |
| [terminal_6.txt](terminal_6.txt) | Base-to-cube TF check |
| [terminal_7.txt](terminal_7.txt) | FR3 MoveIt/robot stack launch |
| [terminal_8.txt](terminal_8.txt) | Base-to-TCP TF check |

Numbering identifies terminals, not execution order. The policy runner depends on the camera, TF transforms, and robot services. Follow the [deployment guide](../sim2real.md) for startup order and preview/execution commands. Historical output includes errors and should not be read as a guarantee that every command completed successfully.
