# Simulation training and policy inspection

The simulation entry point is [`train_sim/train_osc_position_lowdim_workspace.py`](../train_sim/train_osc_position_lowdim_workspace.py). It wraps robosuite's MuJoCo `Lift` task as a Gymnasium environment and trains a Stable-Baselines3 Soft Actor-Critic (SAC) policy. The task is to approach a tabletop cube, grasp it, and hold it near a small target height with low motion.

The policy receives simulator state, rather than camera images. The real robot implementation constructs a vector with the same observation layout from robot feedback and estimated cube pose; its gripper and object-position measurements differ from simulation, as explained in the [deployment documentation](sim2real.md).

## Reproduction status

The training script, final checkpoint, evaluation archive, and TensorBoard event files are included. The original complete Python environment and robosuite/MuJoCo versions were not supplied. The `FR3v2` robot model referenced by the experiment is also not defined or registered in the supplied project files. Its original robosuite extension, robot/gripper assets, and any controller configuration changes are needed to reproduce that robot configuration.

The archived run directory is named `fr3v2_baseframe_lift_h0075_vertical`, and the thesis identifies `FR3v2` in its simulation setup. The checkpoint metadata records policy settings and dimensions but does not identify the robot model, table configuration, controller scaling, or complete original command. The directory name alone cannot recover that configuration.

`--robot Panda` is an alternative supported by the script, with different table dimensions. Running Panda is a separate simulation configuration; it does not reproduce the saved FR3v2 experiment. Even if a policy can be loaded with matching observation/action dimensions, that alone does not establish equivalence of robot geometry, controller scaling, or task dynamics.

The commands below are derived from the preserved source. They have not been run as simulation experiments during repository preparation. A complete simulation reproduction remains conditional on restoring the missing robot/environment dependencies.

## Recorded software environment

The final model ZIP archive contains the following plain-text environment metadata:

| Component | Recorded value |
| --- | --- |
| Python | 3.10.19 |
| Stable-Baselines3 | 2.7.1 |
| PyTorch | 2.10.0+cu128 |
| NumPy | 2.2.6 |
| Gymnasium | 1.2.3 |
| Cloudpickle | 3.1.2 |
| System | Linux 6.8.0-111-generic, x86_64, glibc 2.35 |
| GPU enabled | True |
| robosuite / MuJoCo | Not recorded in these archives |

[`requirements-sim.txt`](../requirements-sim.txt) pins the recorded core packages and lists additional packages used by the source, whose original versions are unknown. It is a partial environment reconstruction. Its `torch==2.10.0` pin does not select the recorded CUDA 12.8 build; the platform-specific PyTorch installation must be chosen separately if that build is required.

The script's type annotations require Python 3.10 or later. The original run used Linux. From the repository root, a starting environment can be prepared with:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-sim.txt
```

Before a FR3v2 run, restore its custom robosuite definition and assets in that environment. Verify that the installed package exposes `load_part_controller_config` and `refactor_composite_controller_config`, registers `FR3v2`, supplies the `robot0_right_center` base site, and uses the intended `OSC_POSITION` input/output scales. The script obtains those scales from the installed controller configuration; numerical meter-per-action limits are not stored in the supplied script or model metadata.

## Train a new policy

This example reconstructs the source's documented defaults with FR3v2 selected. It is not a recovered original training invocation, and it uses a new output directory so the archived run is retained. It requires the original FR3v2 integration:

```bash
python train_sim/train_osc_position_lowdim_workspace.py \
  --robot FR3v2 \
  --total-timesteps 500000 \
  --target-lift-height 0.075 \
  --max-lift-height 0.140 \
  --post-grasp-xy-action-scale 0.0 \
  --eval-freq 5000 \
  --work-dir runs/fr3v2_reproduction
```

For a separate Panda run, use `--robot Panda` and a different `--work-dir`. The script prints the selected robot, table size, action limits, observation layout, and workspace bounds during environment construction. The final checkpoint has a 14-dimensional observation and a 4-dimensional action in `[-1, 1]`.

The final policy and recorded evidence from a run use these paths:

```text
<work-dir>/
├── final_model/sac_panda_lift_final.zip
├── eval_logs/evaluations.npz
└── tb/SAC_<run-number>/events.out.tfevents.*
```

The final filename retains `panda` even when another robot is selected. It is not evidence that a checkpoint was trained on Panda.

## Visualize a saved policy

With the FR3v2 integration restored, render five deterministic episodes:

```bash
python train_sim/train_osc_position_lowdim_workspace.py \
  --robot FR3v2 \
  --visualize-only train_sim/fr3v2_baseframe_lift_h0075_vertical/final_model/sac_panda_lift_final.zip \
  --n-vis-episodes 5
```

To record new simulation footage:

```bash
python train_sim/train_osc_position_lowdim_workspace.py \
  --robot FR3v2 \
  --visualize-only train_sim/fr3v2_baseframe_lift_h0075_vertical/final_model/sac_panda_lift_final.zip \
  --n-vis-episodes 5 \
  --record-mp4 runs/fr3v2_final_preview.mp4 \
  --record-fps 20 \
  --record-camera frontview \
  --record-width 1280 \
  --record-height 720
```

Video recording uses offscreen rendering and `imageio`; the rendering backend must be available on the target machine. `--record-mp4` is applied in the `--visualize-only` path. Visualization episodes use the default success termination behavior; `--no-terminate-on-success` is applied to training and evaluation environments, and is not forwarded to `run_visualization`.

To inspect the included event logs:

```bash
tensorboard --logdir train_sim/fr3v2_baseframe_lift_h0075_vertical/tb
```

## Observation and action contract

The wrapper builds a `float32` vector in the robot base frame defined by the MuJoCo site `robot0_right_center`. For a world position `p`, base translation `b`, and base rotation `R`, it computes `R.T @ (p - b)`. Cube orientation is rotated into that same base frame.

| Python slice | Size | Meaning |
| --- | --- | --- |
| `0:3` | 3 | End-effector position in the robot base frame, meters |
| `3:4` | 1 | Mean absolute gripper joint position, meters |
| `4:7` | 3 | Cube position in the robot base frame, meters |
| `7:11` | 4 | Cube quaternion in the robot base frame, `x, y, z, w` |
| `11:14` | 3 | Cube position minus end-effector position, in the robot base frame, meters |

The gripper observation is computed as `mean(abs(robot0_gripper_qpos))`. It represents a compressed joint opening measure, rather than the total distance between fingers. No observation normalization wrapper is added by this script.

The archived action is `[a_x, a_y, a_z, a_gripper]` in `[-1, 1]^4`. The first three values are normalized translational commands for `OSC_POSITION`; the selected robosuite gripper interprets the final scalar. The policy does not output end-effector orientation commands.

The wrapper scales the translational command using the controller's `input_min`, `input_max`, `output_min`, and `output_max`. It forms a desired end-effector target using the previous **world-frame** end-effector position, clips that target to the cylindrical workspace, and converts the clipped meter delta back to normalized action coordinates. Observation coordinates and workspace coordinates therefore have different frame definitions.

After the first detected cube grasp, the wrapper latches `has_grasped_cube` for the rest of the episode. By default, it multiplies the intended x/y deltas by zero before workspace projection, encouraging a vertical lift. `--post-grasp-xy-action-scale` changes that factor; `--disable-post-grasp-vertical-only` disables the suppression. The latch is reset only when the episode resets.

## Environment and training settings

| Setting | Source value |
| --- | --- |
| Base task | robosuite `Lift`, shaped reward enabled |
| Robot CLI choices | `Panda` (default), `FR3v2` |
| Controller | `OSC_POSITION`, right arm |
| Observation mode | Object/state observations; camera observations disabled |
| Control frequency | 20 Hz |
| Episode horizon | 200 steps, nominally 10 simulation seconds |
| FR3v2 table full size | 1.4 × 0.9 × 0.05 m |
| Panda table full size | 0.8 × 0.8 × 0.05 m |
| Workspace radius, world frame | 0 to 0.85 m from the world x/y origin |
| Workspace z, world frame | 0.78 to 1.30 m |
| Algorithm / policy | SAC / `MlpPolicy` |
| Training steps | 500,000 by default |
| Learning rate | 0.0001 |
| Replay buffer capacity | 200,000 transitions |
| Batch size | 256 |
| Learning starts | 5,000 steps |
| Train frequency / gradient steps | 1 environment step / 1 gradient step |
| Discount factor | 0.99 |
| Target smoothing coefficient | 0.005 |
| Entropy coefficient | Automatically learned (`auto`) |
| Device | `auto` |
| Random seed | No fixed seed specified |

The script does not specify a custom network architecture, so its architecture follows the installed Stable-Baselines3 `MlpPolicy` defaults. The final checkpoint's archive JSON reports `use_sde=False`, one environment, and an automatic target entropy value of `-4.0`.

The table full size can be changed with `--table-size X Y Z`. Workspace constants remain fixed in the source; changing the table dimensions does not automatically update them.

## Success definitions

The metric called cube height is the cube body center's world z minus the arena table offset z. It is not the clearance of the cube's underside.

| Signal | Definition in the wrapper |
| --- | --- |
| `raw_lift_success` | Result of the installed robosuite `Lift._check_success()` |
| `controlled_lift` | Height within 0.025 m of the target, height no greater than the maximum, cube speed ≤ 0.04 m/s, and end-effector speed ≤ 0.06 m/s |
| `is_success` / held success | `controlled_lift` holds for at least 10 consecutive control steps |

With default settings, the target is 0.075 m, the accepted height band is 0.050–0.100 m, and the maximum is 0.140 m (`target + 0.065`). Ten control steps at 20 Hz correspond to 0.5 simulation seconds. A broken controlled-lift condition resets the consecutive-step count. Grasp contact and `raw_lift_success` are not additional Boolean requirements in this implementation's controlled/held tests.

By default, held success terminates the episode. The underlying robosuite `done` becomes Gymnasium `truncated`; this generally corresponds to the 200-step horizon. `--no-terminate-on-success` allows training/evaluation episodes to continue after held success.

The code returns zero speed if its cube or end-effector velocity lookup raises an exception. Those fallbacks can make a missing velocity API appear stable. A reproduction should verify that the selected robosuite version supplies both velocity measurements rather than treating the fallback as evidence of physical stability. The exact raw-lift threshold is supplied by the installed robosuite version, whose source/version is not archived here.

## Reward shaping

The returned reward is the installed robosuite shaped reward plus the following wrapper additions, minus a workspace boundary penalty. Let `h` be cube center height above the table, `h_target` the target, `h_max` the maximum, `v_cube` and `v_eef` the speed magnitudes, and `a_xyz` the action **after** clipping:

| Component | Formula / condition |
| --- | --- |
| Target height reward | `0.45 * exp(-40 * abs(h - h_target))` |
| Stable reward | `0.20` if controlled lift, otherwise zero |
| Height penalty | `8 * max(0, h - h_max) + max(0, h_target - h)` |
| Motion penalty | `0.45 * v_cube + 0.20 * v_eef` when `h > 0.04`, otherwise zero |
| Translation action penalty | `0.006 * norm(a_xyz)` |
| Vertical progress reward | `1.2 * max(0, h - h_previous)` after a grasp has been detected |
| x/y action penalty | `0.04 * norm(a_xy)` after a grasp has been detected |
| Cube x/y drift penalty | `2.0 * norm(cube_xy - grasp_anchor_xy)` after a grasp has been detected |
| Workspace boundary penalty | `0.1 * norm(desired_world_target - clipped_world_target)` |

The grasp anchor is the cube's world x/y position when grasp is first detected. The post-grasp reward/penalty terms remain active after that detection because the grasp flag is latched. The total reward is therefore more than a binary lift reward, and reward curves should be interpreted alongside the separately logged success signals.

## Evaluation and artifact interpretation

At the default interval of 5,000 training steps, the standard evaluation callback runs 10 deterministic episodes and saves their rewards/lengths in `evaluations.npz`. The final scheduled evaluation at 500,000 steps is the batch presented in this repository. The custom callback runs a separate set of 20 deterministic episodes to log held, raw-lift, and controlled-lift success rates in TensorBoard. These are distinct samples; rates from the additional 20 episodes should not be presented as though they were computed from the same 10 reward-evaluation episodes.

The training callback records success rates over the most recent 100 completed episodes. For each of its three rates, an episode counts as successful if that signal was seen at least once during the episode. The custom evaluation success callback uses the same per-episode “ever observed” rule; held success itself still requires the consecutive hold condition.

Plain JSON fields inside the final checkpoint archive record 500,000 training steps. These values were read without executing the model or deserializing its Python object payloads. They identify the saved checkpoint, rather than establishing a new success rate. See the [experiment results](results.md) for the final scheduled evaluation and the supplied videos.

The 500,000-step NPZ batch is the last scheduled training evaluation. It is not a separate evaluation of the saved final ZIP: the callback runs during rollout collection, before the gradient update that follows that rollout in [Stable-Baselines3 2.7.1](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/v2.7.1/stable_baselines3/common/off_policy_algorithm.py). This script saves the final policy after `learn` returns, then runs another ten-episode `evaluate_policy` call whose reward result is printed rather than saved to the NPZ.

For a complete reconstruction, obtain the original FR3v2 registration/assets, robosuite and MuJoCo versions or commits, modified controller files if any, and the original training invocation. A dependency freeze and explicit random-seed handling would be needed for more controlled future comparisons; the present source does not set the simulator or SAC seed.
