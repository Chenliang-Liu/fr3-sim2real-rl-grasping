# Experiments and evidence

This repository documents one saved simulation training run and a preliminary transfer demonstration on a Franka Research 3. Numerical results below are recalculated from the included evaluation archive; the videos illustrate observed behavior. No robot or simulation experiment was rerun when preparing this repository.

## What was recorded

The archive [`evaluations.npz`](../train_sim/fr3v2_baseframe_lift_h0075_vertical/eval_logs/evaluations.npz) contains:

| Array | Shape | Meaning |
|---|---|---|
| `timesteps` | `(100,)` | Training step at each saved evaluation |
| `results` | `(100, 10)` | Total reward in each evaluation episode |
| `ep_lengths` | `(100, 10)` | Episode length in environment control steps |
| `successes` | `(100, 10)` | Boolean `is_success` outcome |

There are 100 checkpoints, from 5,000 through 500,000 training steps at intervals of 5,000 steps, with 10 episodes per checkpoint. The archive holds 1,000 evaluation episodes and 144,313 evaluation environment steps. These batches evaluate changing policies during training; pooling them would not measure the final policy's success rate.

## Final scheduled simulation evaluation

The final scheduled training evaluation at **500,000 steps** recorded the following results:

| Metric | Recorded value |
|---|---:|
| Stable-hold outcomes | **9 / 10 episodes (90%)** |
| Mean total episode reward | 38.3995 |
| Episode reward population standard deviation | 19.3158 |
| Mean episode length | 59.2 control steps |

These results describe a 10-episode simulation evaluation batch from one training run. They are separate from the smoothed thesis curves below and do not estimate a real-robot success rate.

The final ZIP also records 500,000 steps, but the matching step is provenance, rather than proof that its exact saved weights produced this batch. In Stable-Baselines3 2.7.1, the last evaluation callback runs during rollout collection, before the final gradient update. The supplied script saves the final ZIP after `learn()` returns and then performs an additional `evaluate_policy()` call. That post-training console result was not supplied, so the exact saved final policy has no reported fresh evaluation here. The callback-before-gradient order is documented in the official [Stable-Baselines3 2.7.1 off-policy implementation](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/v2.7.1/stable_baselines3/common/off_policy_algorithm.py).

## Original thesis figures

The following figures are reproduced from the thesis as supplied. Their raw traces and smoothed curves summarize changes over training; the exact recorded final scheduled evaluation is reported in the table above.

![Original thesis figure: average reward and episode length over training](../assets/figures/fig_4_3_1_training_convergence.png)

The average reward generally improves over training, with fluctuations as the policy learns the cube-grasping and lifting task. Episode length begins at the 200-step cap and falls below that cap later in training, while continuing to vary. These curves show training rollout episode statistics. Their late episode lengths are therefore distinct from the 59.2-step mean in the separate final scheduled evaluation batch.

![Original thesis figure: raw lift, controlled lift, and held success rates](../assets/figures/fig_4_3_2_task_success_rate.png)

Raw lift success improves first. Controlled lifting and stable holding develop later and improve over training, with visible variation. The progression illustrates the increasingly demanding task requirements: lifting the cube, controlling its height and speed, then maintaining that state for consecutive control steps. The plotted curves use the custom success metrics described below; their smoothed endpoints are not the final scheduled 10-episode batch's 90% stable-hold rate.

## How success is defined

In the supplied [`train_osc_position_lowdim_workspace.py`](../train_sim/train_osc_position_lowdim_workspace.py), the environment sets `info["is_success"]` to `held_success`. The NPZ `successes` array therefore records the strict stable-hold criterion, rather than the original robosuite raw-lift criterion.

The default controlled-lift condition requires cube height above the table within 0.025 m of a 0.075 m target, cube speed at most 0.04 m/s, end-effector speed at most 0.06 m/s, and height below the configured maximum. Stable holding requires that condition for 10 consecutive control steps. At 20 Hz this corresponds to approximately 0.5 seconds. These are code defaults; the archive does not store a full training command or per-episode height and velocity traces.

The project also logs three custom TensorBoard metrics:

| Metric | Interpretation in the supplied callback |
|---|---|
| `eval/raw_lift_success_rate` | An episode ever satisfies the base environment's lift condition |
| `eval/controlled_lift_success_rate` | An episode ever satisfies the height and speed condition |
| `eval/held_success_rate` | An episode ever achieves consecutive stable holding |

The custom callback evaluates an **additional, separate batch of 20 episodes** after the standard 10-episode evaluation. Consequently, its TensorBoard success rates need not equal the NPZ's 10-episode rates. The generated evaluation CSV and JSON summarize only the NPZ archive; raw-lift and controlled-lift outcome arrays are absent from that archive. Raw TensorBoard event files remain available under the archived run's [`tb/`](../train_sim/fr3v2_baseframe_lift_h0075_vertical/tb/), and an independently generated [`tensorboard_summary.json`](../results/tensorboard_summary.json) records their scalar evidence.

### TensorBoard audit

Each event file was parsed independently with TensorBoard's event accumulator and an unlimited scalar history. The archived folders contain the following records:

| Folder | Available record |
|---|---|
| `SAC_1` | 100 standard evaluation points through 500,000 steps and 99 logged points for each custom success metric |
| `SAC_2` | No scalar tags |
| `SAC_3` | One rollout point at 800 steps; no evaluation metrics |

All three standard `SAC_1` evaluation tags (`eval/mean_reward`, `eval/mean_ep_length`, and `eval/success_rate`) match the NPZ's checkpoints and values within float32 rounding. The folders are not independent completed training seeds and are not pooled for statistics.

The custom callback records its 20-episode success metrics after the standard evaluation logger has already dumped its scalars. The custom values can therefore appear at later logger steps. The latest custom values are logged at step 495,532; a custom batch is not stored at logger step 500,000. Their complete per-tag first/latest/maximum values remain in [`tensorboard_summary.json`](../results/tensorboard_summary.json), and the original event files preserve their full histories. They are distinct from the final scheduled 10-episode NPZ batch.

## Thesis claims and available evidence

The original [thesis source](../thesis/robot-rl-thesis.tex) describes a 500,000-step run, 20 Hz control, a 200-step episode limit, three success criteria, and preliminary Sim2Real validation. The evaluation step range and observed maximum episode length agree with the reported training length and episode limit; control frequency and success definitions are documented by the code.

The original figures show improving training reward with fluctuations, episode lengths below the cap later in training, and gains in increasingly strict success criteria. The final scheduled evaluation separately records 9/10 stable holds at 500,000 steps. Together, the curves and final batch document learning progress while keeping their different measurement protocols explicit.

The thesis discusses the motivation for replacing a permissive lift criterion with stable holding. The supplied archive contains one run with the stricter criterion. It does not include independent baseline-versus-improved runs, a controlled ablation study, confidence intervals across training seeds, or a benchmark against other algorithms.

## Real-robot experiment

The real demonstration uses a Franka Research 3, Franka Hand, RealSense camera, ArUco cube pose estimation, ROS 2/TF2, and MoveIt 2. Cube and gripper observations are expressed in the robot base frame before policy inference. The recorded deployment combines policy-guided approach and a policy gripper trigger with a programmed vertical lift after closure; it demonstrates the integrated transfer workflow.

The [simulation video](../assets/videos/sim_grasp.mp4) and [real-robot video](../assets/videos/real_grasp.mp4) provide qualitative examples of grasping and lifting a cube. There is no supplied real-trial table with counts of successes and failures, no repeated disturbance protocol, and no real-world benchmark dataset. Accordingly, this repository does not claim a statistically established real-robot success rate or generalization to other objects and scenes. The thesis itself identifies the limited number of real experiments, reliance on an ArUco marker, and single-cube task as limitations.

## Reproduce the data summary

From the repository root, with NumPy installed:

```bash
python scripts/summarize_evaluations.py
```

This validates aligned, finite numeric arrays and writes [`evaluation_summary.json`](../results/evaluation_summary.json) and [`evaluation_history.csv`](../results/evaluation_history.csv). The JSON includes SHA-256 hashes, array shapes, checkpoint statistics, and plain model metadata. The CSV contains one row per recorded checkpoint. Reward and episode-length standard deviations use `ddof=0` within each batch.

For another archive or output location:

```bash
python scripts/summarize_evaluations.py --input path/to/evaluations.npz --output-dir path/to/results
```

The analysis uses `allow_pickle=False`, requires no MuJoCo/ROS environment, and sends no commands to a robot. Its saved-model metadata inspection reads only adjacent known model ZIPs' JSON entries; it does not load or reevaluate those models.

To also reproduce the TensorBoard scalar summary, install the `tensorboard` package and run:

```bash
python scripts/summarize_evaluations.py --include-tensorboard
```

An alternate event directory can be supplied with `--tensorboard-dir path/to/tb`. This output keeps event files separate, includes each source hash and scalar count/first/latest/maximum, and checks the standard evaluation tags against the NPZ summary.
