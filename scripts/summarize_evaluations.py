#!/usr/bin/env python3
"""Summarize saved evaluation data without simulation or model deserialization.

Requires NumPy. All NPZ arrays are loaded with allow_pickle=False; model ZIPs
are inspected only by parsing their plain JSON `data` entry. Serialized policy
objects and Torch weights are never loaded. The optional TensorBoard summary
additionally requires the tensorboard package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT / "train_sim" / "fr3v2_baseframe_lift_h0075_vertical"
    / "eval_logs" / "evaluations.npz"
)


def load_evaluations(path: Path) -> dict[str, np.ndarray]:
    """Require nonempty, finite, aligned checkpoint-by-episode matrices."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"timesteps", "results", "ep_lengths"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Missing required arrays: {', '.join(sorted(missing))}")
        arrays = {key: archive[key].copy() for key in sorted(required)}
        if "successes" in archive.files:
            arrays["successes"] = archive["successes"].copy()

    timesteps = arrays["timesteps"]
    results = arrays["results"]
    lengths = arrays["ep_lengths"]
    if timesteps.ndim != 1 or timesteps.size == 0:
        raise ValueError("timesteps must be a nonempty 1D array")
    if timesteps.dtype.kind not in "iu" or np.any(timesteps < 0):
        raise ValueError("timesteps must contain nonnegative integers")
    if any(int(right) <= int(left) for left, right in zip(timesteps[:-1], timesteps[1:])):
        raise ValueError("timesteps must be strictly increasing")
    if results.ndim != 2 or results.shape[0] != len(timesteps) or results.shape[1] == 0:
        raise ValueError("results must have shape (n_checkpoints, n_episodes) with n_episodes > 0")
    if results.dtype.kind not in "iuf" or not np.all(np.isfinite(results)):
        raise ValueError("results must contain finite numeric rewards")
    if lengths.shape != results.shape or lengths.dtype.kind not in "iu" or np.any(lengths <= 0):
        raise ValueError("ep_lengths must match results and contain positive integer lengths")
    if "successes" in arrays:
        successes = arrays["successes"]
        if successes.shape != results.shape:
            raise ValueError("successes must have the same shape as results")
        if successes.dtype.kind not in "biu" or not np.all((successes == 0) | (successes == 1)):
            raise ValueError("successes must contain boolean or integer 0/1 outcomes")
        arrays["successes"] = successes.astype(bool)
    return arrays


def checkpoint_rows(arrays: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for index, timestep in enumerate(arrays["timesteps"]):
        reward = arrays["results"][index]
        length = arrays["ep_lengths"][index]
        row = {
            "training_timesteps": int(timestep),
            "evaluation_episodes": int(reward.size),
            "mean_reward": float(np.mean(reward)),
            "std_reward": float(np.std(reward, ddof=0)),
            "min_reward": float(np.min(reward)),
            "max_reward": float(np.max(reward)),
            "mean_episode_length": float(np.mean(length)),
            "std_episode_length": float(np.std(length, ddof=0)),
            "min_episode_length": int(np.min(length)),
            "max_episode_length": int(np.max(length)),
            "success_count": None,
            "success_rate": None,
        }
        if "successes" in arrays:
            row["success_count"] = int(np.sum(arrays["successes"][index]))
            row["success_rate"] = float(np.mean(arrays["successes"][index]))
        rows.append(row)
    return rows


def source_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def saved_model_metadata(input_path: Path, rows: list[dict]) -> list[dict]:
    """Inspect only known adjacent checkpoint JSON entries, without unpickling."""
    run_directory = input_path.resolve().parent.parent
    candidates = (
        run_directory / "best_model" / "best_model.zip",
        run_directory / "final_model" / "sac_panda_lift_final.zip",
    )
    metadata = []
    for path in candidates:
        if not path.is_file():
            continue
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
        fields = {key: data.get(key) for key in (
            "num_timesteps", "seed", "learning_rate", "buffer_size",
            "batch_size", "learning_starts", "tau", "gamma",
        )}
        # Exclude any encoded objects; these fields should be plain JSON values.
        if any(isinstance(value, (dict, list)) for value in fields.values()):
            raise ValueError(f"Unexpected structured checkpoint metadata in {path.name}")
        fields.update({"file": source_name(path), "sha256": sha256(path)})
        fields["matching_logged_evaluation"] = next(
            (row for row in rows if row["training_timesteps"] == fields["num_timesteps"]),
            None,
        )
        if path.name == "best_model.zip":
            fields["evaluation_relation"] = (
                "The supplied EvalCallback saves this reward-selected model during the evaluation callback; "
                "its matching logged evaluation is the tested callback checkpoint. This script verifies only "
                "JSON step metadata and logs, without loading the weights or rerunning evaluation."
            )
        else:
            fields["evaluation_relation"] = (
                "The matching logged training step is provenance only, not an exact evaluation of this saved final ZIP. "
                "In the supplied Stable-Baselines3 2.7.1 training flow, the last scheduled EvalCallback occurs "
                "before the final gradient update; the final ZIP is saved after learn returns. The separate "
                "post-training evaluate_policy console result was not supplied."
            )
        metadata.append(fields)
    return metadata


def summarize(input_path: Path, arrays: dict[str, np.ndarray], rows: list[dict]) -> dict:
    timesteps = arrays["timesteps"]
    intervals = sorted({int(right) - int(left) for left, right in zip(timesteps[:-1], timesteps[1:])})
    summary = {
        "schema_version": 1,
        "source_file": source_name(input_path),
        "source_sha256": sha256(input_path),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "evaluation_checkpoints": len(rows),
        "episodes_per_checkpoint": int(arrays["results"].shape[1]),
        "total_logged_evaluation_episodes": int(arrays["results"].size),
        "total_logged_evaluation_environment_steps": int(np.sum(arrays["ep_lengths"])),
        "first_training_timestep": int(timesteps[0]),
        "last_training_timestep": int(timesteps[-1]),
        "checkpoint_intervals": intervals,
        "standard_deviation": "Population standard deviation over each evaluation batch (ddof=0)",
        "success_semantics": (
            "Archive successes are the environment's is_success flag. In the supplied training code, "
            "this is held_success: 10 consecutive controlled-lift steps. They are not raw-lift outcomes."
            if "successes" in arrays else "No successes array was present; success rates are unavailable."
        ),
        "first_evaluation": rows[0],
        "final_recorded_evaluation": rows[-1],
        "highest_success_rate": None,
        "highest_success_rate_timesteps": [],
        "saved_models": saved_model_metadata(input_path, rows),
        "limitations": [
            "These are stored evaluation batches from one run, not newly executed experiments.",
            "Each checkpoint uses a small episode batch; no multi-seed estimate is available.",
            "A matching logged evaluation is not a fresh model re-evaluation; consult each saved model's evaluation_relation.",
            "The 500,000-step scheduled training evaluation precedes the final gradient update in Stable-Baselines3 2.7.1; it does not evaluate the exact weights saved after learn returns.",
            "The training script separately evaluates the saved final policy after training, but its console output was not supplied in this repository's evidence.",
            "The separately logged TensorBoard custom success metrics use additional 20-episode batches.",
            "This archive does not contain raw-lift or controlled-lift outcome arrays.",
            "Real-robot success rates cannot be computed from the provided demonstrations.",
        ],
    }
    if "successes" in arrays:
        best_rate = max(row["success_rate"] for row in rows)
        summary["highest_success_rate"] = best_rate
        summary["highest_success_rate_timesteps"] = [
            row["training_timesteps"] for row in rows if row["success_rate"] == best_rate
        ]
    return summary


def summarize_tensorboard(directory: Path, rows: list[dict]) -> dict:
    """Keep event files separate and compare standard eval tags with NPZ rows."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise ValueError("TensorBoard summary requires the tensorboard package") from error
    paths = sorted(directory.rglob("events.out.tfevents.*"))
    if not paths:
        raise ValueError(f"No TensorBoard event files found in {directory}")
    files = []
    standard_tags = {
        "eval/mean_reward": "mean_reward",
        "eval/mean_ep_length": "mean_episode_length",
        "eval/success_rate": "success_rate",
    }
    for path in paths:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        tags = {}
        comparisons = {}
        for tag in accumulator.Tags()["scalars"]:
            events = accumulator.Scalars(tag)
            if not events:
                continue
            values = np.asarray([event.value for event in events], dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Nonfinite TensorBoard scalar values in {path.name}: {tag}")
            highest_index = int(np.argmax(values))
            tags[tag] = {
                "count": len(events),
                "first": {"logged_step": int(events[0].step), "value": float(values[0])},
                "latest": {"logged_step": int(events[-1].step), "value": float(values[-1])},
                "minimum": float(np.min(values)),
                "maximum": float(values[highest_index]),
                "first_maximum_logged_step": int(events[highest_index].step),
            }
            if tag in standard_tags:
                column = standard_tags[tag]
                comparable = all(row[column] is not None for row in rows)
                match = bool(
                    comparable and len(events) == len(rows)
                    and all(event.step == row["training_timesteps"] for event, row in zip(events, rows))
                    and np.allclose(values, [row[column] for row in rows], rtol=1e-6, atol=1e-6)
                )
                comparisons[tag] = {
                    "matches_npz_checkpoints_and_values": match,
                    "comparison_tolerance": "rtol=1e-6, atol=1e-6 (TensorBoard stores float32 scalars)",
                }
        files.append({
            "file": source_name(path),
            "sha256": sha256(path),
            "run_directory": path.parent.name,
            "scalar_tag_count": len(tags),
            "scalar_tags": tags,
            "npz_comparisons": comparisons,
        })
    return {
        "schema_version": 1,
        "source_directory": source_name(directory),
        "event_files": files,
        "custom_success_batch_episodes_in_supplied_code": 20,
        "standard_evaluation_batch_episodes_in_supplied_code": 10,
        "interpretation": [
            "Event files are summarized independently; SAC folders are not pooled as training seeds.",
            "Standard eval/success_rate corresponds to the 10-episode NPZ successes batch.",
            "Custom eval/{held,controlled_lift,raw_lift}_success_rate tags come from separate 20-episode batches.",
            "Custom metrics are recorded after the standard evaluation logger dump and can appear at later logged steps.",
            "Latest custom success scalars therefore do not establish the final checkpoint's performance.",
            "Maximum values identify observed batches, not uncertainty bounds or independent benchmark results.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input evaluations.npz archive")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results", help="Directory for JSON and CSV summaries")
    parser.add_argument("--include-tensorboard", action="store_true", help="Also summarize the input run's tb/ files (requires tensorboard)")
    parser.add_argument("--tensorboard-dir", type=Path, help="Optional TensorBoard directory; specifying it also enables its summary")
    args = parser.parse_args()
    try:
        arrays = load_evaluations(args.input)
        rows = checkpoint_rows(arrays)
        summary = summarize(args.input, arrays, rows)
        tb_summary = None
        if args.include_tensorboard or args.tensorboard_dir is not None:
            tb_directory = args.tensorboard_dir or args.input.resolve().parent.parent / "tb"
            tb_summary = summarize_tensorboard(tb_directory, rows)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Evaluation summary failed: {error}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "evaluation_summary.json"
    csv_path = args.output_dir / "evaluation_history.csv"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if tb_summary is not None:
        tb_path = args.output_dir / "tensorboard_summary.json"
        tb_path.write_text(json.dumps(tb_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"TensorBoard JSON: {tb_path}")
    print(f"Validated {len(rows)} checkpoints, {summary['total_logged_evaluation_episodes']} episodes")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"Final scheduled training evaluation: {json.dumps(rows[-1], allow_nan=False)}")


if __name__ == "__main__":
    main()
