"""Apply generated sampling-rule functions to episodes and save results.

Uses the auto-generated rule functions from sampling_rules.py to compute
sampling sources for every episode. Saves results as JSON consumed by
the taxonomy viewer.

Usage:
    # Sample (2 episodes per task):
    python scripts/apply_rules_programmatic.py --episodes-per-task 2

    # All episodes:
    python scripts/apply_rules_programmatic.py --episodes-per-task 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    get_episode_indices,
    get_task_goal,
    get_timestep_indices,
)
from mme_vla_suite.shared.sampling_rules import TASK_RULES, get_sampling_sources

H5_DIR = Path("/coc/testnvme/shalbe3/robomme_data/robomme_data_h5")
OUTPUT_PATH = PROJECT_ROOT / "annotation_tool" / "sampling_results.json"


def extract_segments(ep_data, ts_indices, exec_start):
    segments = []
    current_label = None
    seg_start = None
    for t in ts_indices:
        is_demo = t < exec_start
        raw = ep_data[f"timestep_{t}"]["info"]["simple_subgoal"][()].decode()
        label = raw.strip().lower()
        phase = "demo" if is_demo else "exec"
        key = (phase, label)
        if key != current_label:
            if current_label is not None:
                segments.append({
                    "idx": len(segments),
                    "phase": current_label[0],
                    "label": current_label[1],
                    "start_frame": seg_start,
                    "end_frame": t - 1,
                    "num_frames": t - seg_start,
                })
            current_label = key
            seg_start = t
    if current_label is not None:
        segments.append({
            "idx": len(segments),
            "phase": current_label[0],
            "label": current_label[1],
            "start_frame": seg_start,
            "end_frame": ts_indices[-1],
            "num_frames": ts_indices[-1] - seg_start + 1,
        })
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--output", "-o", default=str(OUTPUT_PATH))
    parser.add_argument("--tasks", nargs="*", help="Subset of tasks")
    args = parser.parse_args()

    h5_files = sorted(H5_DIR.glob("record_dataset_*.h5"))
    results = {}
    stats = {"total": 0, "success": 0, "errors": 0}

    for h5_path in h5_files:
        task_name = h5_path.stem.replace("record_dataset_", "")

        if args.tasks and task_name not in args.tasks:
            continue

        if task_name not in TASK_RULES:
            print(f"[SKIP] {task_name}: no rule function")
            continue

        rule_func = TASK_RULES[task_name]
        print(f"\n{task_name}:")

        with h5py.File(str(h5_path), "r") as f:
            ep_indices = get_episode_indices(f)
            n_sample = min(args.episodes_per_task, len(ep_indices))
            sampled = ep_indices[:n_sample]

            for ep_idx in sampled:
                ep = f[f"episode_{ep_idx}"]
                task_goal = get_task_goal(ep)
                ts_indices = get_timestep_indices(ep)
                exec_start = first_execution_step(ep)
                segments = extract_segments(ep, ts_indices, exec_start)

                stats["total"] += 1
                try:
                    sampling_map = get_sampling_sources(task_name, segments, task_goal=task_goal)
                    # Convert int keys to str for JSON
                    sampling_map = {str(k): v for k, v in sampling_map.items()}
                    stats["success"] += 1
                except Exception as e:
                    print(f"  ep{ep_idx}: ERROR - {e}")
                    stats["errors"] += 1
                    continue

                ep_key = f"{task_name}_ep{ep_idx}"
                results[ep_key] = {
                    "task": task_name,
                    "episode_idx": ep_idx,
                    "task_goal": task_goal,
                    "segments": segments,
                    "sampling_map": sampling_map,
                }

                # Print summary
                exec_segs = [s for s in segments if s["phase"] == "exec" and "all tasks completed" not in s["label"]]
                print(f"  ep{ep_idx}: {len(exec_segs)} exec segs, {len(sampling_map)} mapped")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f_out:
        json.dump(results, f_out, indent=2)

    print(f"\n{stats['success']}/{stats['total']} episodes OK, {stats['errors']} errors")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
