"""Merge per-task preprocessed datasets into a single dataset.

After running build_dataset_single_task.py in parallel (one task per GPU),
this script renumbers episodes and samples into a unified dataset.

Uses symlinks for episode feature dirs (instant) and hardlinks for pkl
files (no data copy), only rewriting pkls that need epis_idx updates.

Usage:
    python scripts/merge_task_datasets.py \
        --per_task_dir data/robomme_preprocessed_data/per_task \
        --output_dir data/robomme_preprocessed_data
"""

import argparse
import json
import os
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_task_dir", type=str, required=True,
                        help="Directory containing per-task subdirs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Final merged output directory")
    args = parser.parse_args()

    per_task_dir = os.path.abspath(args.per_task_dir)
    output_dir = os.path.abspath(args.output_dir)

    output_features = os.path.join(output_dir, "features")
    output_data = os.path.join(output_dir, "data")
    output_meta = os.path.join(output_dir, "meta")

    # Clean previous merged output (but not per_task/)
    for d in (output_features, output_data, output_meta):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    task_dirs = sorted([
        os.path.join(per_task_dir, d)
        for d in os.listdir(per_task_dir)
        if os.path.isdir(os.path.join(per_task_dir, d))
    ])

    global_episode_idx = 0
    global_exec_sample_id = 0
    global_total_sample_id = 0

    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        task_features = os.path.join(task_dir, "features")
        task_data = os.path.join(task_dir, "data")
        task_meta = os.path.join(task_dir, "meta")

        if not os.path.exists(task_features):
            print(f"Skipping {task_name}: no features/ dir")
            continue

        # Load task stats
        stats_path = os.path.join(task_meta, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                task_stats = json.load(f)
        else:
            task_stats = {}

        # List episodes and samples
        task_episode_dirs = sorted([
            d for d in os.listdir(task_features)
            if d.startswith("episode_")
        ], key=lambda x: int(x.split("_")[1]))

        task_sample_files = sorted([
            f for f in os.listdir(task_data)
            if f.endswith(".pkl")
        ], key=lambda x: int(x.split(".")[0])) if os.path.exists(task_data) else []

        episode_offset = global_episode_idx

        # Symlink episode dirs (instant, no copy)
        for ep_dir_name in task_episode_dirs:
            src = os.path.join(task_features, ep_dir_name)
            dst = os.path.join(output_features, f"episode_{global_episode_idx}")
            os.symlink(os.path.abspath(src), dst)
            global_episode_idx += 1

        # Hardlink pkl files with renumbering (no data copy)
        # epis_idx inside each pkl is already 0-based within this task,
        # so we add episode_offset
        needs_rewrite = (episode_offset != 0)

        if needs_rewrite:
            import pickle
            for sample_file in task_sample_files:
                src = os.path.join(task_data, sample_file)
                dst = os.path.join(output_data, f"{global_exec_sample_id}.pkl")

                with open(src, "rb") as f:
                    sample = pickle.load(f)
                old_epis_idx = int(sample["epis_idx"].item())
                sample["epis_idx"] = np.array([old_epis_idx + episode_offset], dtype=np.int32)
                with open(dst, "wb") as f:
                    pickle.dump(sample, f)

                global_exec_sample_id += 1
        else:
            # First task: just hardlink, no rewrite needed
            for sample_file in task_sample_files:
                src = os.path.join(task_data, sample_file)
                dst = os.path.join(output_data, f"{global_exec_sample_id}.pkl")
                os.link(src, dst)
                global_exec_sample_id += 1

        global_total_sample_id += task_stats.get("total_samples", 0)

        print(f"{task_name}: {len(task_episode_dirs)} episodes, "
              f"{len(task_sample_files)} exec samples "
              f"(offset={episode_offset})")

    # Write merged stats
    merged_stats = {
        "execution_samples": global_exec_sample_id,
        "total_samples": global_total_sample_id,
    }
    with open(os.path.join(output_meta, "stats.json"), "w") as f:
        json.dump(merged_stats, f, indent=2)

    print(f"\nMerged: {global_episode_idx} episodes, "
          f"{global_exec_sample_id} exec samples, "
          f"{global_total_sample_id} total samples")


if __name__ == "__main__":
    main()
