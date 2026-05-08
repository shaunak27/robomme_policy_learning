"""Scan all RoboMME H5 episodes and build a per-task subtask taxonomy.

For each of the 16 tasks, produces:
  - Unique subtask labels and their aliases
  - Typical ordering (median position in the sequence)
  - Transition graph (which subtask follows which, with counts)
  - Frame coverage (% of episode frames each subtask occupies, on average)
  - Dependencies (A always precedes B across episodes)

Outputs a single JSON file consumed by the taxonomy viewer Flask app.

Usage:
    python scripts/build_subtask_taxonomy.py [--output taxonomy.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    get_episode_indices,
    get_task_goal,
    get_timestep_indices,
)

H5_DIR = Path("/coc/testnvme/shalbe3/robomme_data/robomme_data_h5")


def normalize_label(label: str) -> str:
    """Lowercase and strip whitespace for consistent comparison."""
    return label.strip().lower()


def extract_episode_subtask_info(
    ep_data: h5py.Group, ts_indices: list[int], exec_start: int, include_demo: bool
) -> dict:
    """Extract subtask sequence and per-subtask frame counts for one episode.

    Returns dict with:
      - exec_sequence: ordered list of (label, frame_count) for exec frames
      - demo_sequence: ordered list of (label, frame_count) for demo frames (if include_demo)
      - total_exec_frames: int
      - total_demo_frames: int
    """
    result = {"exec_sequence": [], "demo_sequence": [], "total_exec_frames": 0, "total_demo_frames": 0}

    for phase, is_demo_phase in [("demo", True), ("exec", False)]:
        if is_demo_phase and not include_demo:
            continue

        current_label = None
        current_count = 0
        sequence = []
        total = 0

        for t in ts_indices:
            if is_demo_phase and t >= exec_start:
                break
            if not is_demo_phase and t < exec_start:
                continue

            raw = ep_data[f"timestep_{t}"]["info"]["simple_subgoal"][()].decode()
            label = normalize_label(raw)
            total += 1

            if label != current_label:
                if current_label is not None:
                    sequence.append((current_label, current_count))
                current_label = label
                current_count = 1
            else:
                current_count += 1

        if current_label is not None:
            sequence.append((current_label, current_count))

        key = f"{phase}_sequence"
        result[key] = sequence
        result[f"total_{phase}_frames"] = total

    return result


def build_task_taxonomy(h5_path: str, task_name: str) -> dict:
    """Build taxonomy for a single task by scanning all its episodes."""
    # Accumulators
    all_exec_sequences: list[list[str]] = []
    all_demo_sequences: list[list[str]] = []
    subtask_frame_fractions: defaultdict[str, list[float]] = defaultdict(list)
    demo_subtask_frame_fractions: defaultdict[str, list[float]] = defaultdict(list)
    transition_counts: Counter = Counter()
    demo_transition_counts: Counter = Counter()
    subtask_positions: defaultdict[str, list[float]] = defaultdict(list)
    demo_subtask_positions: defaultdict[str, list[float]] = defaultdict(list)
    task_goal = None
    has_demo = False

    with h5py.File(h5_path, "r") as f:
        episode_indices = get_episode_indices(f)
        num_episodes = len(episode_indices)

        for ep_idx in episode_indices:
            ep = f[f"episode_{ep_idx}"]
            if task_goal is None:
                task_goal = get_task_goal(ep)
            ts_indices = get_timestep_indices(ep)
            exec_start = first_execution_step(ep)

            if exec_start > 0:
                has_demo = True

            info = extract_episode_subtask_info(ep, ts_indices, exec_start, include_demo=True)

            # --- Exec phase ---
            exec_seq = info["exec_sequence"]
            total_exec = info["total_exec_frames"]
            exec_labels = [label for label, _ in exec_seq]
            all_exec_sequences.append(exec_labels)

            # Frame coverage
            for label, count in exec_seq:
                if total_exec > 0:
                    subtask_frame_fractions[label].append(count / total_exec)

            # Normalized position (0..1 within the sequence)
            for i, label in enumerate(exec_labels):
                if len(exec_labels) > 1:
                    subtask_positions[label].append(i / (len(exec_labels) - 1))
                else:
                    subtask_positions[label].append(0.5)

            # Transitions
            for i in range(len(exec_labels) - 1):
                transition_counts[(exec_labels[i], exec_labels[i + 1])] += 1

            # --- Demo phase ---
            demo_seq = info["demo_sequence"]
            total_demo = info["total_demo_frames"]
            demo_labels = [label for label, _ in demo_seq]
            if demo_labels:
                all_demo_sequences.append(demo_labels)

                for label, count in demo_seq:
                    if total_demo > 0:
                        demo_subtask_frame_fractions[label].append(count / total_demo)

                for i, label in enumerate(demo_labels):
                    if len(demo_labels) > 1:
                        demo_subtask_positions[label].append(i / (len(demo_labels) - 1))
                    else:
                        demo_subtask_positions[label].append(0.5)

                for i in range(len(demo_labels) - 1):
                    demo_transition_counts[(demo_labels[i], demo_labels[i + 1])] += 1

    # --- Aggregate per-subtask stats (exec) ---
    all_exec_labels = set()
    for seq in all_exec_sequences:
        all_exec_labels.update(seq)

    # Filter out "all tasks completed" sentinel
    sentinel_labels = {"all tasks completed", "complete", "completed"}
    all_exec_labels -= sentinel_labels

    subtask_stats = {}
    for label in sorted(all_exec_labels):
        episodes_with = sum(1 for seq in all_exec_sequences if label in seq)
        fracs = subtask_frame_fractions.get(label, [])
        positions = subtask_positions.get(label, [])
        subtask_stats[label] = {
            "frequency": round(episodes_with / num_episodes, 4),
            "episode_count": episodes_with,
            "avg_coverage": round(float(np.mean(fracs)) if fracs else 0, 4),
            "std_coverage": round(float(np.std(fracs)) if fracs else 0, 4),
            "median_position": round(float(np.median(positions)) if positions else 0, 4),
            "avg_position": round(float(np.mean(positions)) if positions else 0, 4),
        }

    # --- Demo subtask stats ---
    all_demo_labels = set()
    for seq in all_demo_sequences:
        all_demo_labels.update(seq)
    all_demo_labels -= sentinel_labels

    demo_subtask_stats = {}
    for label in sorted(all_demo_labels):
        episodes_with = sum(1 for seq in all_demo_sequences if label in seq)
        fracs = demo_subtask_frame_fractions.get(label, [])
        positions = demo_subtask_positions.get(label, [])
        demo_subtask_stats[label] = {
            "frequency": round(episodes_with / max(len(all_demo_sequences), 1), 4),
            "episode_count": episodes_with,
            "avg_coverage": round(float(np.mean(fracs)) if fracs else 0, 4),
            "std_coverage": round(float(np.std(fracs)) if fracs else 0, 4),
            "median_position": round(float(np.median(positions)) if positions else 0, 4),
            "avg_position": round(float(np.mean(positions)) if positions else 0, 4),
        }

    # --- Transitions ---
    transitions = [
        {"from": a, "to": b, "count": c}
        for (a, b), c in transition_counts.most_common()
        if a not in sentinel_labels and b not in sentinel_labels
    ]
    demo_transitions = [
        {"from": a, "to": b, "count": c}
        for (a, b), c in demo_transition_counts.most_common()
        if a not in sentinel_labels and b not in sentinel_labels
    ]

    # --- Dependencies: A always precedes B ---
    dependencies = []
    sorted_labels = sorted(all_exec_labels)
    for i, a in enumerate(sorted_labels):
        for b in sorted_labels[i + 1 :]:
            a_before_b = 0
            b_before_a = 0
            both_present = 0
            for seq in all_exec_sequences:
                filtered = [l for l in seq if l in {a, b}]
                if a in filtered and b in filtered:
                    both_present += 1
                    first_a = filtered.index(a)
                    first_b = filtered.index(b)
                    if first_a < first_b:
                        a_before_b += 1
                    else:
                        b_before_a += 1
            if both_present >= 5:  # need sufficient co-occurrence
                if a_before_b / both_present >= 0.95:
                    dependencies.append({"before": a, "after": b, "strength": round(a_before_b / both_present, 3), "co_occurrences": both_present})
                elif b_before_a / both_present >= 0.95:
                    dependencies.append({"before": b, "after": a, "strength": round(b_before_a / both_present, 3), "co_occurrences": both_present})

    # --- Canonical sequence (most common ordering) ---
    # Use the most frequent unique sequence
    seq_counter = Counter(tuple(l for l in seq if l not in sentinel_labels) for seq in all_exec_sequences)
    canonical_sequences = [
        {"sequence": list(seq), "count": count}
        for seq, count in seq_counter.most_common(10)
    ]

    demo_seq_counter = Counter(tuple(l for l in seq if l not in sentinel_labels) for seq in all_demo_sequences)
    demo_canonical = [
        {"sequence": list(seq), "count": count}
        for seq, count in demo_seq_counter.most_common(5)
    ]

    return {
        "task_name": task_name,
        "task_goal": task_goal,
        "num_episodes": num_episodes,
        "has_video_demo": has_demo,
        "exec": {
            "subtasks": subtask_stats,
            "transitions": transitions,
            "dependencies": dependencies,
            "canonical_sequences": canonical_sequences,
        },
        "demo": {
            "subtasks": demo_subtask_stats,
            "transitions": demo_transitions,
            "canonical_sequences": demo_canonical,
        } if has_demo else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Build subtask taxonomy from RoboMME H5 data")
    parser.add_argument("--output", "-o", default=str(PROJECT_ROOT / "annotation_tool" / "taxonomy.json"))
    parser.add_argument("--h5-dir", default=str(H5_DIR))
    args = parser.parse_args()

    h5_dir = Path(args.h5_dir)
    h5_files = sorted(h5_dir.glob("record_dataset_*.h5"))
    print(f"Found {len(h5_files)} H5 files in {h5_dir}")

    taxonomy = {}
    for h5_path in h5_files:
        task_name = h5_path.stem.replace("record_dataset_", "")
        print(f"\nProcessing {task_name} ({h5_path.name})...")
        task_data = build_task_taxonomy(str(h5_path), task_name)
        taxonomy[task_name] = task_data
        n_sub = len(task_data["exec"]["subtasks"])
        n_trans = len(task_data["exec"]["transitions"])
        print(f"  {task_data['num_episodes']} episodes, {n_sub} unique subtasks, {n_trans} transitions")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"\nTaxonomy saved to {output_path}")


if __name__ == "__main__":
    main()
