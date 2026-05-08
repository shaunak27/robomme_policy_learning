"""Extract keyframes from video demonstration subtask segments.

Video demos are perfect demonstrations — no failures, no TOPReward needed.
We just extract subtask segments and pick 5 keyframes per segment using
visual saliency (same logic as the TOPReward pipeline but without scoring).

Usage:
    python scripts/extract_videodemo_keyframes.py \
        --raw_data_path data/robomme_data_h5 \
        --output_dir data/topreward_full
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    get_episode_indices,
    get_task_goal,
    get_timestep_indices,
)
from mme_vla_suite.pseudo_labeling.episode_sampler import SubtaskSegment
from mme_vla_suite.pseudo_labeling.success_interval import select_keyframes


# ---------------------------------------------------------------------------
# Extract video demo subtask segments
# ---------------------------------------------------------------------------


def extract_videodemo_segments(
    h5_path: str, episode_idx: int, env_id: str,
) -> list[SubtaskSegment]:
    """Extract subtask segments from the video demo portion of an episode."""
    segments: list[SubtaskSegment] = []
    with h5py.File(h5_path, "r") as f:
        ep = f[f"episode_{episode_idx}"]
        task_goal = get_task_goal(ep, lower=True)
        exec_start = first_execution_step(ep)

        if exec_start == 0:
            return []  # No video demo for this episode

        current_label: str | None = None
        seg_start: int | None = None
        all_static = True  # Track if entire demo is "static"

        for t in range(exec_start):
            ts = ep[f"timestep_{t}"]
            label = ts["info"]["simple_subgoal"][()].decode().lower()

            # Skip "static" segments at the end of video demos
            if "static" in label:
                if current_label is not None and seg_start is not None:
                    segments.append(
                        SubtaskSegment(
                            episode_h5_path=h5_path,
                            episode_idx=episode_idx,
                            task_goal=task_goal,
                            subtask_label=current_label,
                            start_frame=seg_start,
                            end_frame=t - 1,
                            exec_start_idx=exec_start,
                            env_id=env_id,
                        )
                    )
                current_label = None
                seg_start = None
                continue

            all_static = False
            if label != current_label:
                if current_label is not None and seg_start is not None:
                    segments.append(
                        SubtaskSegment(
                            episode_h5_path=h5_path,
                            episode_idx=episode_idx,
                            task_goal=task_goal,
                            subtask_label=current_label,
                            start_frame=seg_start,
                            end_frame=t - 1,
                            exec_start_idx=exec_start,
                            env_id=env_id,
                        )
                    )
                current_label = label
                seg_start = t

        # Close final segment
        if current_label is not None and seg_start is not None:
            segments.append(
                SubtaskSegment(
                    episode_h5_path=h5_path,
                    episode_idx=episode_idx,
                    task_goal=task_goal,
                    subtask_label=current_label,
                    start_frame=seg_start,
                    end_frame=exec_start - 1,
                    exec_start_idx=exec_start,
                    env_id=env_id,
                )
            )

        # If entire demo is "static" (e.g. VideoUnmask showing scene
        # before masking), treat the whole demo as one context segment.
        if all_static and exec_start >= 5:
            segments.append(
                SubtaskSegment(
                    episode_h5_path=h5_path,
                    episode_idx=episode_idx,
                    task_goal=task_goal,
                    subtask_label="scene context",
                    start_frame=0,
                    end_frame=exec_start - 1,
                    exec_start_idx=exec_start,
                    env_id=env_id,
                )
            )

    # Filter short segments
    segments = [s for s in segments if s.end_frame - s.start_frame >= 4]
    return segments


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_segment_frames(
    segment: SubtaskSegment,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load frames and states for a segment."""
    frames: list[np.ndarray] = []
    states_list: list[np.ndarray] = []

    with h5py.File(segment.episode_h5_path, "r") as f:
        ep = f[f"episode_{segment.episode_idx}"]
        for t in range(segment.start_frame, segment.end_frame + 1):
            ts_key = f"timestep_{t}"
            if ts_key not in ep:
                continue
            ts = ep[ts_key]
            frames.append(ts["obs"]["front_rgb"][()])
            joint_state = ts["obs"]["joint_state"][()]
            gripper_state = ts["obs"]["gripper_state"][()]
            states_list.append(
                np.concatenate([joint_state, gripper_state[:1]], dtype=np.float32)
            )

    states = np.stack(states_list, axis=0) if states_list else np.zeros((0, 8))
    return frames, states


# ---------------------------------------------------------------------------
# Keyframe strip visualization
# ---------------------------------------------------------------------------


def save_keyframe_strip(
    frames: list[np.ndarray],
    keyframe_indices: list[int],
    subtask_label: str,
    output_path: str,
) -> None:
    if not keyframe_indices:
        return
    n = len(keyframe_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    labels = ["Early Context", "Ramp Start", "Transition", "Completion", "Post-Completion"]
    for i, kf in enumerate(keyframe_indices):
        idx = min(kf, len(frames) - 1)
        axes[i].imshow(frames[idx])
        lbl = labels[i] if i < len(labels) else f"KF{i}"
        axes[i].set_title(f"{lbl}\nframe {kf}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle(f"[Video Demo] {subtask_label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Process one segment
# ---------------------------------------------------------------------------


def process_demo_segment(
    segment: SubtaskSegment,
    output_dir: str,
) -> dict:
    """Extract keyframes for one video demo subtask segment."""
    seg_id = (
        f"{segment.env_id}_ep{segment.episode_idx}"
        f"_vdemo_f{segment.start_frame}-{segment.end_frame}"
    )
    seg_dir = os.path.join(output_dir, seg_id)
    os.makedirs(seg_dir, exist_ok=True)

    frames, states = load_segment_frames(segment)
    T = len(frames)
    if T < 5:
        return {"seg_id": seg_id, "skipped": True, "reason": "too_few_frames"}

    # For video demos: the whole segment is a successful execution.
    # Set ramp at ~20% so early context has room, completion at ~80%
    # so post-completion can show the final result.
    interval_start = max(1, int(0.2 * T))
    completion_frame = max(interval_start + 1, int(0.8 * T))
    interval_end = T - 1

    keyframes = select_keyframes(
        frames,
        states if states.shape[0] > 0 else None,
        interval_start,
        completion_frame,
        interval_end,
        num_keyframes=5,
    )

    # Save outputs
    summary = {
        "seg_id": seg_id,
        "skipped": False,
        "subtask_label": segment.subtask_label,
        "task_goal": segment.task_goal,
        "env_id": segment.env_id,
        "episode_idx": segment.episode_idx,
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "num_frames": T,
        "is_video_demo": True,
        "success_detected": True,
        "successful_interval_start_frame": 0,
        "completion_start_frame": completion_frame,
        "successful_interval_end_frame": interval_end,
        "failed_regions": [],
        "selected_keyframes": keyframes,
        "confidence": 1.0,
        "method": "video_demo_saliency",
    }

    with open(os.path.join(seg_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Keyframes JSON
    kf_data = {
        "selected_keyframes": keyframes,
        "num_keyframes": len(keyframes),
        "interval_start": 0,
        "completion_start": completion_frame,
        "interval_end": interval_end,
    }
    with open(os.path.join(seg_dir, "selected_keyframes.json"), "w") as f:
        json.dump(kf_data, f, indent=2)

    # Interval JSON (for viewer compatibility)
    interval_dict = {
        "successful_interval_start_frame": 0,
        "completion_start_frame": completion_frame,
        "successful_interval_end_frame": interval_end,
        "success_detected": True,
        "failed_regions": [],
        "selected_keyframes": keyframes,
        "confidence": 1.0,
        "completion_peak_idx": None,
        "ramp_start_idx": 0,
    }
    with open(os.path.join(seg_dir, "successful_interval.json"), "w") as f:
        json.dump(interval_dict, f, indent=2)

    # Keyframe strip visualization
    save_keyframe_strip(
        frames, keyframes, segment.subtask_label,
        os.path.join(seg_dir, "keyframe_strip.png"),
    )

    print(f"  {seg_id}: {T} frames, keyframes={keyframes}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract keyframes from video demo subtask segments"
    )
    parser.add_argument(
        "--raw_data_path", type=str, default="data/robomme_data_h5",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/topreward_full",
    )
    parser.add_argument(
        "--episodes_per_task", type=int, default=100,
        help="Number of episodes to process per task (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    rng = random.Random(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    h5_files = sorted(
        f for f in os.listdir(args.raw_data_path) if f.endswith(".h5")
    )

    total_segments = 0
    total_processed = 0

    for fname in h5_files:
        h5_path = os.path.join(args.raw_data_path, fname)
        env_id = fname.split(".")[0].split("_")[-1]

        # Check if this task has video demos
        with h5py.File(h5_path, "r") as f:
            ep = f["episode_1"]
            exec_start = first_execution_step(ep)
        if exec_start == 0:
            continue

        print(f"\n{'='*60}")
        print(f"Task: {env_id} (video demo: {exec_start} frames)")
        print(f"{'='*60}")

        with h5py.File(h5_path, "r") as f:
            episode_indices = get_episode_indices(f)

        n_sample = min(args.episodes_per_task, len(episode_indices))
        sampled = sorted(rng.sample(episode_indices, n_sample))

        for ep_idx in sampled:
            segments = extract_videodemo_segments(h5_path, ep_idx, env_id)
            if not segments:
                continue

            print(f"\n  Episode {ep_idx}: {len(segments)} video demo subtasks")
            total_segments += len(segments)

            for seg in segments:
                result = process_demo_segment(seg, args.output_dir)
                if not result.get("skipped"):
                    total_processed += 1

    print(f"\n{'='*60}")
    print(f"Done. Processed {total_processed}/{total_segments} video demo segments.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
