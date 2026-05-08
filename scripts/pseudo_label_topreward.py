"""TOPReward-based pseudo-labeling using Qwen3-VL-8B.

Estimates task completion progress over video prefixes, finds the final
successful ramp, filters out failed attempts, and selects keyframes from
the successful interval.

Usage:
    python scripts/pseudo_label_topreward.py \
        --raw_data_path data/robomme_data_h5 \
        --output_dir data/topreward_labels \
        --episodes_per_task 3 \
        --model_name Qwen/Qwen3-VL-8B-Instruct \
        --num_prefixes 16 \
        --target_fps 4.0 \
        --num_keyframes 5 \
        --max_segments 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mme_vla_suite.dataset_builder.robomme_h5_utils import get_episode_indices
from mme_vla_suite.pseudo_labeling.episode_sampler import (
    SubtaskSegment,
    extract_subtask_segments,
    sample_episodes_stratified,
)
from mme_vla_suite.pseudo_labeling.progress_curve import (
    normalize_rewards,
)
from mme_vla_suite.pseudo_labeling.success_interval import (
    SuccessIntervalResult,
    detect_success_interval,
)
from mme_vla_suite.pseudo_labeling.topreward_qwen_scorer import (
    score_prefixes,
    score_window,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (reused pattern from pseudo_label_phases.py)
# ---------------------------------------------------------------------------


def load_segment_data(
    segment: SubtaskSegment,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Load frames and states for a subtask segment from H5."""
    frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    states_list: list[np.ndarray] = []

    with h5py.File(segment.episode_h5_path, "r") as f:
        ep = f[f"episode_{segment.episode_idx}"]
        for t in range(segment.start_frame, segment.end_frame + 1):
            ts_key = f"timestep_{t}"
            if ts_key not in ep:
                continue
            ts = ep[ts_key]
            frames.append(ts["obs"]["front_rgb"][()])
            frame_indices.append(t)
            joint_state = ts["obs"]["joint_state"][()]
            gripper_state = ts["obs"]["gripper_state"][()]
            states_list.append(
                np.concatenate([joint_state, gripper_state[:1]], dtype=np.float32)
            )

    states = np.stack(states_list, axis=0) if states_list else np.zeros((0, 8))
    return frames, frame_indices, states


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _plot_progress_curve(
    prefix_end_indices: list[int],
    raw_reward: list[float],
    reward_smooth: np.ndarray,
    interval_result: SuccessIntervalResult,
    selected_keyframes: list[int],
    subtask_label: str,
    output_path: str,
) -> None:
    """Plot the progress curve with annotations."""
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.array(prefix_end_indices)

    # Raw reward points
    ax.scatter(x, raw_reward, c="steelblue", alpha=0.5, s=30, label="raw reward (norm)", zorder=3)

    # Smoothed curve
    ax.plot(x, reward_smooth, c="navy", linewidth=2, label="smoothed", zorder=4)

    # Failed regions as red spans
    for fr in interval_result.failed_regions:
        ax.axvspan(fr[0], fr[1], color="red", alpha=0.15, label="_failed")

    # Successful interval start
    ax.axvline(
        interval_result.successful_interval_start_frame,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"interval start ({interval_result.successful_interval_start_frame})",
    )

    # Completion start
    ax.axvline(
        interval_result.completion_start_frame,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"completion start ({interval_result.completion_start_frame})",
    )

    # Keyframes as markers
    for kf in selected_keyframes:
        ax.axvline(kf, color="purple", linestyle=":", alpha=0.6, linewidth=1)
    if selected_keyframes:
        ax.scatter(
            selected_keyframes,
            [0.02] * len(selected_keyframes),
            c="purple",
            marker="^",
            s=80,
            zorder=5,
            label="keyframes",
        )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Normalized TOPReward")
    ax.set_title(
        f"Progress curve: {subtask_label}\n"
        f"success={interval_result.success_detected}, "
        f"conf={interval_result.confidence:.2f}"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_keyframe_strip(
    frames: list[np.ndarray],
    keyframe_indices: list[int],
    subtask_label: str,
    output_path: str,
) -> None:
    """Save a horizontal strip of keyframes."""
    if not keyframe_indices:
        return
    n = len(keyframe_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, kf in enumerate(keyframe_indices):
        idx = min(kf, len(frames) - 1)
        axes[i].imshow(frames[idx])
        axes[i].set_title(f"frame {kf}", fontsize=10)
        axes[i].axis("off")
    fig.suptitle(f"Keyframes: {subtask_label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _save_failed_regions_strip(
    frames: list[np.ndarray],
    failed_regions: list[list[int]],
    subtask_label: str,
    output_path: str,
) -> None:
    """Save a strip showing one representative frame per failed region."""
    if not failed_regions:
        return
    n = len(failed_regions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, (start, end) in enumerate(failed_regions):
        mid = (start + end) // 2
        mid = min(mid, len(frames) - 1)
        axes[i].imshow(frames[mid])
        axes[i].set_title(f"failed [{start}-{end}]", fontsize=10)
        axes[i].axis("off")
    fig.suptitle(f"Failed regions: {subtask_label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Segment processing
# ---------------------------------------------------------------------------


def process_segment(
    segment: SubtaskSegment,
    model,
    processor,
    output_dir: str,
    num_prefixes: int = 16,
    target_fps: float = 4.0,
    num_keyframes: int = 5,
    verify_attempt_windows: bool = False,
    save_visualizations: bool = True,
    debug_save_prefix_inputs: bool = False,
) -> dict:
    """Process one subtask segment through the TOPReward pipeline."""
    seg_id = (
        f"{segment.env_id}_ep{segment.episode_idx}"
        f"_f{segment.start_frame}-{segment.end_frame}"
    )
    seg_dir = os.path.join(output_dir, seg_id)
    os.makedirs(seg_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Segment: {seg_id}")
    print(f"  Subtask: {segment.subtask_label}")
    print(f"  Frames: {segment.start_frame} - {segment.end_frame}")
    print(f"{'=' * 60}")

    # --- Load data ---
    frames, frame_indices, states = load_segment_data(segment)
    T = len(frames)
    if T < 5:
        print(f"  Skipping: too few frames ({T})")
        return {"seg_id": seg_id, "skipped": True, "reason": "too_few_frames"}

    print(f"  Loaded {T} frames")

    # --- Score prefixes ---
    t0 = time.time()
    prefix_result = score_prefixes(
        model,
        processor,
        frames,
        segment.subtask_label,
        num_prefixes=num_prefixes,
        target_fps=target_fps,
    )
    print(f"  Prefix scoring: {time.time() - t0:.1f}s")

    raw_reward = np.array(prefix_result["raw_reward"])
    reward_norm = np.array(prefix_result["reward_norm"])

    # No smoothing — use normalized rewards directly (matching TOPReward)
    delta = np.zeros_like(reward_norm)
    delta[1:] = reward_norm[1:] - reward_norm[:-1]

    # --- Detect success interval ---
    interval_result = detect_success_interval(
        reward_smooth=reward_norm,
        prefix_end_indices=prefix_result["prefix_end_indices"],
        total_frames=T,
        frames=frames,
        states=states if states.shape[0] > 0 else None,
        num_keyframes=num_keyframes,
    )

    # --- Optional: verify attempt windows ---
    window_verification = None
    if verify_attempt_windows and interval_result.success_detected:
        t0 = time.time()
        window_verification = _verify_windows(
            model, processor, frames, segment.subtask_label,
            interval_result, target_fps,
        )
        print(f"  Window verification: {time.time() - t0:.1f}s")

    # --- Debug: save prefix inputs ---
    if debug_save_prefix_inputs:
        _save_prefix_inputs(
            frames, prefix_result["prefix_end_indices"],
            target_fps, seg_dir,
        )

    # --- Console log ---
    print(f"  Rewards: {[round(r, 2) for r in reward_norm.tolist()]}")
    print(f"  Failed regions: {interval_result.failed_regions}")
    print(
        f"  Successful interval: "
        f"{interval_result.successful_interval_start_frame}-"
        f"{interval_result.successful_interval_end_frame}"
    )
    print(f"  Keyframes: {interval_result.selected_keyframes}")
    print(f"  Success detected: {interval_result.success_detected}")
    print(f"  Confidence: {interval_result.confidence:.2f}")

    # --- Save outputs ---
    _save_all_outputs(
        seg_dir,
        segment,
        prefix_result,
        reward_norm,
        delta,
        interval_result,
        window_verification,
        T,
    )

    # --- Visualizations ---
    if save_visualizations:
        _plot_progress_curve(
            prefix_result["prefix_end_indices"],
            reward_norm.tolist(),
            reward_norm,
            interval_result,
            interval_result.selected_keyframes,
            segment.subtask_label,
            os.path.join(seg_dir, "progress_curve.png"),
        )
        _save_keyframe_strip(
            frames,
            interval_result.selected_keyframes,
            segment.subtask_label,
            os.path.join(seg_dir, "keyframe_strip.png"),
        )
        if interval_result.failed_regions:
            _save_failed_regions_strip(
                frames,
                interval_result.failed_regions,
                segment.subtask_label,
                os.path.join(seg_dir, "failed_regions_strip.png"),
            )

    # --- Summary ---
    summary = {
        "seg_id": seg_id,
        "skipped": False,
        "subtask_label": segment.subtask_label,
        "task_goal": segment.task_goal,
        "env_id": segment.env_id,
        "episode_idx": segment.episode_idx,
        "num_frames": T,
        "num_prefixes": num_prefixes,
        "success_detected": interval_result.success_detected,
        "successful_interval_start_frame": interval_result.successful_interval_start_frame,
        "completion_start_frame": interval_result.completion_start_frame,
        "successful_interval_end_frame": interval_result.successful_interval_end_frame,
        "failed_regions": interval_result.failed_regions,
        "selected_keyframes": interval_result.selected_keyframes,
        "confidence": interval_result.confidence,
        "method": "topreward_qwen3vl8b_prefix_logp_true",
    }

    with open(os.path.join(seg_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Window verification (Section 6)
# ---------------------------------------------------------------------------


def _verify_windows(
    model,
    processor,
    frames: list[np.ndarray],
    subtask_label: str,
    interval_result: SuccessIntervalResult,
    target_fps: float,
) -> dict:
    """Score individual attempt windows to verify success detection."""
    windows = []

    # Successful ramp window
    windows.append({
        "type": "success_ramp",
        "start": interval_result.successful_interval_start_frame,
        "end": interval_result.successful_interval_end_frame,
    })

    # Failed region windows
    for i, (s, e) in enumerate(interval_result.failed_regions):
        windows.append({"type": f"failed_{i}", "start": s, "end": e})

    results = []
    for w in windows:
        r = score_window(
            model, processor, frames,
            w["start"], w["end"],
            subtask_label, target_fps=target_fps,
        )
        results.append({
            **w,
            "logp_true": r["logp_true"],
            "reward": r["reward"],
        })

    return {"window_scores": results}


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _save_all_outputs(
    seg_dir: str,
    segment: SubtaskSegment,
    prefix_result: dict,
    reward_norm: np.ndarray,
    delta: np.ndarray,
    interval_result: SuccessIntervalResult,
    window_verification: dict | None,
    T: int,
) -> None:
    """Save all JSON outputs for one segment."""
    # progress_scores.json
    progress = {
        "prefix_end_indices": prefix_result["prefix_end_indices"],
        "logp_true": prefix_result["logp_true"],
        "raw_reward": prefix_result["raw_reward"],
        "reward_norm": reward_norm.tolist(),
        "delta": delta.tolist(),
        "frames_sent_per_prefix": prefix_result["frames_sent_per_prefix"],
    }
    with open(os.path.join(seg_dir, "progress_scores.json"), "w") as f:
        json.dump(progress, f, indent=2)

    # successful_interval.json
    interval_dict = interval_result.to_dict()
    with open(os.path.join(seg_dir, "successful_interval.json"), "w") as f:
        json.dump(interval_dict, f, indent=2)

    # selected_keyframes.json
    kf_data = {
        "selected_keyframes": interval_result.selected_keyframes,
        "num_keyframes": len(interval_result.selected_keyframes),
        "interval_start": interval_result.successful_interval_start_frame,
        "completion_start": interval_result.completion_start_frame,
        "interval_end": interval_result.successful_interval_end_frame,
    }
    with open(os.path.join(seg_dir, "selected_keyframes.json"), "w") as f:
        json.dump(kf_data, f, indent=2)

    # Window verification
    if window_verification is not None:
        with open(os.path.join(seg_dir, "window_verification.json"), "w") as f:
            json.dump(window_verification, f, indent=2)


def _save_prefix_inputs(
    frames: list[np.ndarray],
    prefix_end_indices: list[int],
    target_fps: float,
    seg_dir: str,
) -> None:
    """Save the exact frames sent to Qwen for each prefix (debug mode)."""
    from mme_vla_suite.pseudo_labeling.topreward_qwen_scorer import subsample_to_fps

    prefix_dir = os.path.join(seg_dir, "sampled_prefix_frames")
    os.makedirs(prefix_dir, exist_ok=True)

    for i, end_idx in enumerate(prefix_end_indices):
        pdir = os.path.join(prefix_dir, f"prefix_{i:03d}")
        os.makedirs(pdir, exist_ok=True)

        prefix_frames = frames[: end_idx + 1]
        sampled = subsample_to_fps(prefix_frames, target_fps=target_fps)
        for j, frame in enumerate(sampled):
            img = Image.fromarray(frame)
            img.save(os.path.join(pdir, f"frame_{j:03d}.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="TOPReward-based pseudo-labeling with Qwen3-VL-8B"
    )
    parser.add_argument(
        "--raw_data_path", type=str, default="data/robomme_data_h5",
        help="Path to raw H5 data files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/topreward_labels",
        help="Output directory",
    )
    parser.add_argument(
        "--fraction", type=float, default=0.10,
        help="Fraction of episodes to sample (default: 0.10)",
    )
    parser.add_argument(
        "--episodes_per_task", type=int, default=3,
        help="Fixed number of episodes per task (overrides --fraction)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--task", type=str, default=None,
        help="Run on a specific task only (e.g. BinFill). Requires --episode.",
    )
    parser.add_argument(
        "--episode", type=int, default=None,
        help="Run on a specific episode index (e.g. 2). Requires --task.",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen VL model name or local checkpoint path",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--dtype", type=str, default="auto",
        help="Model dtype (auto, bfloat16, float16, float32)",
    )
    parser.add_argument(
        "--num_prefixes", type=int, default=16,
        help="Number of prefix endpoints to evaluate per segment",
    )
    parser.add_argument(
        "--target_fps", type=float, default=4.0,
        help="Subsample each prefix to this fps before feeding to Qwen (source is 20Hz)",
    )
    parser.add_argument(
        "--num_keyframes", type=int, default=5,
        help="Number of keyframes to select (3 or 5)",
    )
    parser.add_argument(
        "--verify_attempt_windows", action="store_true",
        help="Run optional window verification pass (multiplies compute)",
    )
    parser.add_argument(
        "--max_segments", type=int, default=None,
        help="Max segments to process (for debugging)",
    )
    parser.add_argument(
        "--save_visualizations", action="store_true", default=True,
        help="Save progress curve plots and keyframe strips",
    )
    parser.add_argument(
        "--no_visualizations", action="store_true",
        help="Disable visualization saving",
    )
    parser.add_argument(
        "--debug_save_prefix_inputs", action="store_true",
        help="Save exact frames sent to Qwen for each prefix",
    )
    args = parser.parse_args()

    if args.no_visualizations:
        args.save_visualizations = False

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Sample episodes ---
    print("=" * 60)
    print("Step 1: Sampling episodes")
    print("=" * 60)

    if args.task is not None and args.episode is not None:
        # Run on a specific task + episode
        h5_path = os.path.join(args.raw_data_path, f"record_dataset_{args.task}.h5")
        if not os.path.exists(h5_path):
            h5_path = os.path.join(args.raw_data_path, f"data_{args.task}.h5")
        print(f"  Task: {args.task}, Episode: {args.episode}")
        print(f"  H5: {h5_path}")
        segments = extract_subtask_segments(h5_path, args.episode, args.task)
        print(f"  Found {len(segments)} subtask segments")
    elif args.task is not None:
        # Run on a specific task, sample N episodes
        h5_path = os.path.join(args.raw_data_path, f"record_dataset_{args.task}.h5")
        if not os.path.exists(h5_path):
            h5_path = os.path.join(args.raw_data_path, f"data_{args.task}.h5")
        print(f"  Task: {args.task}, sampling {args.episodes_per_task} episodes")
        print(f"  H5: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            episode_indices = get_episode_indices(f)
        rng = random.Random(args.seed)
        n_sample = min(args.episodes_per_task, len(episode_indices))
        sampled_eps = sorted(rng.sample(episode_indices, n_sample))
        print(f"  Sampled episodes: {sampled_eps}")

        segments = []
        for ep_idx in sampled_eps:
            segments.extend(extract_subtask_segments(h5_path, ep_idx, args.task))
        print(f"  Found {len(segments)} subtask segments")
    else:
        segments = sample_episodes_stratified(
            args.raw_data_path,
            fraction=args.fraction,
            episodes_per_task=args.episodes_per_task,
            seed=args.seed,
        )

    if args.max_segments is not None:
        segments = segments[: args.max_segments]
        print(f"  (Limited to {args.max_segments} segments for debugging)")

    # --- Step 2: Load model ---
    print("\n" + "=" * 60)
    print("Step 2: Loading Qwen VL model")
    print("=" * 60)

    from mme_vla_suite.pseudo_labeling.topreward_qwen_scorer import (
        _load_model_and_processor,
    )

    model, processor, tokenizer = _load_model_and_processor(
        args.model_name, device=args.device, dtype=args.dtype,
    )
    print(f"  Model: {args.model_name}")
    print(f"  Device: {args.device}, dtype: {args.dtype}")

    # --- Step 3: Process segments ---
    print("\n" + "=" * 60)
    print(f"Step 3: Processing {len(segments)} segments")
    print("=" * 60)

    all_summaries: list[dict] = []
    env_counts: dict[str, int] = defaultdict(int)

    for i, seg in enumerate(segments):
        print(f"\n--- Segment {i + 1}/{len(segments)} ---")
        summary = process_segment(
            seg,
            model,
            processor,
            args.output_dir,
            num_prefixes=args.num_prefixes,
            target_fps=args.target_fps,
            num_keyframes=args.num_keyframes,
            verify_attempt_windows=args.verify_attempt_windows,
            save_visualizations=args.save_visualizations,
            debug_save_prefix_inputs=args.debug_save_prefix_inputs,
        )
        all_summaries.append(summary)
        env_counts[seg.env_id] += 1

    # --- Step 4: Global summary ---
    print("\n" + "=" * 60)
    print("Step 4: Summary")
    print("=" * 60)

    processed = [s for s in all_summaries if not s.get("skipped", False)]
    skipped = [s for s in all_summaries if s.get("skipped", False)]

    print(f"  Processed: {len(processed)} segments")
    print(f"  Skipped: {len(skipped)} segments")
    print(f"  Environments: {dict(env_counts)}")

    if processed:
        confidences = [s["confidence"] for s in processed]
        n_success = sum(1 for s in processed if s.get("success_detected", False))
        n_failed_regions = sum(
            len(s.get("failed_regions", [])) for s in processed
        )
        print(f"  Mean confidence: {np.mean(confidences):.3f}")
        print(f"  Success detected: {n_success}/{len(processed)}")
        print(f"  Total failed regions detected: {n_failed_regions}")

    global_summary = {
        "args": vars(args),
        "num_segments": len(segments),
        "num_processed": len(processed),
        "num_skipped": len(skipped),
        "env_counts": dict(env_counts),
        "summaries": all_summaries,
    }

    with open(os.path.join(args.output_dir, "global_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
