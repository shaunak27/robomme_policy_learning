"""Pseudo-label frame phases within subtask segments.

Orchestrates the full pipeline:
1. Sample episodes per task, stratified across tasks.
2. For each subtask segment:
   a. Load frames and robot states from H5.
   b. Score frames with CLIP against 3 phase-specific text prompts.
   c. Compute combined attempt score (CLIP contact + visual + state + gripper).
   d. Detect candidate action windows from attempt score peaks.
   e. Score candidate windows with MLLM (Gemini).
   f. Fuse CLIP + MLLM scores using temporal constraints.
   g. Save raw scores, final labels, and representative frames.
   h. Generate visualizations.

Usage:
    python scripts/pseudo_label_phases.py \
        --raw_data_path data/robomme_data_h5 \
        --output_dir data/pseudo_labels \
        --episodes_per_task 3 \
        --clip_model openai/clip-vit-large-patch14 \
        --gemini_model gemini-2.5-flash-lite \
        --use_triples           # Use image triples instead of video for MLLM
        --skip_mllm             # Skip MLLM scoring (CLIP-only mode)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import h5py
import numpy as np

from mme_vla_suite.pseudo_labeling.episode_sampler import (
    SubtaskSegment,
    sample_episodes_stratified,
)
from mme_vla_suite.pseudo_labeling.clip_scorer import (
    CLIPScorer,
    CLIPPhaseScores,
    generate_phase_prompts,
)
from mme_vla_suite.pseudo_labeling.candidate_detector import (
    compute_attempt_score,
    detect_candidate_windows,
    AttemptScoreResult,
    CandidateWindow,
)
from mme_vla_suite.pseudo_labeling.label_fusion import (
    fuse_labels,
    FusedLabels,
    LABEL_NAMES,
)
from mme_vla_suite.pseudo_labeling.visualization import (
    generate_all_visualizations,
)


def load_segment_data(
    segment: SubtaskSegment,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Load frames and states for a subtask segment from H5.

    Returns:
        frames: List of RGB images (H,W,3 uint8).
        frame_indices: List of global timestep indices.
        states: (N, 8) robot states.
    """
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


def process_segment(
    segment: SubtaskSegment,
    clip_scorer: CLIPScorer,
    mllm_scorer,
    output_dir: str,
    use_triples: bool = True,
    smooth_kernel: int = 5,
) -> dict:
    """Process one subtask segment through the full pipeline.

    Returns a summary dict with key results.
    """
    seg_id = (
        f"{segment.env_id}_ep{segment.episode_idx}"
        f"_f{segment.start_frame}-{segment.end_frame}"
    )
    seg_output_dir = os.path.join(output_dir, seg_id)
    os.makedirs(seg_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Segment: {seg_id}")
    print(f"  Subtask: {segment.subtask_label}")
    print(f"  Frames: {segment.start_frame} - {segment.end_frame}")
    print(f"{'='*60}")

    # --- Load data ---
    frames, frame_indices, states = load_segment_data(segment)
    if len(frames) < 5:
        print(f"  Skipping: too few frames ({len(frames)})")
        return {"seg_id": seg_id, "skipped": True, "reason": "too_few_frames"}

    print(f"  Loaded {len(frames)} frames")

    # --- CLIP scoring ---
    t0 = time.time()
    clip_scores_raw = clip_scorer.score_segment(
        frames, frame_indices, segment.subtask_label
    )
    clip_scores_smooth = clip_scores_raw.smoothed(kernel_size=smooth_kernel) #FLAG: Do we need smoothing??
    print(f"  CLIP scoring: {time.time() - t0:.1f}s")

    # --- Compute attempt score ---
    t0 = time.time()
    attempt_result = compute_attempt_score(
        frames, frame_indices, states, clip_scores_smooth,
    )
    print(f"  Attempt score: {time.time() - t0:.1f}s")

    # --- Candidate detection ---
    t0 = time.time()
    candidates = detect_candidate_windows(
        attempt_result, frame_indices,
    )
    print(f"  Candidate detection: {time.time() - t0:.1f}s, found {len(candidates)} candidates")

    # --- MLLM scoring ---
    mllm_results = []
    if mllm_scorer is not None and candidates:
        t0 = time.time()
        mllm_results = mllm_scorer.score_windows(
            candidates, frames, frame_indices, segment.subtask_label,
            use_triples=use_triples,
        )
        n_success = sum(1 for r in mllm_results if r.is_success)
        n_fail = sum(1 for r in mllm_results if r.is_failed_attempt)
        print(f"  MLLM scoring: {time.time() - t0:.1f}s, {n_success} success, {n_fail} fail")

    # --- Label fusion ---
    fused = fuse_labels(
        clip_scores_smooth, candidates, mllm_results,
        attempt_result, frame_indices,
    )
    print(f"  Label source: {fused.label_source}")
    print(f"  Transition start: {fused.transition_start}")
    print(f"  Transition peak: {fused.transition_frame}")
    print(f"  Completion start: {fused.completion_start}")
    print(f"  Failed attempt windows: {len(fused.failed_attempt_windows)}")

    # --- Save raw scores ---
    _save_scores(seg_output_dir, clip_scores_raw, clip_scores_smooth, fused, segment)

    # --- Save attempt score ---
    _save_attempt_score(seg_output_dir, attempt_result)

    # --- Save candidate windows ---
    _save_candidate_windows(seg_output_dir, candidates)

    # --- Save fusion decisions ---
    _save_fusion_decisions(seg_output_dir, fused)

    # --- Save MLLM results ---
    if mllm_results:
        _save_mllm_results(seg_output_dir, mllm_results)

    # --- Visualizations ---
    generate_all_visualizations(
        clip_scores_raw,
        clip_scores_smooth,
        fused,
        attempt_result,
        candidates,
        frames,
        frame_indices,
        segment.subtask_label,
        seg_output_dir,
    )

    # --- Summary ---
    label_counts = {}
    for lv, ln in LABEL_NAMES.items():
        count = int(np.sum(fused.labels == lv))
        if count > 0:
            label_counts[ln] = count

    summary = {
        "seg_id": seg_id,
        "skipped": False,
        "env_id": segment.env_id,
        "episode_idx": segment.episode_idx,
        "subtask_label": segment.subtask_label,
        "task_goal": segment.task_goal,
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "num_frames": len(frames),
        "num_candidates": len(candidates),
        "label_source": fused.label_source,
        "transition_start": fused.transition_start,
        "transition_frame": fused.transition_frame,
        "completion_start": fused.completion_start,
        "num_failed_attempts": len(fused.failed_attempt_windows),
        "failed_attempt_frames": fused.failed_attempt_frames,
        "label_counts": label_counts,
        "mean_confidence": float(fused.confidence.mean()),
    }

    with open(os.path.join(seg_output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _save_scores(
    output_dir: str,
    clip_raw: CLIPPhaseScores,
    clip_smooth: CLIPPhaseScores,
    fused: FusedLabels,
    segment: SubtaskSegment,
) -> None:
    """Save all raw scores and final labels."""
    np.savez_compressed(
        os.path.join(output_dir, "clip_scores_raw.npz"),
        **clip_raw.as_dict(),
    )
    np.savez_compressed(
        os.path.join(output_dir, "clip_scores_smooth.npz"),
        **clip_smooth.as_dict(),
    )
    np.savez_compressed(
        os.path.join(output_dir, "fused_labels.npz"),
        frame_indices=fused.frame_indices,
        labels=fused.labels,
        confidence=fused.confidence,
        clip_dominant_phase=fused.clip_dominant_phase,
        mllm_labels=fused.mllm_labels,
    )

    # Also save as human-readable JSON
    label_list = []
    for i in range(len(fused.frame_indices)):
        label_list.append({
            "frame": int(fused.frame_indices[i]),
            "label": LABEL_NAMES.get(int(fused.labels[i]), "unknown"),
            "label_id": int(fused.labels[i]),
            "confidence": round(float(fused.confidence[i]), 4),
        })

    with open(os.path.join(output_dir, "frame_labels.json"), "w") as f:
        json.dump(
            {
                "subtask_label": segment.subtask_label,
                "task_goal": segment.task_goal,
                "env_id": segment.env_id,
                "episode_idx": segment.episode_idx,
                "label_source": fused.label_source,
                "transition_start": fused.transition_start,
                "transition_frame": fused.transition_frame,
                "completion_start": fused.completion_start,
                "num_failed_attempts": len(fused.failed_attempt_windows),
                "failed_attempt_frames": fused.failed_attempt_frames,
                "frames": label_list,
            },
            f,
            indent=2,
        )

    # Save the text prompts used
    prompts = generate_phase_prompts(segment.subtask_label)
    with open(os.path.join(output_dir, "clip_prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)


def _save_attempt_score(
    output_dir: str,
    attempt_result: AttemptScoreResult,
) -> None:
    """Save attempt score and its components."""
    np.savez_compressed(
        os.path.join(output_dir, "attempt_score.npz"),
        attempt_score=attempt_result.attempt_score,
        visual_change=attempt_result.visual_change,
        gripper_change=attempt_result.gripper_change,
        state_change=attempt_result.state_change,
        clip_contact=attempt_result.clip_contact,
        frame_indices=attempt_result.frame_indices,
    )


def _save_fusion_decisions(
    output_dir: str,
    fused: FusedLabels,
) -> None:
    """Save fusion decisions as JSON for diagnostics."""
    decisions = fused.metadata.get("fusion_decisions", {})

    # Convert any numpy types to native Python
    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(os.path.join(output_dir, "fusion_decisions.json"), "w") as f:
        json.dump(_clean(decisions), f, indent=2)


def _save_candidate_windows(
    output_dir: str,
    candidates: list[CandidateWindow],
) -> None:
    """Save candidate windows as JSON."""
    windows = []
    for cand in candidates:
        windows.append({
            "center_frame": int(cand.center_frame),
            "center_global": int(cand.center_global),
            "start_frame": int(cand.start_frame),
            "end_frame": int(cand.end_frame),
            "start_local": int(cand.start_local),
            "end_local": int(cand.end_local),
            "attempt_score": float(cand.attempt_score),
            "source_contributions": {
                k: float(v) for k, v in cand.source_contributions.items()
            },
        })

    with open(os.path.join(output_dir, "candidate_windows.json"), "w") as f:
        json.dump(windows, f, indent=2)


def _save_mllm_results(
    output_dir: str, mllm_results: list,
) -> None:
    """Save MLLM results as JSON."""
    results_list = []
    for r in mllm_results:
        results_list.append({
            "center_global": r.window.center_global,
            "start_frame": r.window.start_frame,
            "end_frame": r.window.end_frame,
            "is_success": r.is_success,
            "is_action_happening": r.is_action_happening,
            "is_failed_attempt": r.is_failed_attempt,
            "confidence": r.confidence,
            "reason": r.reason,
            "raw_responses": {
                k: str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None)))
                else v
                for k, v in r.raw_responses.items()
            },
        })

    with open(os.path.join(output_dir, "mllm_results.json"), "w") as f:
        json.dump(results_list, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Pseudo-label frame phases within subtask segments"
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/robomme_data_h5",
        help="Path to raw H5 data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/pseudo_labels",
        help="Output directory for pseudo-labels and visualizations",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="Fraction of episodes to sample (default: 0.10)",
    )
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=3,
        help="Fixed number of episodes per task (overrides --fraction)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP model name from HuggingFace",
    )
    parser.add_argument(
        "--clip_device",
        type=str,
        default=None,
        help="Device for CLIP (default: auto-detect)",
    )
    parser.add_argument(
        "--clip_batch_size",
        type=int,
        default=32,
        help="Batch size for CLIP image encoding",
    )
    parser.add_argument(
        "--smooth_kernel",
        type=int,
        default=5,
        help="Kernel size for temporal smoothing of CLIP scores",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model name for MLLM scoring",
    )
    parser.add_argument(
        "--use_triples",
        action="store_true",
        help="Use image triples instead of video clips for MLLM",
    )
    parser.add_argument(
        "--skip_mllm",
        action="store_true",
        help="Skip MLLM scoring (CLIP-only mode for fast iteration)",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=None,
        help="Max segments to process (for debugging)",
    )
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=0.5,
        help="Delay between MLLM API calls (seconds)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Sample episodes ---
    print("=" * 60)
    print("Step 1: Sampling episodes")
    print("=" * 60)
    segments = sample_episodes_stratified(
        args.raw_data_path,
        fraction=args.fraction,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
    )

    if args.max_segments is not None:
        segments = segments[: args.max_segments]
        print(f"  (Limited to {args.max_segments} segments for debugging)")

    # --- Step 2: Initialize models ---
    print("\n" + "=" * 60)
    print("Step 2: Initializing models")
    print("=" * 60)

    clip_scorer = CLIPScorer(
        model_name=args.clip_model,
        device=args.clip_device,
        batch_size=args.clip_batch_size,
    )

    mllm_scorer = None
    if not args.skip_mllm:
        from mme_vla_suite.pseudo_labeling.mllm_scorer import MLLMScorer

        mllm_scorer = MLLMScorer(
            model_name=args.gemini_model,
            tmp_dir=os.path.join(args.output_dir, "_mllm_tmp"),
            rate_limit_delay=args.rate_limit_delay,
        )
        print(f"  MLLM scorer: {args.gemini_model}")
    else:
        print("  MLLM scoring: SKIPPED (--skip_mllm)")

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
            clip_scorer,
            mllm_scorer,
            args.output_dir,
            use_triples=args.use_triples,
            smooth_kernel=args.smooth_kernel,
        )
        all_summaries.append(summary)
        env_counts[seg.env_id] += 1

    # --- Step 4: Save global summary ---
    print("\n" + "=" * 60)
    print("Step 4: Summary")
    print("=" * 60)

    processed = [s for s in all_summaries if not s.get("skipped", True)]
    skipped = [s for s in all_summaries if s.get("skipped", True)]

    print(f"  Processed: {len(processed)} segments")
    print(f"  Skipped: {len(skipped)} segments")
    print(f"  Environments: {dict(env_counts)}")

    if processed:
        all_label_counts: dict[str, int] = defaultdict(int)
        for s in processed:
            for label, count in s.get("label_counts", {}).items():
                all_label_counts[label] += count
        print(f"  Total label distribution: {dict(all_label_counts)}")
        print(
            f"  Mean confidence: "
            f"{np.mean([s['mean_confidence'] for s in processed]):.3f}"
        )
        n_with_transition = sum(
            1 for s in processed if s.get("transition_frame") is not None
        )
        n_mllm_verified = sum(
            1 for s in processed if s.get("label_source") == "mllm_verified"
        )
        print(f"  Segments with transition frame: {n_with_transition}/{len(processed)}")
        print(f"  MLLM-verified segments: {n_mllm_verified}/{len(processed)}")

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
