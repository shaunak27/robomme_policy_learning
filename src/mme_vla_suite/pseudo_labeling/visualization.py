"""Visualization of pseudo-labeling results.

Generates:
1. Timeline plot of CLIP scores (3 phases) + attempt score over frames
2. Colored frame-label bar over time
3. Candidate window spans and success/failed markers
4. Side-by-side CLIP vs MLLM label comparison
5. Disagreement highlights for manual inspection
6. Thumbnail strips for top-scoring frames per phase
"""

from __future__ import annotations

import os
from typing import Sequence

import imageio
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
except ImportError:
    plt = None

from mme_vla_suite.pseudo_labeling.clip_scorer import (
    CLIPPhaseScores,
    CLIP_PHASE_NAMES,
    PHASE_NAMES,
)
from mme_vla_suite.pseudo_labeling.candidate_detector import (
    AttemptScoreResult,
    CandidateWindow,
)
from mme_vla_suite.pseudo_labeling.label_fusion import (
    FusedLabels,
    LABEL_PRECONDITION,
    LABEL_CONTACT_TRANSITION,
    LABEL_POSTCONDITION_SUCCESS,
    LABEL_FAILED_ATTEMPT,
    LABEL_UNKNOWN,
    LABEL_NAMES,
)
from mme_vla_suite.pseudo_labeling.mllm_scorer import MLLMWindowResult


# Colors for each phase
PHASE_COLORS = {
    LABEL_PRECONDITION: "#3498db",       # blue
    LABEL_CONTACT_TRANSITION: "#f39c12", # orange
    LABEL_POSTCONDITION_SUCCESS: "#2ecc71",  # green
    LABEL_FAILED_ATTEMPT: "#e74c3c",     # red
    LABEL_UNKNOWN: "#95a5a6",            # gray
}

# Colors for the 3 CLIP-scored phases only
CLIP_PHASE_COLOR_LIST = [
    PHASE_COLORS[LABEL_PRECONDITION],       # precondition
    PHASE_COLORS[LABEL_CONTACT_TRANSITION],  # contact_or_transition
    PHASE_COLORS[LABEL_POSTCONDITION_SUCCESS],  # postcondition_success
]


def plot_score_timeline(
    clip_scores: CLIPPhaseScores,
    clip_scores_smooth: CLIPPhaseScores,
    fused: FusedLabels,
    attempt_result: AttemptScoreResult | None,
    candidates: list[CandidateWindow],
    subtask_label: str,
    output_path: str,
) -> None:
    """Plot CLIP scores, attempt score, and candidate windows over time."""
    if plt is None:
        return

    n_panels = 4 if attempt_result is not None else 3
    ratios = [3, 3, 2, 1] if n_panels == 4 else [3, 3, 1]
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(16, 3 * n_panels),
        height_ratios=ratios,
    )
    fig.suptitle(f'Phase Scores: "{subtask_label}"', fontsize=13)

    frames = clip_scores.frame_indices

    # --- Panel 1: Raw CLIP scores (3 phases only) ---
    ax = axes[0]
    ax.set_title("CLIP Scores (Raw) — 3 phases")
    for i, phase in enumerate(CLIP_PHASE_NAMES):
        scores = getattr(clip_scores, phase)
        ax.plot(frames, scores, label=phase, color=CLIP_PHASE_COLOR_LIST[i], alpha=0.7)
    ax.set_ylabel("Similarity")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _add_event_markers(ax, fused, candidates)

    # --- Panel 2: Smoothed CLIP scores (3 phases only) ---
    ax = axes[1]
    ax.set_title("CLIP Scores (Smoothed) — 3 phases")
    for i, phase in enumerate(CLIP_PHASE_NAMES):
        scores = getattr(clip_scores_smooth, phase)
        ax.plot(frames, scores, label=phase, color=CLIP_PHASE_COLOR_LIST[i], linewidth=2)
    ax.set_ylabel("Similarity")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _add_event_markers(ax, fused, candidates)

    # --- Panel 3 (optional): Attempt score + components ---
    if attempt_result is not None:
        ax = axes[2]
        ax.set_title("Attempt Score (combined + components)")
        att_frames = attempt_result.frame_indices
        ax.plot(att_frames, attempt_result.attempt_score, label="attempt (combined)",
                color="black", linewidth=2)
        ax.plot(att_frames, attempt_result.clip_contact, label="clip_contact",
                color=PHASE_COLORS[LABEL_CONTACT_TRANSITION], alpha=0.5, linestyle="--")
        ax.plot(att_frames, attempt_result.visual_change, label="visual_change",
                color="#8e44ad", alpha=0.5, linestyle="--")
        ax.plot(att_frames, attempt_result.state_change, label="state_change",
                color="#1abc9c", alpha=0.5, linestyle="--")
        ax.plot(att_frames, attempt_result.gripper_change, label="gripper_change",
                color="#e67e22", alpha=0.5, linestyle="--")

        # Mark candidate peaks
        for cand in candidates:
            gi = cand.center_global
            ax.axvline(gi, color="#7f8c8d", linestyle=":", alpha=0.4)
            # Shade window
            ax.axvspan(cand.start_frame, cand.end_frame,
                       alpha=0.08, color="#7f8c8d")

        # Mark success/failed windows
        if fused.success_window is not None:
            sw = fused.success_window
            ax.axvspan(sw.start_frame, sw.end_frame, alpha=0.15, color="green",
                       label="success window")

        for fw in fused.failed_attempt_windows:
            ax.axvspan(fw.start_frame, fw.end_frame, alpha=0.15, color="red")

        ax.set_ylabel("Score (0-1)")
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)

    # --- Last panel: Colored label bar ---
    ax = axes[-1]
    ax.set_title("Final Frame Labels")
    _draw_label_bar(ax, fused.frame_indices, fused.labels)
    ax.set_xlabel("Frame Index")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _add_event_markers(
    ax: "plt.Axes",
    fused: FusedLabels,
    candidates: list[CandidateWindow],
) -> None:
    """Add vertical lines for transition region, completion, and failed attempts."""
    if fused.transition_start is not None:
        ax.axvline(fused.transition_start, color="#e67e22", linestyle="--",
                   alpha=0.7, label="transition_start")
    if fused.transition_frame is not None:
        ax.axvline(fused.transition_frame, color="green", linestyle="--",
                   alpha=0.8, label="transition_peak")
    if fused.completion_start is not None:
        ax.axvline(fused.completion_start, color="#2ecc71", linestyle="-.",
                   alpha=0.6, label="completion_start")
    for fw in fused.failed_attempt_windows:
        ax.axvline(fw.center_global, color="red", linestyle=":", alpha=0.6)


def plot_clip_vs_mllm(
    fused: FusedLabels,
    subtask_label: str,
    output_path: str,
) -> None:
    """Side-by-side comparison of CLIP-only vs MLLM-informed labels."""
    if plt is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 5), height_ratios=[1, 1, 1])
    fig.suptitle(f'CLIP vs MLLM Labels: "{subtask_label}"', fontsize=12)

    # CLIP-only labels (3-way argmax)
    axes[0].set_title("CLIP-only (3-way argmax)")
    _draw_label_bar(axes[0], fused.frame_indices, fused.clip_dominant_phase)

    # MLLM labels (where available)
    axes[1].set_title("MLLM-informed")
    _draw_label_bar(axes[1], fused.frame_indices, fused.mllm_labels)

    # Final fused labels
    axes[2].set_title(f"Final Fused ({fused.label_source})")
    _draw_label_bar(axes[2], fused.frame_indices, fused.labels)
    axes[2].set_xlabel("Frame Index")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_disagreement(
    fused: FusedLabels,
    subtask_label: str,
    output_path: str,
) -> None:
    """Highlight frames where CLIP and MLLM disagree."""
    if plt is None:
        return

    clip_labels = fused.clip_dominant_phase
    mllm_labels = fused.mllm_labels

    # Disagreement = both have a label and they differ
    has_mllm = mllm_labels != LABEL_UNKNOWN
    disagree = has_mllm & (clip_labels != mllm_labels)

    if not np.any(disagree):
        with open(output_path.replace(".png", ".txt"), "w") as f:
            f.write("No CLIP/MLLM disagreements found.\n")
        return

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.set_title(f'Disagreement Cases: "{subtask_label}"')

    frames = fused.frame_indices
    for i in range(len(frames)):
        color = "#95a5a6"  # gray = no info
        if disagree[i]:
            color = "#9b59b6"  # purple = disagreement
        elif has_mllm[i]:
            color = "#bdc3c7"  # light gray = agreement
        ax.barh(0, 1, left=frames[i], color=color, height=0.8, edgecolor="none")

    legend_patches = [
        mpatches.Patch(color="#9b59b6", label="Disagreement"),
        mpatches.Patch(color="#bdc3c7", label="Agreement"),
        mpatches.Patch(color="#95a5a6", label="No MLLM data"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    ax.set_xlabel("Frame Index")
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_representative_thumbnails(
    frames: list[np.ndarray],
    frame_indices: list[int],
    fused: FusedLabels,
    clip_scores_smooth: CLIPPhaseScores,
    output_dir: str,
    top_k: int = 3,
) -> None:
    """Save thumbnail strips for the top-scoring frames per phase."""
    os.makedirs(output_dir, exist_ok=True)

    # Map label values to the CLIP phase attribute to sort by
    label_to_clip_attr = {
        LABEL_PRECONDITION: "precondition",
        LABEL_CONTACT_TRANSITION: "contact_or_transition",
        LABEL_POSTCONDITION_SUCCESS: "postcondition_success",
    }

    for label_val, phase_name in LABEL_NAMES.items():
        if label_val == LABEL_UNKNOWN:
            continue

        mask = fused.labels == label_val
        if not np.any(mask):
            continue

        phase_indices = np.where(mask)[0]

        # Sort by CLIP score for pre/contact/post; by confidence for failed
        clip_attr = label_to_clip_attr.get(label_val)
        if clip_attr is not None:
            phase_scores = getattr(clip_scores_smooth, clip_attr)
            sorted_by_score = sorted(
                phase_indices, key=lambda i: phase_scores[i], reverse=True
            )
        else:
            # failed_attempt — sort by fused confidence
            sorted_by_score = sorted(
                phase_indices, key=lambda i: fused.confidence[i], reverse=True
            )

        selected = sorted_by_score[: min(top_k, len(sorted_by_score))]

        thumbs = []
        for local_idx in selected:
            if local_idx < len(frames):
                img = frames[local_idx]
                h, w = img.shape[:2]
                thumb_h = 128
                thumb_w = int(w * thumb_h / h)
                try:
                    from PIL import Image

                    pil = Image.fromarray(img).resize(
                        (thumb_w, thumb_h), Image.BILINEAR
                    )
                    thumbs.append(np.array(pil))
                except ImportError:
                    import cv2

                    thumbs.append(cv2.resize(img, (thumb_w, thumb_h)))

        if thumbs:
            max_w = max(t.shape[1] for t in thumbs)
            padded = []
            for t in thumbs:
                if t.shape[1] < max_w:
                    pad = np.zeros((t.shape[0], max_w - t.shape[1], 3), dtype=np.uint8)
                    t = np.concatenate([t, pad], axis=1)
                padded.append(t)

            strip = np.concatenate(padded, axis=0)
            imageio.imwrite(
                os.path.join(output_dir, f"top_{phase_name}.png"), strip
            )

            for j, local_idx in enumerate(selected):
                global_idx = frame_indices[local_idx]
                imageio.imwrite(
                    os.path.join(
                        output_dir,
                        f"{phase_name}_rank{j}_frame{global_idx}.png",
                    ),
                    frames[local_idx],
                )


def _draw_label_bar(
    ax: "plt.Axes",
    frame_indices: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Draw a colored horizontal bar showing labels over time."""
    all_colors = {**PHASE_COLORS}
    for i in range(len(frame_indices)):
        color = all_colors.get(int(labels[i]), "#95a5a6")
        ax.barh(0, 1, left=frame_indices[i], color=color, height=0.8, edgecolor="none")

    ax.set_yticks([])
    ax.set_xlim(frame_indices[0] - 1, frame_indices[-1] + 1)

    legend_patches = [
        mpatches.Patch(color=PHASE_COLORS[k], label=LABEL_NAMES[k])
        for k in [
            LABEL_PRECONDITION,
            LABEL_CONTACT_TRANSITION,
            LABEL_POSTCONDITION_SUCCESS,
            LABEL_FAILED_ATTEMPT,
            LABEL_UNKNOWN,
        ]
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=5)


def save_key_frame_thumbnails(
    frames: list[np.ndarray],
    frame_indices: list[int],
    fused: FusedLabels,
    candidates: list[CandidateWindow],
    output_dir: str,
) -> None:
    """Save thumbnails for key decision frames and candidate window triplets."""
    os.makedirs(output_dir, exist_ok=True)

    def _save_frame(local_idx: int, name: str) -> None:
        if 0 <= local_idx < len(frames):
            global_idx = frame_indices[local_idx]
            imageio.imwrite(
                os.path.join(output_dir, f"{name}_frame{global_idx}.png"),
                frames[local_idx],
            )

    # Key decision frames
    # Precondition: first frame of segment
    _save_frame(0, "precondition")

    # Transition start and peak
    if fused.transition_start is not None:
        local = _global_to_local(fused.transition_start, frame_indices)
        if local is not None:
            _save_frame(local, "transition_start")

    if fused.transition_frame is not None:
        local = _global_to_local(fused.transition_frame, frame_indices)
        if local is not None:
            _save_frame(local, "transition_peak")

    # Completion start and end
    if fused.completion_start is not None:
        local = _global_to_local(fused.completion_start, frame_indices)
        if local is not None:
            _save_frame(local, "completion_start")
    _save_frame(len(frames) - 1, "completion_end")

    # Each failed attempt window center
    for i, fw in enumerate(fused.failed_attempt_windows):
        _save_frame(fw.center_frame, f"failed_{i}")

    # Candidate window triplets: before / center / after
    triplet_dir = os.path.join(output_dir, "triplets")
    os.makedirs(triplet_dir, exist_ok=True)

    for cand in candidates:
        center = cand.center_frame
        n = len(frames)
        before = max(0, center - 3)
        after = min(n - 1, center + 3)

        triplet_frames = []
        triplet_labels = []
        for idx, label in [(before, "before"), (center, "center"), (after, "after")]:
            if 0 <= idx < n:
                triplet_frames.append(frames[idx])
                triplet_labels.append(label)

        if triplet_frames:
            # Resize to same height and concatenate horizontally
            target_h = 128
            resized = []
            for img in triplet_frames:
                h, w = img.shape[:2]
                target_w = int(w * target_h / h)
                try:
                    from PIL import Image as PILImage
                    pil = PILImage.fromarray(img).resize(
                        (target_w, target_h), PILImage.BILINEAR
                    )
                    resized.append(np.array(pil))
                except ImportError:
                    import cv2
                    resized.append(cv2.resize(img, (target_w, target_h)))

            strip = np.concatenate(resized, axis=1)
            global_center = frame_indices[center]
            imageio.imwrite(
                os.path.join(triplet_dir, f"cand_{global_center}_triplet.png"),
                strip,
            )


def _global_to_local(global_idx: int, frame_indices: list[int]) -> int | None:
    """Convert global frame index to local index."""
    try:
        return frame_indices.index(global_idx)
    except ValueError:
        return None


def generate_all_visualizations(
    clip_scores_raw: CLIPPhaseScores,
    clip_scores_smooth: CLIPPhaseScores,
    fused: FusedLabels,
    attempt_result: AttemptScoreResult | None,
    candidates: list[CandidateWindow],
    frames: list[np.ndarray],
    frame_indices: list[int],
    subtask_label: str,
    output_dir: str,
) -> None:
    """Generate all visualization outputs for one segment."""
    os.makedirs(output_dir, exist_ok=True)

    plot_score_timeline(
        clip_scores_raw,
        clip_scores_smooth,
        fused,
        attempt_result,
        candidates,
        subtask_label,
        os.path.join(output_dir, "score_timeline.png"),
    )

    plot_clip_vs_mllm(
        fused,
        subtask_label,
        os.path.join(output_dir, "clip_vs_mllm.png"),
    )

    plot_disagreement(
        fused,
        subtask_label,
        os.path.join(output_dir, "disagreement.png"),
    )

    save_representative_thumbnails(
        frames,
        frame_indices,
        fused,
        clip_scores_smooth,
        os.path.join(output_dir, "thumbnails"),
    )

    save_key_frame_thumbnails(
        frames,
        frame_indices,
        fused,
        candidates,
        os.path.join(output_dir, "key_frames"),
    )
