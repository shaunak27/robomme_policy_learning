"""Candidate action-attempt window detection.

Combines CLIP contact scores, visual frame differences, and robot state
changes into a single attempt score.  Peaks in this score indicate likely
action attempts.  Each peak becomes a candidate window with
start/center/end for downstream MLLM classification.

The MLLM is run only on these candidate windows, not on all frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from mme_vla_suite.pseudo_labeling.clip_scorer import CLIPPhaseScores


@dataclass
class CandidateWindow:
    """A temporal window that is a candidate for an action attempt."""

    center_frame: int      # peak frame index (local to segment, 0-based)
    start_frame: int       # window start (global timestep)
    center_global: int     # peak frame (global timestep)
    end_frame: int         # window end (global timestep)
    start_local: int       # window start (local index)
    end_local: int         # window end (local index)
    attempt_score: float   # combined attempt score at the peak
    source_contributions: dict = field(default_factory=dict)  # per-signal strength


@dataclass
class AttemptScoreResult:
    """Combined attempt score and its components."""

    attempt_score: np.ndarray         # (N,) combined score
    visual_change: np.ndarray         # (N,) normalized visual difference
    gripper_change: np.ndarray        # (N,) normalized gripper state change
    state_change: np.ndarray          # (N,) normalized joint state change
    clip_contact: np.ndarray          # (N,) normalized CLIP contact score
    frame_indices: np.ndarray         # (N,) global timestep indices


def compute_attempt_score(
    frames: list[np.ndarray],
    frame_indices: list[int],
    states: np.ndarray | None,
    clip_scores: CLIPPhaseScores | None,
    w_contact: float = 0.35,
    w_visual: float = 0.25,
    w_state: float = 0.20,
    w_gripper: float = 0.20,
    smooth_size: int = 3,
) -> AttemptScoreResult:
    """Compute a combined attempt score per frame.

    attempt_score[t] = w1 * clip_contact[t]
                     + w2 * visual_change[t]
                     + w3 * state_change[t]
                     + w4 * gripper_change[t]

    All component signals are normalized to [0, 1] before combination.
    """
    n = len(frames)

    # --- Visual frame difference ---
    visual_raw = np.zeros(n, dtype=np.float32)
    if n >= 2:
        for i in range(1, n):
            visual_raw[i] = np.mean(np.abs(
                frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)
            ))

    # --- Gripper state change ---
    gripper_raw = np.zeros(n, dtype=np.float32)
    state_raw = np.zeros(n, dtype=np.float32)
    if states is not None and states.shape[0] == n and n >= 2:
        gripper = states[:, -1]
        gripper_diff = np.abs(np.diff(gripper))
        gripper_raw[1:] = gripper_diff

        # Joint state change (exclude gripper = last column)
        joint_diff = np.linalg.norm(np.diff(states[:, :-1], axis=0), axis=1)
        state_raw[1:] = joint_diff

    # --- CLIP contact/transition score ---
    clip_contact_raw = np.zeros(n, dtype=np.float32)
    if clip_scores is not None:
        clip_contact_raw = clip_scores.contact_or_transition.copy()

    # --- Normalize each to [0, 1] ---
    def normalize(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    visual_norm = normalize(visual_raw)
    gripper_norm = normalize(gripper_raw)
    state_norm = normalize(state_raw)
    contact_norm = normalize(clip_contact_raw)

    # --- Smooth ---
    if smooth_size > 1 and n >= smooth_size:
        visual_norm = uniform_filter1d(visual_norm, size=smooth_size)
        gripper_norm = uniform_filter1d(gripper_norm, size=smooth_size)
        state_norm = uniform_filter1d(state_norm, size=smooth_size)
        contact_norm = uniform_filter1d(contact_norm, size=smooth_size)

    # --- Combine ---
    attempt = (
        w_contact * contact_norm
        + w_visual * visual_norm
        + w_state * state_norm
        + w_gripper * gripper_norm
    )

    return AttemptScoreResult(
        attempt_score=attempt,
        visual_change=visual_norm,
        gripper_change=gripper_norm,
        state_change=state_norm,
        clip_contact=contact_norm,
        frame_indices=np.array(frame_indices),
    )


def detect_candidate_windows(
    attempt_result: AttemptScoreResult,
    frame_indices: list[int],
    window_half: int = 3,
    max_candidates: int = 12,
    min_distance: int = 5,
    min_prominence: float = 0.08,
) -> list[CandidateWindow]:
    """Detect candidate windows from the combined attempt score.

    Uses peak detection with temporal non-max suppression.

    Args:
        attempt_result: Combined attempt score and components.
        frame_indices: Global timestep indices.
        window_half: Half-width of candidate windows in frames.
        max_candidates: Maximum number of candidates.
        min_distance: Minimum distance between peaks (frames).
        min_prominence: Minimum prominence for peaks.

    Returns:
        List of CandidateWindow sorted by time (center_frame).
    """
    n = len(frame_indices)
    if n < 3:
        return []

    score = attempt_result.attempt_score

    # Find peaks with prominence and distance constraints
    peaks, props = find_peaks(
        score,
        distance=min_distance,
        prominence=min_prominence,
    )

    if len(peaks) == 0:
        # Fallback: use the global maximum
        peaks = np.array([int(np.argmax(score))])
        props = {"prominences": np.array([score[peaks[0]]])}

    # Sort by attempt score descending, take top candidates
    peak_scores = score[peaks]
    order = np.argsort(-peak_scores)
    peaks = peaks[order[:max_candidates]]

    # Build candidate windows
    candidates = []
    for peak_local in sorted(peaks):
        start_local = max(0, peak_local - window_half)
        end_local = min(n - 1, peak_local + window_half)

        candidates.append(CandidateWindow(
            center_frame=int(peak_local),
            start_frame=frame_indices[start_local],
            center_global=frame_indices[peak_local],
            end_frame=frame_indices[end_local],
            start_local=start_local,
            end_local=end_local,
            attempt_score=float(score[peak_local]),
            source_contributions={
                "clip_contact": float(attempt_result.clip_contact[peak_local]),
                "visual_change": float(attempt_result.visual_change[peak_local]),
                "state_change": float(attempt_result.state_change[peak_local]),
                "gripper_change": float(attempt_result.gripper_change[peak_local]),
            },
        ))

    return candidates
