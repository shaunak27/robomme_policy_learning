"""Temporal rule-based label fusion.

Derives final frame-level phase labels from CLIP scores, candidate windows,
and MLLM classifications using temporal constraints.

Central rule:
    CLIP provides weak static evidence for 3 phases only.
    MLLM verifies success/failure over temporal windows.
    failed_attempt is derived temporally, never from CLIP argmax.

Flow:
    1. Locate the successful attempt (MLLM first, CLIP postcondition fallback).
    2. Find completion_start as stable postcondition tail after success.
    3. Backtrack from success to find the full transition region (approach
       through action through settling) using the attempt score threshold.
    4. Mark stable frames before transition as precondition.
    5. Identify failed mini-attempts: candidate windows before the successful
       transition with no stable postcondition afterward.
    6. Assign per-frame labels with priority:
       completion > transition > failed_attempt > precondition.

Phase grammar:
    precondition  ->  transition  ->  completion
    failure = contact/motion window that reverts to precondition-like state
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mme_vla_suite.pseudo_labeling.clip_scorer import CLIPPhaseScores
from mme_vla_suite.pseudo_labeling.candidate_detector import (
    CandidateWindow,
    AttemptScoreResult,
)
from mme_vla_suite.pseudo_labeling.mllm_scorer import MLLMWindowResult


# Label enum values
LABEL_PRECONDITION = 0
LABEL_CONTACT_TRANSITION = 1
LABEL_POSTCONDITION_SUCCESS = 2
LABEL_FAILED_ATTEMPT = 3
LABEL_UNKNOWN = -1

LABEL_NAMES = {
    LABEL_PRECONDITION: "precondition",
    LABEL_CONTACT_TRANSITION: "contact_or_transition",
    LABEL_POSTCONDITION_SUCCESS: "postcondition_success",
    LABEL_FAILED_ATTEMPT: "failed_attempt",
    LABEL_UNKNOWN: "unknown",
}


@dataclass
class FusedLabels:
    """Final pseudo-labels for a subtask segment."""

    frame_indices: np.ndarray          # (N,) global timestep indices
    labels: np.ndarray                 # (N,) int labels
    confidence: np.ndarray             # (N,) per-frame confidence 0-1
    success_window: CandidateWindow | None
    transition_frame: int | None       # global index of transition peak
    transition_start: int | None       # global index where transition begins
    completion_start: int | None       # global index
    failed_attempt_windows: list[CandidateWindow]
    label_source: str                  # "mllm_verified" or "clip_heuristic_only"
    metadata: dict = field(default_factory=dict)

    # Legacy compat properties
    @property
    def success_frame(self) -> int | None:
        return self.transition_frame

    @property
    def failed_attempt_frames(self) -> list[int]:
        return [w.center_global for w in self.failed_attempt_windows]

    @property
    def clip_dominant_phase(self) -> np.ndarray:
        return self.metadata.get("clip_dominant_phase", np.array([]))

    @property
    def mllm_labels(self) -> np.ndarray:
        return self.metadata.get("mllm_labels", np.full(len(self.frame_indices), LABEL_UNKNOWN))


def fuse_labels(
    clip_scores: CLIPPhaseScores,
    candidates: list[CandidateWindow],
    mllm_results: list[MLLMWindowResult],
    attempt_result: AttemptScoreResult,
    frame_indices: list[int],
    postcondition_persistence: int = 5,
) -> FusedLabels:
    """Derive final frame-level labels using temporal constraints.

    Does NOT use CLIP argmax as the label. Instead:
    1. Selects the successful attempt window (MLLM-first, CLIP-fallback).
    2. Derives completion as stable postcondition tail.
    3. Derives transition as the contiguous active region before completion.
    4. Identifies failed mini-attempts before the successful transition.
    5. Assigns labels with priority: completion > transition > failed > precondition.
    """
    n = len(frame_indices)
    decisions: dict = {"fallbacks_used": []}

    # --- Step A: Choose the successful attempt window ---
    success_window, label_source = _select_success_window(
        candidates, mllm_results, clip_scores, frame_indices,
        postcondition_persistence,
    )
    decisions["label_source"] = label_source
    if success_window is not None:
        decisions["selected_success_window"] = {
            "center_frame": success_window.center_frame,
            "center_global": success_window.center_global,
            "start_local": success_window.start_local,
            "end_local": success_window.end_local,
            "attempt_score": success_window.attempt_score,
        }
    else:
        decisions["selected_success_window"] = None

    # --- Step B: Derive completion_start ---
    completion_start_local: int | None = None
    completion_start_global: int | None = None

    if success_window is not None:
        completion_start_local = _find_stable_postcondition(
            clip_scores.postcondition_success,
            start_from=success_window.center_frame,
            persistence=postcondition_persistence,
        )
        if completion_start_local is not None:
            completion_start_global = frame_indices[completion_start_local]
            decisions["completion_start_reason"] = (
                f"stable postcondition from local frame {completion_start_local} "
                f"(global {completion_start_global}), persistence={postcondition_persistence}"
            )
        else:
            decisions["completion_start_reason"] = "no stable postcondition found"
            decisions["fallbacks_used"].append("no_stable_postcondition")

    # --- Step C: Derive transition region ---
    transition_start_local: int | None = None
    transition_peak_local: int | None = None
    transition_end_local: int | None = None
    transition_start_global: int | None = None
    transition_peak_global: int | None = None

    if success_window is not None:
        transition_start_local, transition_peak_local, transition_end_local = (
            _find_transition_region(
                attempt_result.attempt_score,
                clip_scores.contact_or_transition,
                success_center=success_window.center_frame,
                completion_start_local=completion_start_local,
                n=n,
            )
        )
        if transition_start_local is not None:
            transition_start_global = frame_indices[transition_start_local]
        if transition_peak_local is not None:
            transition_peak_global = frame_indices[transition_peak_local]
        decisions["transition_start_reason"] = (
            f"activity above threshold from local frame {transition_start_local} "
            f"(global {transition_start_global}), "
            f"peak at local {transition_peak_local} (global {transition_peak_global}), "
            f"region end at local {transition_end_local}"
        )

    # --- Step D: Derive failed_attempt windows ---
    failed_windows = _find_failed_windows(
        candidates, mllm_results, clip_scores,
        success_window, transition_start_local,
        postcondition_persistence,
    )
    decisions["failed_attempt_windows"] = [
        {"center_frame": fw.center_frame, "center_global": fw.center_global}
        for fw in failed_windows
    ]

    # --- Steps E & F: Assign per-frame labels ---
    labels, confidence = _assign_frame_labels(
        n, clip_scores, attempt_result,
        transition_start_local, transition_peak_local, transition_end_local,
        completion_start_local,
        failed_windows,
    )

    # Build CLIP-only 3-way dominant for diagnostics
    clip_matrix = np.stack([
        clip_scores.precondition,
        clip_scores.contact_or_transition,
        clip_scores.postcondition_success,
    ], axis=-1)
    clip_dominant = np.argmax(clip_matrix, axis=-1)

    # Build MLLM per-frame labels for diagnostics
    mllm_labels = np.full(n, LABEL_UNKNOWN, dtype=np.int32)
    for result in mllm_results:
        local_center = result.window.center_frame
        if local_center < n:
            if result.is_success and result.confidence >= 0.5:
                mllm_labels[local_center] = LABEL_CONTACT_TRANSITION
            elif result.is_failed_attempt and result.confidence >= 0.4:
                mllm_labels[local_center] = LABEL_FAILED_ATTEMPT

    return FusedLabels(
        frame_indices=np.array(frame_indices),
        labels=labels,
        confidence=confidence,
        success_window=success_window,
        transition_frame=transition_peak_global,
        transition_start=transition_start_global,
        completion_start=completion_start_global,
        failed_attempt_windows=failed_windows,
        label_source=label_source,
        metadata={
            "clip_dominant_phase": clip_dominant,
            "mllm_labels": mllm_labels,
            "fusion_decisions": decisions,
        },
    )


# ---------------------------------------------------------------------------
# Step A: Select the successful attempt window
# ---------------------------------------------------------------------------

def _select_success_window(
    candidates: list[CandidateWindow],
    mllm_results: list[MLLMWindowResult],
    clip_scores: CLIPPhaseScores,
    frame_indices: list[int],
    persistence: int,
) -> tuple[CandidateWindow | None, str]:
    """Select the successful attempt from MLLM results, falling back to CLIP.

    Returns (success_window, label_source).
    """
    # Try MLLM first
    mllm_successes = [
        r for r in mllm_results
        if r.is_success and r.confidence >= 0.5
    ]

    if mllm_successes:
        # Prefer: higher confidence, stronger postcondition stability
        best = max(mllm_successes, key=lambda r: (
            r.confidence,
            _postcondition_stability_after(
                clip_scores.postcondition_success,
                r.window.center_frame, persistence
            ),
        ))
        return best.window, "mllm_verified"

    # Fallback: find the candidate after which postcondition is most stable
    if not candidates:
        # No candidates at all -- use CLIP postcondition onset
        onset = _find_stable_postcondition(
            clip_scores.postcondition_success, start_from=0,
            persistence=persistence,
        )
        if onset is not None:
            # Create a synthetic window
            n = len(frame_indices)
            center = max(0, onset - 1)
            window = CandidateWindow(
                center_frame=center,
                start_frame=frame_indices[max(0, center - 3)],
                center_global=frame_indices[center],
                end_frame=frame_indices[min(n - 1, center + 3)],
                start_local=max(0, center - 3),
                end_local=min(n - 1, center + 3),
                attempt_score=0.0,
            )
            return window, "clip_heuristic_only"
        return None, "clip_heuristic_only"

    # Score each candidate by postcondition stability after it
    best_cand = max(candidates, key=lambda c: _postcondition_stability_after(
        clip_scores.postcondition_success, c.center_frame, persistence
    ))

    # Verify: does postcondition actually stabilize after this candidate?
    stability = _postcondition_stability_after(
        clip_scores.postcondition_success, best_cand.center_frame, persistence
    )
    if stability > 0:
        return best_cand, "clip_heuristic_only"

    # Last resort: candidate with highest attempt score
    return max(candidates, key=lambda c: c.attempt_score), "clip_heuristic_only"


def _postcondition_stability_after(
    post_scores: np.ndarray, after_frame: int, persistence: int
) -> float:
    """Measure how stable postcondition is after a given frame.

    Returns the mean postcondition score in the persistence window, or 0.
    """
    n = len(post_scores)
    start = min(after_frame + 1, n - 1)
    end = min(start + persistence, n)
    if end <= start:
        return 0.0
    window = post_scores[start:end]
    threshold = np.percentile(post_scores, 60)
    if np.all(window >= threshold):
        return float(window.mean())
    return 0.0


# ---------------------------------------------------------------------------
# Step B: Find stable postcondition
# ---------------------------------------------------------------------------

def _find_stable_postcondition(
    post_scores: np.ndarray,
    start_from: int,
    persistence: int,
) -> int | None:
    """First frame >= start_from where postcondition stays high for persistence frames."""
    n = len(post_scores)
    if n < persistence:
        return None

    threshold = np.percentile(post_scores, 60)

    for i in range(start_from, n - persistence + 1):
        if np.all(post_scores[i: i + persistence] >= threshold):
            return i

    # Fallback: peak postcondition after start_from
    if start_from < n:
        return int(start_from + np.argmax(post_scores[start_from:]))
    return None


# ---------------------------------------------------------------------------
# Step C: Find transition region
# ---------------------------------------------------------------------------

def _find_transition_region(
    attempt_score: np.ndarray,
    contact_scores: np.ndarray,
    success_center: int,
    completion_start_local: int | None,
    n: int,
    threshold_frac: float = 0.25,
) -> tuple[int, int, int]:
    """Find the transition region as a contiguous active region.

    The transition region covers approach, contact, movement, and settling --
    everything from when activity begins through the action completing.

    Args:
        attempt_score: (N,) combined attempt score.
        contact_scores: (N,) CLIP contact/transition score.
        success_center: Local index of the success window center.
        completion_start_local: Local index where completion begins (or None).
        n: Total number of frames.
        threshold_frac: Fraction of peak combined score used as threshold.

    Returns:
        (region_start, peak, region_end) as local indices.
        region_end is exclusive (first frame of completion or end of segment).
    """
    # Search region: from a bit before success center to completion_start
    search_end = completion_start_local if completion_start_local is not None else n
    search_end = min(search_end, n)

    # Find the peak near the success center
    peak_search_start = max(0, success_center - 5)
    if peak_search_start >= search_end:
        # Edge case: success center is at or past completion
        return success_center, success_center, min(success_center + 1, n)

    # Combined transition score in the search region
    combined = attempt_score + contact_scores

    region = combined[peak_search_start:search_end]
    if len(region) == 0:
        return success_center, success_center, min(success_center + 1, n)

    peak = peak_search_start + int(np.argmax(region))
    peak_value = combined[peak]

    if peak_value < 1e-8:
        # No meaningful activity
        return success_center, success_center, min(success_center + 1, n)

    # Threshold: fraction of peak value, but at least the 25th percentile
    threshold = max(
        peak_value * threshold_frac,
        float(np.percentile(combined, 25)),
    )

    # Walk backward from peak to find where activity begins
    region_start = peak
    for i in range(peak - 1, -1, -1):
        if combined[i] >= threshold:
            region_start = i
        else:
            break

    # Transition extends through to completion_start
    # (frames between peak and completion are settling/stabilizing)
    region_end = search_end

    # If no completion was found, cap the forward extent
    if completion_start_local is None:
        # Extend from peak to where activity drops below threshold
        region_end = peak + 1
        for i in range(peak + 1, n):
            if combined[i] >= threshold:
                region_end = i + 1
            else:
                break
        # But extend at least a few frames past peak
        region_end = max(region_end, min(peak + 4, n))

    return region_start, peak, region_end


# ---------------------------------------------------------------------------
# Step D: Find failed attempt windows
# ---------------------------------------------------------------------------

def _find_failed_windows(
    candidates: list[CandidateWindow],
    mllm_results: list[MLLMWindowResult],
    clip_scores: CLIPPhaseScores,
    success_window: CandidateWindow | None,
    transition_start_local: int | None,
    persistence: int,
) -> list[CandidateWindow]:
    """Identify failed attempt windows.

    A candidate is a failed attempt if:
    - It is a candidate window (has contact/attempt evidence)
    - It is NOT the success window
    - Its center is BEFORE the successful transition region
    - Postcondition does NOT stabilize after it
    - OR MLLM explicitly says is_failed_attempt
    """
    failed: list[CandidateWindow] = []
    success_center = success_window.center_frame if success_window else None

    # Cutoff: candidates must be before the transition region
    cutoff = transition_start_local
    if cutoff is None and success_center is not None:
        cutoff = success_center

    # Build MLLM lookup
    mllm_by_center: dict[int, MLLMWindowResult] = {}
    for r in mllm_results:
        mllm_by_center[r.window.center_frame] = r

    for cand in candidates:
        # Skip the success window itself
        if success_center is not None and cand.center_frame == success_center:
            continue

        # Only consider candidates before the successful transition
        if cutoff is not None and cand.center_frame >= cutoff:
            continue

        # Check MLLM
        mllm_r = mllm_by_center.get(cand.center_frame)
        if mllm_r is not None:
            if mllm_r.is_failed_attempt and mllm_r.confidence >= 0.4:
                failed.append(cand)
                continue
            if mllm_r.is_success and mllm_r.confidence >= 0.5:
                continue  # MLLM says success -- don't mark as failed

        # Heuristic: postcondition doesn't stabilize after this candidate
        stability = _postcondition_stability_after(
            clip_scores.postcondition_success,
            cand.center_frame, persistence,
        )
        if stability == 0.0 and cand.attempt_score > 0.15:
            failed.append(cand)

    return failed


# ---------------------------------------------------------------------------
# Steps E & F: Assign per-frame labels
# ---------------------------------------------------------------------------

def _assign_frame_labels(
    n: int,
    clip_scores: CLIPPhaseScores,
    attempt_result: AttemptScoreResult,
    transition_start_local: int | None,
    transition_peak_local: int | None,
    transition_end_local: int | None,
    completion_start_local: int | None,
    failed_windows: list[CandidateWindow],
) -> tuple[np.ndarray, np.ndarray]:
    """Assign final per-frame labels using temporal constraints.

    Priority: completion > transition > failed_attempt > precondition.
    """
    labels = np.full(n, LABEL_PRECONDITION, dtype=np.int32)
    confidence = np.full(n, 0.5, dtype=np.float32)

    # --- Completion: frames from completion_start onward ---
    if completion_start_local is not None:
        labels[completion_start_local:] = LABEL_POSTCONDITION_SUCCESS
        # Confidence from CLIP postcondition score (normalized)
        post = clip_scores.postcondition_success
        post_norm = post / (post.max() + 1e-8)
        confidence[completion_start_local:] = np.clip(
            post_norm[completion_start_local:] * 0.8 + 0.2, 0.3, 1.0
        )

    # --- Transition: contiguous active region ---
    if transition_start_local is not None and transition_end_local is not None:
        t_start = transition_start_local
        t_end = transition_end_local  # exclusive

        # Don't override completion labels
        if completion_start_local is not None:
            t_end = min(t_end, completion_start_local)

        if t_end > t_start:
            labels[t_start:t_end] = LABEL_CONTACT_TRANSITION
            # Confidence from attempt score (higher near peak)
            attempt = attempt_result.attempt_score[t_start:t_end]
            attempt_norm = attempt / (attempt.max() + 1e-8)
            confidence[t_start:t_end] = np.clip(
                attempt_norm * 0.5 + 0.4, 0.4, 0.9
            )

    # --- Failed attempt windows ---
    for fw in failed_windows:
        # Mark the window region as failed_attempt
        fail_start = fw.start_local
        fail_end = fw.end_local + 1  # exclusive

        # Only override precondition labels, never completion/transition
        for i in range(fail_start, min(fail_end, n)):
            if labels[i] == LABEL_PRECONDITION:
                labels[i] = LABEL_FAILED_ATTEMPT
                confidence[i] = 0.5

    # --- Precondition: everything before transition (already defaulted) ---
    # Boost confidence where CLIP precondition score is high
    pre = clip_scores.precondition
    pre_norm = pre / (pre.max() + 1e-8)
    for i in range(n):
        if labels[i] == LABEL_PRECONDITION:
            confidence[i] = float(np.clip(pre_norm[i] * 0.6 + 0.3, 0.3, 0.8))

    return labels, confidence
