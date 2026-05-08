"""Detect the successful interval and failed regions from a progress curve.

Given a TOPReward progress curve over prefix endpoints, finds:
  1. The completion peak (highest reward point).
  2. The ramp leading into it (the successful attempt).
  3. Dips and earlier ramps that represent failed attempts.
  4. Keyframe selection within the successful interval.

Design notes
------------
- Prefix-based reward can *drop* after the subtask completes because later
  frames show new behaviour that dilutes the task-completion signal.  We
  therefore locate the completion point at the reward **peak**, not at a
  sustained plateau.
- A "failed attempt" is any significant dip in the reward curve — either
  before the final ramp (an earlier attempt that didn't stick) or within
  the ramp itself (a temporary regression).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SuccessIntervalResult:
    """Output of success interval detection for one segment."""

    # Core interval (frame-space indices into the *frames* list, not prefix indices)
    successful_interval_start_frame: int
    completion_start_frame: int
    successful_interval_end_frame: int

    success_detected: bool

    # Failed regions: list of (start_frame, end_frame) pairs
    failed_regions: list[list[int]] = field(default_factory=list)

    # Selected keyframes (frame-space indices)
    selected_keyframes: list[int] = field(default_factory=list)

    # Confidence
    confidence: float = 0.0

    # Internals for debugging
    completion_peak_idx: int | None = None  # prefix-space index of reward peak
    ramp_start_idx: int | None = None  # prefix-space index

    def to_dict(self) -> dict:
        return {
            "successful_interval_start_frame": self.successful_interval_start_frame,
            "completion_start_frame": self.completion_start_frame,
            "successful_interval_end_frame": self.successful_interval_end_frame,
            "success_detected": self.success_detected,
            "failed_regions": self.failed_regions,
            "selected_keyframes": self.selected_keyframes,
            "confidence": round(self.confidence, 4),
            "completion_peak_idx": self.completion_peak_idx,
            "ramp_start_idx": self.ramp_start_idx,
        }


# ---------------------------------------------------------------------------
# 5.1  Find completion peak
# ---------------------------------------------------------------------------


def _find_completion_peak(
    reward: np.ndarray,
    peak_fraction: float = 0.8,
) -> int:
    """Find the last significant peak in the reward curve.

    Steps:
    1. Detect all peaks (left neighbour strictly lower, right neighbour
       lower or equal).  First and last indices are also candidates if
       they satisfy one-sided conditions.
    2. Keep only peaks whose value is >= peak_fraction * max_peak_value.
    3. Return the **last** such peak — this is the completion point.

    Using the last significant peak (rather than the global max) handles
    cases where the reward peaks mid-segment then drops, but peaks again
    later at a comparable level for the actual completion.
    """
    K = len(reward)
    if K <= 1:
        return 0

    # Detect peaks: left < peak and right <= peak
    peaks: list[int] = []

    # First index: peak if right is lower or equal
    if K >= 2 and reward[0] >= reward[1]:
        peaks.append(0)

    for i in range(1, K - 1):
        if reward[i] > reward[i - 1] and reward[i] >= reward[i + 1]:
            peaks.append(i)

    # Last index: peak if left is strictly lower
    if K >= 2 and reward[-1] > reward[-2]:
        peaks.append(K - 1)

    if not peaks:
        # No peaks found (flat curve) — fall back to argmax
        return int(np.argmax(reward))

    # Filter: keep peaks >= 0.8 * max peak value
    max_peak_val = max(reward[p] for p in peaks)
    threshold = peak_fraction * max_peak_val
    significant = [p for p in peaks if reward[p] >= threshold]

    if not significant:
        return peaks[-1]

    # Return the last significant peak
    return significant[-1]


# ---------------------------------------------------------------------------
# 5.2  Find start of final successful ramp
# ---------------------------------------------------------------------------


def _find_ramp_start(
    reward: np.ndarray,
    peak_idx: int,
) -> int:
    """Find the last significant local minimum before the peak.

    Walks backward from the peak and picks the deepest trough that
    represents a clear dip before the final rise.  Falls back to the
    global argmin in the pre-peak region.
    """
    if peak_idx == 0:
        return 0

    search = reward[: peak_idx + 1]

    # Collect local minima (strictly ≤ both neighbours)
    local_mins: list[int] = []
    for i in range(1, len(search) - 1):
        if search[i] <= search[i - 1] and search[i] <= search[i + 1]:
            local_mins.append(i)

    if local_mins:
        # Take the last local min before the peak (closest to the ramp)
        return local_mins[-1]

    # Fallback: argmin in the pre-peak region
    return int(np.argmin(search))


# ---------------------------------------------------------------------------
# 5.3  Detect failed regions (dips anywhere before the peak)
# ---------------------------------------------------------------------------


def _detect_failed_regions(
    reward: np.ndarray,
    peak_idx: int,
    prefix_end_indices: list[int],
    dip_threshold: float = 0.08,
    warmup_reward: float = 0.25,
) -> list[list[int]]:
    """Detect dips in the reward curve that represent failed attempts.

    A "dip" is any valley flanked by higher points where the drop from
    the preceding local high is at least *dip_threshold*.  We scan the
    entire curve up to (and including) the peak so that mid-ramp
    regressions are caught.

    Early prefixes where the reward is still below *warmup_reward* are
    not flagged — the model hasn't seen enough frames yet, so low scores
    are expected noise rather than genuine failed attempts.

    Parameters
    ----------
    reward : normalized reward curve (K values).
    peak_idx : index of the completion peak.
    prefix_end_indices : frame indices for each prefix endpoint.
    dip_threshold : minimum drop (preceding_high - valley) to count as
        a failed attempt.  Default 0.08 catches the 0.733→0.64 dip.
    warmup_reward : ignore dips where the preceding peak's reward is
        below this value.  Default 0.25.

    Returns
    -------
    List of [start_frame, end_frame] pairs in frame-space.
    """
    region = reward[: peak_idx + 1]
    K = len(region)
    if K < 3:
        return []

    # Find local peaks and valleys in the region up to the completion peak
    peaks: list[int] = []
    valleys: list[int] = []

    for i in range(1, K - 1):
        if region[i] >= region[i - 1] and region[i] >= region[i + 1]:
            peaks.append(i)
        if region[i] <= region[i - 1] and region[i] <= region[i + 1]:
            valleys.append(i)

    # Also treat index 0 as a potential valley
    valleys = [0] + valleys

    failed_regions: list[list[int]] = []

    for v in valleys:
        if v == 0:
            continue  # only interested in dips *after* things started rising

        # Find the preceding local high (latest peak before this valley)
        preceding_high_idx = None
        for p in peaks:
            if p < v:
                preceding_high_idx = p
        if preceding_high_idx is None:
            # No peak before this valley — check if initial value was higher
            if region[0] > region[v]:
                preceding_high_idx = 0
            else:
                continue

        # Skip warmup: if the preceding peak is still low, this isn't a
        # real failed attempt — the model just hasn't seen enough frames.
        if region[preceding_high_idx] < warmup_reward:
            continue

        drop = region[preceding_high_idx] - region[v]
        if drop < dip_threshold:
            continue

        # Find where the curve recovers after the valley (next peak or end)
        recovery_idx = v
        for p in peaks:
            if p > v:
                recovery_idx = p
                break
        else:
            recovery_idx = min(v + 1, K - 1)

        failed_start = prefix_end_indices[preceding_high_idx]
        failed_end = prefix_end_indices[min(recovery_idx, len(prefix_end_indices) - 1)]
        failed_regions.append([int(failed_start), int(failed_end)])

    return failed_regions


# ---------------------------------------------------------------------------
# Keyframe selection
# ---------------------------------------------------------------------------


def _compute_saliency(
    frames: list[np.ndarray],
    states: np.ndarray | None,
    interval_start: int,
    interval_end: int,
) -> np.ndarray:
    """Compute per-frame saliency within the successful interval.

    Returns an array of shape (interval_length,) with combined saliency.
    """
    n = interval_end - interval_start + 1
    motion = np.zeros(n)
    robot_delta = np.zeros(n)
    gripper_delta = np.zeros(n)

    for i in range(1, n):
        abs_idx = interval_start + i
        prev_idx = abs_idx - 1

        # Motion: mean absolute pixel difference
        f_curr = frames[abs_idx].astype(np.float32)
        f_prev = frames[prev_idx].astype(np.float32)
        motion[i] = np.mean(np.abs(f_curr - f_prev))

        # Robot state delta
        if states is not None and states.shape[0] > abs_idx:
            # Joint state: first 7 dims; gripper: last dim
            robot_delta[i] = np.linalg.norm(
                states[abs_idx, :7] - states[prev_idx, :7]
            )
            gripper_delta[i] = abs(
                float(states[abs_idx, -1]) - float(states[prev_idx, -1])
            )

    # Normalize each to [0, 1]
    def _norm(x: np.ndarray) -> np.ndarray:
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-8:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    motion_n = _norm(motion)
    robot_n = _norm(robot_delta)
    gripper_n = _norm(gripper_delta)

    saliency = 0.5 * motion_n + 0.3 * robot_n + 0.2 * gripper_n
    return saliency


def _visual_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Visual difference between two frames.

    Combines pixel-level difference with color histogram difference
    to catch both spatial changes and color/highlight shifts.
    """
    af = a.astype(np.float32)
    bf = b.astype(np.float32)

    # Pixel MAE (spatial structure)
    pixel_diff = float(np.mean(np.abs(af - bf)) / 255.0)

    # Color histogram difference (catches highlights, masks, saturation)
    hist_diff = 0.0
    for c in range(min(af.shape[2], 3)):
        ha = np.histogram(a[:, :, c], bins=32, range=(0, 256))[0].astype(np.float32)
        hb = np.histogram(b[:, :, c], bins=32, range=(0, 256))[0].astype(np.float32)
        ha /= ha.sum() + 1e-8
        hb /= hb.sum() + 1e-8
        hist_diff += float(np.sum(np.abs(ha - hb)))
    hist_diff /= 3.0  # normalize across channels

    return 0.5 * pixel_diff + 0.5 * hist_diff


def select_keyframes(
    frames: list[np.ndarray],
    states: np.ndarray | None,
    interval_start: int,
    completion_start: int,
    interval_end: int,
    num_keyframes: int = 5,
) -> list[int]:
    """Select 5 keyframes spanning the full segment.

    Layout (always 5, never deduplicated away):
        1. Early context  — frame near the start of the segment (before ramp),
           showing the initial scene.  Always index 0 or close to it.
        2. Ramp start     — where the successful attempt begins.
        3. Transition     — max-saliency frame between ramp start and completion.
        4. Completion     — the reward peak frame.
        5. Post-completion — most salient frame well *after* the peak, so that
           the effect of the action is visible (e.g., button actually pressed,
           cube landed in bin).  Always included.

    Parameters
    ----------
    frames : full segment frames (index 0 = segment start)
    states : (N, 8) robot states or None
    interval_start : frame index where the ramp begins
    completion_start : frame index of the reward peak
    interval_end : frame index of interval end (== completion_start)
    num_keyframes : ignored, always produces 5

    Returns
    -------
    Sorted list of 5 frame indices (into the *frames* list).
    """
    T = len(frames)

    # Compute saliency over the ramp region (interval_start → completion)
    saliency = _compute_saliency(frames, states, interval_start,
                                 min(completion_start, T - 1))

    def _saliency_at(frame_idx: int) -> float:
        rel = frame_idx - interval_start
        if 0 <= rel < len(saliency):
            return float(saliency[rel])
        return 0.0

    def _max_saliency_in_range(start: int, end: int) -> int:
        best_idx = start
        best_val = -1.0
        for f in range(start, min(end + 1, T)):
            s = _saliency_at(f)
            if s > best_val:
                best_val = s
                best_idx = f
        return best_idx

    # --- 4. Completion (peak reward) — compute first, used as reference ---
    kf_completion = min(completion_start, T - 1)
    completion_frame_data = frames[kf_completion]

    # --- 1. Early context: most visually different from completion ---
    # Search [0, 20% of segment]. If ramp_start is nonzero and smaller,
    # use that as the boundary instead.
    early_end = max(1, int(0.2 * T))
    if interval_start > 0:
        early_end = min(early_end, interval_start - 1)
    best_early = 0
    best_early_diff = -1.0
    for f in range(0, early_end + 1):
        d = _visual_diff(frames[f], completion_frame_data)
        if d > best_early_diff:
            best_early_diff = d
            best_early = f
    kf_early = best_early

    # --- 2. Ramp start ---
    kf_ramp = interval_start

    # --- 3. Transition: max saliency during the ramp ---
    kf_transition = _max_saliency_in_range(interval_start, completion_start)

    # --- 5. Post-completion: most visually different from completion ---
    # Search [peak + 25% of segment, end of segment].
    # Picks the frame that differs most from the completion frame so we
    # can see the effect of the action (button pressed, cube in bin, etc.).
    post_margin = max(5, int(0.25 * T))
    post_search_start = min(kf_completion + post_margin, T - 1)
    post_search_end = T - 1

    if post_search_start < post_search_end:
        best_post = post_search_start
        best_post_diff = -1.0
        for f in range(post_search_start, post_search_end + 1):
            d = _visual_diff(frames[f], completion_frame_data)
            if d > best_post_diff:
                best_post_diff = d
                best_post = f
        kf_post = best_post
    else:
        kf_post = T - 1

    # Collapse duplicates but keep all 5 slots filled.
    # If two keyframes land on the same index, nudge the later one forward.
    raw = [kf_early, kf_ramp, kf_transition, kf_completion, kf_post]
    final: list[int] = []
    for idx in raw:
        idx = max(0, min(idx, T - 1))
        # Nudge forward if duplicate
        while idx in final and idx < T - 1:
            idx += 1
        final.append(idx)

    return sorted(set(final))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def detect_success_interval(
    reward_smooth: np.ndarray,
    prefix_end_indices: list[int],
    total_frames: int,
    frames: list[np.ndarray] | None = None,
    states: np.ndarray | None = None,
    num_keyframes: int = 3,
) -> SuccessIntervalResult:
    """Detect the successful interval from a TOPReward progress curve.

    Parameters
    ----------
    reward_smooth : 1-D array of normalized rewards (K values).
        Despite the parameter name (kept for API compat), this should be
        raw normalized reward — no smoothing.
    prefix_end_indices : list of frame indices corresponding to each prefix endpoint.
    total_frames : total number of frames in the segment (T).
    frames : optional full frames list (needed for keyframe selection).
    states : optional (T, 8) robot states (needed for keyframe selection).
    num_keyframes : number of keyframes to select.

    Returns
    -------
    SuccessIntervalResult
    """
    K = len(reward_smooth)
    reward = reward_smooth  # alias for clarity

    # 5.1 Find completion peak
    peak_idx = _find_completion_peak(reward)
    peak_val = float(reward[peak_idx])

    # Check if this looks like a real success
    # A peak below 0.4 on the normalized curve is unlikely to be a real completion
    success_detected = peak_val >= 0.4

    if not success_detected:
        fallback_frame = prefix_end_indices[peak_idx]
        pad = max(2, int(0.05 * total_frames))
        result = SuccessIntervalResult(
            successful_interval_start_frame=max(0, fallback_frame - pad),
            completion_start_frame=fallback_frame,
            successful_interval_end_frame=total_frames - 1,
            success_detected=False,
            confidence=round(0.3 * peak_val, 4),
            completion_peak_idx=peak_idx,
        )
        if frames is not None:
            result.selected_keyframes = select_keyframes(
                frames, states,
                result.successful_interval_start_frame,
                result.completion_start_frame,
                result.successful_interval_end_frame,
                num_keyframes,
            )
        return result

    # 5.2 Find ramp start (last local min before the peak)
    ramp_idx = _find_ramp_start(reward, peak_idx)

    # Map to frame space
    pad = max(2, int(0.05 * total_frames))
    interval_start_frame = max(0, prefix_end_indices[ramp_idx] - pad)
    completion_start_frame = prefix_end_indices[peak_idx]
    # End at the peak frame — frames after completion are post-task
    interval_end_frame = prefix_end_indices[peak_idx]

    # 5.3 Detect failed regions (dips anywhere up to the peak)
    failed_regions = _detect_failed_regions(
        reward, peak_idx, prefix_end_indices,
    )

    # Confidence
    ramp_gain = float(reward[peak_idx] - reward[ramp_idx])
    # How sharp is the peak relative to the rest of the curve?
    mean_reward = float(np.mean(reward))
    peak_prominence = peak_val - mean_reward

    confidence = (
        0.4 * peak_val
        + 0.4 * float(np.clip(ramp_gain, 0, 1))
        + 0.2 * float(np.clip(peak_prominence, 0, 1))
    )

    result = SuccessIntervalResult(
        successful_interval_start_frame=interval_start_frame,
        completion_start_frame=completion_start_frame,
        successful_interval_end_frame=interval_end_frame,
        success_detected=True,
        failed_regions=failed_regions,
        confidence=round(confidence, 4),
        completion_peak_idx=peak_idx,
        ramp_start_idx=ramp_idx,
    )

    # Keyframe selection
    if frames is not None:
        result.selected_keyframes = select_keyframes(
            frames, states,
            interval_start_frame,
            completion_start_frame,
            interval_end_frame,
            num_keyframes,
        )

    return result
