"""Target distribution construction for QKFS training.

For each training timestep t, builds:
1. P_target[j]: source-subtask distribution — which past subtasks to attend to.
2. T_j(i): within-subtask frame distribution — which frames inside each
   relevant subtask matter (soft keyframe targets + uniform filler).

Uses:
- Subtask segment annotations (segments.json)
- Sampling rules (which past subtasks are relevant)
- TOPReward success intervals and heuristic keyframes (density_keyframes.json)
- Soft Gaussian neighborhoods around key events
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from mme_vla_suite.shared.sampling_rules import get_sampling_sources, _is_completed

logger = logging.getLogger(__name__)


def load_topreward_intervals(
    topreward_dir: str,
    task_name: str,
    epis_idx: int,
    segments: list[dict],
) -> dict[int, dict]:
    """Load TOPReward success intervals for each segment in an episode.

    Returns mapping: seg_idx -> {
        "start": int,  # success interval start (relative to segment)
        "end": int,    # success interval end (relative to segment)
        "keyframes": list[int],  # keyframe indices (relative to segment)
    }
    """
    topreward_path = Path(topreward_dir)
    result = {}

    for seg in segments:
        if _is_completed(seg["label"]):
            continue
        seg_key = f"{task_name}_ep{epis_idx}_f{seg['start_frame']}-{seg['end_frame']}"
        interval_file = topreward_path / seg_key / "successful_interval.json"

        if interval_file.exists():
            with open(interval_file) as f:
                data = json.load(f)
            result[seg["idx"]] = {
                "start": data.get("successful_interval_start_frame", 0),
                "end": data.get("successful_interval_end_frame", seg["end_frame"] - seg["start_frame"]),
                "keyframes": data.get("selected_keyframes", []),
                "success": data.get("success_detected", False),
            }
        else:
            # No TOPReward data — will use density_keyframes fallback
            result[seg["idx"]] = None

    return result


def build_soft_keyframe_distribution(
    seg_start: int,
    seg_end: int,
    keyframe_centers: list[int],
    sigma: float,
    topreward_window: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build K_j(i): soft key-event distribution inside a subtask.

    For each key event center k, creates a Gaussian G_k(i) and
    restricts it to the TOPReward window if available.

    Args:
        seg_start: absolute start frame of segment
        seg_end: absolute end frame of segment (inclusive)
        keyframe_centers: absolute frame indices of key events
        sigma: Gaussian width
        topreward_window: (start, end) absolute frames of success window, or None

    Returns:
        K_j: normalized distribution over frames [seg_start, seg_end], shape (num_frames,)
    """
    num_frames = seg_end - seg_start + 1
    if num_frames <= 0:
        return np.zeros(0)

    frame_indices = np.arange(seg_start, seg_end + 1)
    K_j = np.zeros(num_frames, dtype=np.float64)

    for k in keyframe_centers:
        # Gaussian centered at k
        G_k = np.exp(-((frame_indices - k) ** 2) / (2 * sigma ** 2))

        # Restrict to TOPReward window if available
        if topreward_window is not None:
            w_start, w_end = topreward_window
            window_mask = (frame_indices >= w_start) & (frame_indices <= w_end)
            G_k = G_k * window_mask
        else:
            # Small radius fallback: only apply within +-3*sigma of center
            radius = int(3 * sigma)
            radius_mask = np.abs(frame_indices - k) <= radius
            G_k = G_k * radius_mask

        K_j += G_k

    # Normalize
    total = K_j.sum()
    if total > 0:
        K_j /= total

    return K_j


def build_within_subtask_target(
    seg_start: int,
    seg_end: int,
    keyframe_centers: list[int],
    sigma: float,
    budget: int,
    num_key_events: int,
    topreward_window: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build T_j(i) = rho_j * K_j(i) + (1 - rho_j) * U_j(i).

    Args:
        seg_start, seg_end: absolute frame range (inclusive)
        keyframe_centers: absolute frame indices of key events in this subtask
        sigma: Gaussian width for soft keyframe targets
        budget: B_j — how many frames allocated to this subtask
        num_key_events: M_j — number of key events
        topreward_window: optional (start, end) window

    Returns:
        T_j: normalized distribution over frames [seg_start, seg_end], shape (num_frames,)
    """
    num_frames = seg_end - seg_start + 1
    if num_frames <= 0:
        return np.zeros(0)

    # Compute rho_j = min(M_j, B_j) / B_j
    if budget > 0:
        rho_j = min(num_key_events, budget) / budget
    else:
        rho_j = 0.5

    # K_j: soft keyframe distribution
    K_j = build_soft_keyframe_distribution(
        seg_start, seg_end, keyframe_centers, sigma, topreward_window
    )

    # U_j: uniform distribution over segment frames
    U_j = np.ones(num_frames, dtype=np.float64) / num_frames

    # Combine
    T_j = rho_j * K_j + (1 - rho_j) * U_j

    # Normalize
    total = T_j.sum()
    if total > 0:
        T_j /= total

    return T_j


def build_target_distributions(
    step_idx: int,
    segments: list[dict],
    task_name: str,
    task_goal: str,
    density_keyframes: dict[int, list[int]],
    topreward_intervals: dict[int, dict | None],
    sigma: float,
    max_frames: int = 32,
    epsilon: float = 1e-6,
) -> dict:
    """Build complete target distributions for a training timestep.

    Returns:
        {
            "source_seg_indices": list[int],   # A_t: relevant source subtask indices
            "P_target": dict[int, float],      # source-subtask distribution
            "T_j": dict[int, np.ndarray],      # per-subtask frame distribution
            "seg_budgets": dict[int, int],      # B_j per subtask
        }
    """
    # Find current segment
    current_seg = None
    for seg in segments:
        if seg["start_frame"] <= step_idx <= seg["end_frame"]:
            current_seg = seg
            break

    if current_seg is None or _is_completed(current_seg.get("label", "")):
        return None

    # Get relevant source subtasks from sampling rules
    source_map = get_sampling_sources(task_name, segments, task_goal=task_goal)
    current_seg_idx = current_seg["idx"]
    source_seg_indices = source_map.get(current_seg_idx, [])

    # Filter to only past segments (frames before step_idx)
    source_seg_indices = [
        idx for idx in source_seg_indices
        if any(s["idx"] == idx and s["start_frame"] < step_idx for s in segments)
    ]

    if not source_seg_indices:
        # No specific sources — VLA falls back to uniform over all past frames.
        # Build a uniform target across all past segments.
        all_past = [
            s["idx"] for s in segments
            if s["start_frame"] < step_idx and not _is_completed(s.get("label", ""))
        ]
        if not all_past:
            return None
        source_seg_indices = all_past
        # Flag this as a uniform fallback (no keyframe weighting)
        uniform_fallback = True
    else:
        uniform_fallback = False

    # Build P_target
    seg_lookup = {s["idx"]: s for s in segments}
    n_sources = len(source_seg_indices)

    if uniform_fallback:
        # VLA does np.linspace(0, step_idx, max_frames) — uniform over ALL
        # past frames, ignoring segment boundaries.  Weight each segment
        # proportionally to its number of past frames so the combined
        # distribution P_target[j] * T_j(i) is truly uniform.
        seg_past_frames = {}
        total_past = 0
        for idx in source_seg_indices:
            seg = seg_lookup[idx]
            n_past = min(seg["end_frame"], step_idx - 1) - seg["start_frame"] + 1
            n_past = max(n_past, 0)
            seg_past_frames[idx] = n_past
            total_past += n_past
        if total_past == 0:
            return None
        P_target = {idx: seg_past_frames[idx] / total_past for idx in source_seg_indices}
    else:
        # Equal weight across relevant source subtasks
        w_j = 1.0 / n_sources
        P_target = {idx: w_j for idx in source_seg_indices}

    # Allocate frame budgets per subtask proportional to weight
    seg_budgets = {}
    if uniform_fallback:
        # Proportional to past frame count
        allocated = 0
        for i, idx in enumerate(source_seg_indices):
            b = int(round(P_target[idx] * max_frames))
            seg_budgets[idx] = max(b, 1) if P_target[idx] > 0 else 0
            allocated += seg_budgets[idx]
        # Adjust for rounding
        diff = max_frames - allocated
        if diff != 0 and source_seg_indices:
            seg_budgets[source_seg_indices[0]] += diff
    else:
        per_seg_budget = max_frames // n_sources
        remainder = max_frames % n_sources
        for i, idx in enumerate(source_seg_indices):
            seg_budgets[idx] = per_seg_budget + (1 if i < remainder else 0)

    # Build T_j for each relevant source subtask
    T_j_map = {}

    for seg_idx in source_seg_indices:
        seg = seg_lookup[seg_idx]
        seg_start = seg["start_frame"]
        seg_end = min(seg["end_frame"], step_idx - 1)  # only past frames

        if seg_end < seg_start:
            # No past frames in this segment
            T_j_map[seg_idx] = np.zeros(0)
            continue

        num_frames = seg_end - seg_start + 1

        if uniform_fallback:
            # VLA uses uniform_linspace for subtasks with no assigned sources:
            # pure uniform distribution over all past frames, no keyframe weighting.
            T_j = np.ones(num_frames, dtype=np.float64) / num_frames
        else:
            # Get keyframes for this segment
            raw_kfs = density_keyframes.get(str(seg_idx), density_keyframes.get(seg_idx, []))
            # density_keyframes are relative to segment start
            abs_keyframes = [seg["start_frame"] + k for k in raw_kfs]
            # Filter to only past keyframes
            abs_keyframes = [k for k in abs_keyframes if k <= seg_end]

            # Get TOPReward window
            topreward_window = None
            tr_info = topreward_intervals.get(seg_idx)
            if tr_info is not None and tr_info.get("success", False):
                tr_start = seg["start_frame"] + tr_info["start"]
                tr_end = seg["start_frame"] + tr_info["end"]
                # Clamp to past
                tr_end = min(tr_end, seg_end)
                if tr_start <= tr_end:
                    topreward_window = (tr_start, tr_end)

            B_j = seg_budgets[seg_idx]
            M_j = len(abs_keyframes)

            T_j = build_within_subtask_target(
                seg_start, seg_end, abs_keyframes, sigma, B_j, M_j, topreward_window
            )

        T_j_map[seg_idx] = T_j

    return {
        "source_seg_indices": source_seg_indices,
        "P_target": P_target,
        "T_j": T_j_map,
        "seg_budgets": seg_budgets,
        "current_seg_idx": current_seg_idx,
    }


def compute_qkfs_loss(
    log_probs: np.ndarray,           # (N,) log-probabilities over candidate frames
    candidate_frame_indices: np.ndarray,  # (N,) absolute frame index for each candidate
    cand_mask: np.ndarray,           # (N,) bool
    targets: dict,
    segments: list[dict],
    epsilon: float = 1e-6,
    lambda_frame: float = 1.0,
) -> tuple[float, dict]:
    """Compute QKFS training loss = L_src + lambda * L_frame.

    Args:
        log_probs: model's log-probabilities over candidate frames (N,)
        candidate_frame_indices: absolute frame index for each candidate position (N,)
        cand_mask: which candidates are valid (N,)
        targets: output of build_target_distributions()
        segments: full segment list
        epsilon: numerical stability
        lambda_frame: weight for within-subtask loss

    Returns:
        (total_loss, info_dict)
    """
    if targets is None:
        return 0.0, {"L_src": 0.0, "L_frame": 0.0}

    seg_lookup = {s["idx"]: s for s in segments}
    source_seg_indices = targets["source_seg_indices"]
    P_target = targets["P_target"]
    T_j_map = targets["T_j"]

    probs = np.exp(np.array(log_probs))
    probs = np.where(cand_mask, probs, 0.0)

    # ---- Loss 1: Source-subtask loss ----
    # Aggregate predicted probabilities by subtask
    P_pred = {}
    all_seg_indices = set(s["idx"] for s in segments)
    for seg_idx in all_seg_indices:
        seg = seg_lookup[seg_idx]
        # Find candidates that fall within this segment
        in_seg = (candidate_frame_indices >= seg["start_frame"]) & \
                 (candidate_frame_indices <= seg["end_frame"]) & cand_mask
        P_pred[seg_idx] = float(np.sum(probs[in_seg]))

    # Normalize P_pred to sum to 1
    total_pred = sum(P_pred.values()) + epsilon
    P_pred = {k: v / total_pred for k, v in P_pred.items()}

    # KL(P_target || P_pred)
    L_src = 0.0
    for seg_idx in source_seg_indices:
        p_t = P_target.get(seg_idx, 0.0)
        p_p = P_pred.get(seg_idx, epsilon)
        p_p = max(p_p, epsilon)
        if p_t > 0:
            L_src += p_t * np.log(p_t / p_p)

    # ---- Loss 2: Within-subtask frame loss ----
    L_frame = 0.0
    for seg_idx in source_seg_indices:
        seg = seg_lookup[seg_idx]
        T_j = T_j_map.get(seg_idx)
        if T_j is None or len(T_j) == 0:
            continue

        seg_start = seg["start_frame"]
        seg_end = seg_start + len(T_j) - 1

        # Get predicted probabilities inside this subtask
        in_seg = (candidate_frame_indices >= seg_start) & \
                 (candidate_frame_indices <= seg_end) & cand_mask

        if not np.any(in_seg):
            continue

        # Build predicted distribution within this subtask
        seg_probs = probs[in_seg]
        seg_frame_ids = candidate_frame_indices[in_seg]
        total_seg = seg_probs.sum() + epsilon
        p_pred_in_j = seg_probs / total_seg

        # Map T_j to the same frame indices
        # T_j is indexed by (frame - seg_start)
        t_j_values = T_j[seg_frame_ids - seg_start]

        # Normalize t_j_values (they should already be normalized over full segment,
        # but we need to re-normalize over just the candidate frames)
        t_j_sum = t_j_values.sum() + epsilon
        t_j_norm = t_j_values / t_j_sum

        # KL(T_j || p_pred_in_j)
        w_j = P_target.get(seg_idx, 0.0)
        kl_j = 0.0
        for k in range(len(t_j_norm)):
            if t_j_norm[k] > epsilon:
                p_pred_k = max(float(p_pred_in_j[k]), epsilon)
                kl_j += t_j_norm[k] * np.log(t_j_norm[k] / p_pred_k)
        L_frame += w_j * kl_j

    total_loss = L_src + lambda_frame * L_frame

    return total_loss, {"L_src": L_src, "L_frame": L_frame}


def compute_qkfs_loss_jax(
    log_probs: "jnp.ndarray",
    candidate_frame_indices: "jnp.ndarray",
    cand_mask: "jnp.ndarray",
    target_frame_dist: "jnp.ndarray",
    target_seg_assignment: "jnp.ndarray",
    target_seg_weights: "jnp.ndarray",
    num_segments: int,
    epsilon: float = 1e-6,
    lambda_frame: float = 1.0,
) -> tuple["jnp.ndarray", dict]:
    """JAX-differentiable QKFS loss computation.

    Args:
        log_probs: (B, N) log-probabilities from the model
        candidate_frame_indices: (B, N) int32 — not used directly, see target arrays
        cand_mask: (B, N) bool
        target_frame_dist: (B, N) float — per-frame target probability (T_j mapped to candidates)
        target_seg_assignment: (B, N) int32 — which segment each candidate belongs to
        target_seg_weights: (B, S) float — P_target[j] for each segment index
        num_segments: S — max number of segments
        epsilon: numerical stability
        lambda_frame: weight for L_frame

    Returns:
        (scalar_loss, info_dict)
    """
    import jax
    import jax.numpy as jnp

    probs = jnp.exp(log_probs)  # (B, N)
    cand_mask_f = cand_mask.astype(jnp.float32)
    probs = probs * cand_mask_f

    B, N = probs.shape
    S = num_segments

    # ---- L_src: source-subtask KL ----
    # Aggregate probs by segment: P_pred[b, s] = sum of probs for frames in segment s
    seg_one_hot = jax.nn.one_hot(target_seg_assignment, S)  # (B, N, S)
    seg_one_hot = seg_one_hot * cand_mask_f[:, :, None]
    P_pred = jnp.sum(probs[:, :, None] * seg_one_hot, axis=1)  # (B, S)
    P_pred = P_pred / (P_pred.sum(axis=-1, keepdims=True) + epsilon)
    P_pred = jnp.maximum(P_pred, epsilon)

    # KL(P_target || P_pred)
    P_target = target_seg_weights  # (B, S)
    P_target_safe = jnp.maximum(P_target, epsilon)
    # Compute log ratio safely
    log_ratio_src = jnp.log(P_target_safe) - jnp.log(P_pred)
    kl_src = P_target * log_ratio_src
    # Zero out where target is zero
    kl_src = jnp.where(P_target > epsilon, kl_src, 0.0)
    L_src = jnp.sum(kl_src, axis=-1)  # (B,)

    # ---- L_frame: within-subtask KL ----
    # Fully vectorized across segments using one_hot indexing
    # seg_one_hot: (B, N, S), probs: (B, N)

    # Per-segment predicted probs: (B, N, S) — probs multiplied by segment membership
    seg_probs = probs[:, :, None] * seg_one_hot  # (B, N, S)
    seg_totals = jnp.sum(seg_probs, axis=1, keepdims=True) + epsilon  # (B, 1, S)
    p_pred_in_j = seg_probs / seg_totals  # (B, N, S) — normalized within each segment
    p_pred_in_j = jnp.maximum(p_pred_in_j, epsilon)

    # Per-segment target distribution: (B, N, S)
    t_j = target_frame_dist[:, :, None] * seg_one_hot  # (B, N, S)
    t_j_totals = jnp.sum(t_j, axis=1, keepdims=True) + epsilon  # (B, 1, S)
    t_j_norm = t_j / t_j_totals  # (B, N, S)

    # KL(T_j || p_pred_in_j) per segment, per frame
    log_ratio = jnp.log(jnp.maximum(t_j_norm, epsilon)) - jnp.log(p_pred_in_j)
    kl_per_frame = t_j_norm * log_ratio  # (B, N, S)
    # Zero out where target is negligible
    kl_per_frame = jnp.where(t_j_norm > epsilon, kl_per_frame, 0.0)
    kl_per_seg = jnp.sum(kl_per_frame, axis=1)  # (B, S)

    # Weight by P_target[j] and sum across segments
    L_frame = jnp.sum(P_target * kl_per_seg, axis=-1)  # (B,)

    total_loss = jnp.mean(L_src + lambda_frame * L_frame)

    return total_loss, {
        "L_src": jnp.mean(L_src),
        "L_frame": jnp.mean(L_frame),
        "total_loss": total_loss,
    }
