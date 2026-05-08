"""Segment-aware frame sampling with per-subtask density strategies.

Given an exec subtask's source segments (from sampling_rules.py) and
per-segment keyframes (from the topreward pipeline), this module decides
which frame indices to load into the memory buffer.

Frame budget: 32 frames (512 tokens / 16 tokens-per-image / 1 view).

Strategies
----------
A  equal_weight_keyframes
     Use keyframes from all chosen source segments, distributing the
     frame budget equally across segments.

B  uniform_linspace
     Ignore keyframes.  Uniformly sample (linspace) from ALL frames
     before the current step, regardless of source segments.

C  priority_keyframes_then_equal
     Include ALL keyframes from one or more priority segments first,
     then distribute remaining budget equally (via keyframes) across
     the other chosen source segments.

D  priority_keyframes_then_uniform
     Include ALL keyframes from a priority segment first, then fill
     remaining budget with uniformly spaced frames (linspace) from
     the full past.

E  exec_keyframes_then_uniform_demo
     Use keyframes from chosen exec segments (equal weight), fill
     remaining budget with uniformly spaced frames across chosen
     demo segments.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np

from mme_vla_suite.shared.sampling_rules import (
    TASK_RULES,
    _exec_segs,
    _is_completed,
    _matching_demo,
    get_sampling_sources,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_FRAMES = 32  # 512 tokens / 16 tokens_per_image

Strategy = Literal[
    "equal_weight_keyframes",
    "uniform_linspace",
    "priority_keyframes_then_equal",
    "priority_keyframes_then_uniform",
    "exec_keyframes_then_uniform_demo",
]

# ---------------------------------------------------------------------------
# Colors / ordinals for category matching (mirrors taxonomy_viewer.py)
# ---------------------------------------------------------------------------

_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "cyan",
    "white", "black", "brown", "gray", "grey",
]
_ORDINALS = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
]


def _normalize_label(label: str) -> str:
    result = label
    for color in _COLORS:
        result = re.sub(rf'\b{color}\b', '{color}', result)
    for ordinal in _ORDINALS:
        result = re.sub(rf'\b{ordinal}\b', '{N}', result)
    result = re.sub(r'\{N\}\s+\{color\}', '{N} {color}', result)
    result = re.sub(r'\b\d+(st|nd|rd|th)\b', '{N}', result)
    result = re.sub(r'for \d+ time', 'for {N} time', result)
    return result


# ---------------------------------------------------------------------------
# Per-task density dispatch
# ---------------------------------------------------------------------------
# Maps (task_name, generic_exec_label) -> (strategy, priority_match_fn | None)
#
# priority_match_fn(seg, segments) -> bool   tells whether a source segment
# is the "priority" segment for strategy C/D.
# ---------------------------------------------------------------------------

def _label_contains(keyword: str):
    """Return a matcher that checks if a segment label contains *keyword*."""
    def match(seg: dict, segments: list[dict]) -> bool:
        return keyword in seg["label"]
    return match


def _phase_is(phase: str):
    """Return a matcher that checks the segment phase."""
    def match(seg: dict, segments: list[dict]) -> bool:
        return seg["phase"] == phase
    return match


def _is_corresponding_demo(exec_label: str):
    """Return a matcher for the demo segment whose label best matches exec_label."""
    def match(seg: dict, segments: list[dict]) -> bool:
        if seg["phase"] != "demo":
            return False
        matching_idxs = _matching_demo(exec_label, segments)
        return seg["idx"] in matching_idxs
    return match


def _always(_seg, _segs):
    return True


# ---------------------------------------------------------------------------
# Build the full dispatch table
# ---------------------------------------------------------------------------

def _build_density_table() -> dict[tuple[str, str], tuple[Strategy, object]]:
    """Build the (task, generic_label) -> (strategy, priority_fn) table."""
    T = {}

    # --- BinFill ---
    T[("BinFill", "pick up the {N} {color} cube")] = ("equal_weight_keyframes", None)
    T[("BinFill", "press the button")] = ("equal_weight_keyframes", None)
    T[("BinFill", "put it into the bin")] = ("uniform_linspace", None)

    # --- ButtonUnmask ---
    T[("ButtonUnmask", "pick up the container that hides the {color} cube")] = (
        "priority_keyframes_then_equal", _label_contains("press"))
    T[("ButtonUnmask", "press the button")] = ("uniform_linspace", None)
    T[("ButtonUnmask", "put down the container")] = ("uniform_linspace", None)

    # --- ButtonUnmaskSwap ---
    T[("ButtonUnmaskSwap", "pick up the container that hides the {color} cube")] = (
        "priority_keyframes_then_equal", _label_contains("press"))
    T[("ButtonUnmaskSwap", "press the {N} button")] = (
        "priority_keyframes_then_equal", _label_contains("first button"))
    T[("ButtonUnmaskSwap", "put down the container")] = ("uniform_linspace", None)

    # --- InsertPeg ---
    for lbl in [
        "insert the peg from the left side",
        "insert the peg from the right side",
        "pick up the peg by grasping the far end",
        "pick up the peg by grasping the near end",
    ]:
        T[("InsertPeg", lbl)] = ("priority_keyframes_then_uniform", _phase_is("demo"))

    # --- MoveCube ---
    for lbl in [
        "close the gripper and push the cube to the target",
        "hook the cube to the target with the peg",
        "pick up the cube",
        "pick up the peg",
        "place the cube onto the target",
    ]:
        T[("MoveCube", lbl)] = ("equal_weight_keyframes", None)

    # --- PatternLock ---
    for direction in [
        "move backward", "move backward-left", "move backward-right",
        "move forward", "move forward-left", "move forward-right",
        "move left", "move right",
    ]:
        T[("PatternLock", direction)] = ("equal_weight_keyframes", None)

    # --- PickHighlight ---
    T[("PickHighlight", "pick up the highlighted cube, which is {color}")] = (
        "priority_keyframes_then_equal", _label_contains("press"))
    T[("PickHighlight", "pick up the {N} highlighted cube, which is {color}")] = (
        "priority_keyframes_then_equal", _label_contains("press"))
    T[("PickHighlight", "place the cube onto the table")] = ("uniform_linspace", None)
    T[("PickHighlight", "press the button")] = ("uniform_linspace", None)

    # --- PickXtimes ---
    T[("PickXtimes", "pick up the {color} cube for the {N} time")] = ("equal_weight_keyframes", None)
    T[("PickXtimes", "place the {color} cube onto the target")] = ("uniform_linspace", None)
    T[("PickXtimes", "press the button to stop")] = ("equal_weight_keyframes", None)

    # --- RouteStick ---
    for lbl in [
        "move to the nearest left target by circling around the stick clockwise",
        "move to the nearest left target by circling around the stick counterclockwise",
        "move to the nearest right target by circling around the stick clockwise",
        "move to the nearest right target by circling around the stick counterclockwise",
    ]:
        T[("RouteStick", lbl)] = ("priority_keyframes_then_equal", _is_corresponding_demo(lbl))

    # --- StopCube ---
    T[("StopCube", "move to the top of the button to prepare")] = ("uniform_linspace", None)
    T[("StopCube", "press the button to stop the cube on the target")] = ("uniform_linspace", None)
    T[("StopCube", "remain static")] = ("uniform_linspace", None)

    # --- SwingXtimes ---
    for lbl in [
        "move to the top of the left-side target for the {N} time",
        "move to the top of the right-side target for the {N} time",
        "pick up the {color} cube",
        "press the button",
        "put the {color} cube on the table",
    ]:
        T[("SwingXtimes", lbl)] = ("equal_weight_keyframes", None)

    # --- VideoPlaceButton ---
    T[("VideoPlaceButton", "pick up the cube")] = (
        "priority_keyframes_then_uniform", _phase_is("demo"))
    T[("VideoPlaceButton", "place the cube onto the correct target")] = (
        "priority_keyframes_then_uniform", _phase_is("demo"))

    # --- VideoPlaceOrder ---
    T[("VideoPlaceOrder", "pick up the cube")] = ("equal_weight_keyframes", None)
    T[("VideoPlaceOrder", "place the cube onto the correct target")] = ("equal_weight_keyframes", None)

    # --- VideoRepick ---
    T[("VideoRepick", "pick up the correct cube for the {N} time")] = ("equal_weight_keyframes", None)
    T[("VideoRepick", "press the button to finish")] = (
        "exec_keyframes_then_uniform_demo", None)
    T[("VideoRepick", "put it down")] = ("uniform_linspace", None)

    # --- VideoUnmask ---
    T[("VideoUnmask", "pick up the container that hides the {color} cube")] = (
        "priority_keyframes_then_equal", _phase_is("demo"))
    T[("VideoUnmask", "put down the container")] = ("uniform_linspace", None)

    # --- VideoUnmaskSwap ---
    T[("VideoUnmaskSwap", "pick up the container that hides the {color} cube")] = (
        "priority_keyframes_then_equal", _phase_is("demo"))
    T[("VideoUnmaskSwap", "put down the container")] = ("uniform_linspace", None)

    return T


DENSITY_TABLE = _build_density_table()


def get_density_rule(
    task_name: str, exec_label: str
) -> tuple[Strategy, object]:
    """Look up the density strategy for a given task + exec subtask label."""
    generic = _normalize_label(exec_label)
    key = (task_name, generic)
    if key in DENSITY_TABLE:
        return DENSITY_TABLE[key]
    raise ValueError(
        f"No density rule for task={task_name}, label={exec_label!r} "
        f"(generic={generic!r})"
    )


# ---------------------------------------------------------------------------
# Core frame-index computation
# ---------------------------------------------------------------------------

def _collect_segment_keyframes(
    seg: dict,
    keyframes_by_seg: dict[int, list[int]],
) -> list[int]:
    """Get absolute-frame keyframe indices for a segment.

    keyframes_by_seg maps segment idx -> list of keyframe indices
    (relative to segment start_frame).
    """
    seg_idx = seg["idx"]
    start = seg["start_frame"]
    if seg_idx in keyframes_by_seg:
        return sorted(start + k for k in keyframes_by_seg[seg_idx])
    return []


def _uniform_indices(start: int, end: int, n: int) -> list[int]:
    """Uniformly spaced indices in [start, end] inclusive."""
    if n <= 0:
        return []
    if end <= start:
        return [start] if n > 0 else []
    if n == 1:
        return [(start + end) // 2]
    return np.linspace(start, end, n, dtype=np.int64).tolist()


def compute_frame_indices(
    task_name: str,
    step_idx: int,
    segments: list[dict],
    source_seg_indices: list[int],
    keyframes_by_seg: dict[int, list[int]],
    task_goal: str = "",
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[int]:
    """Compute which frame indices to load for a given exec step.

    Parameters
    ----------
    task_name : str
        e.g. "BinFill"
    step_idx : int
        Current absolute timestep index within the episode.
    segments : list[dict]
        Full list of segment dicts for the episode (both demo and exec).
        Each dict has keys: idx, phase, label, start_frame, end_frame.
    source_seg_indices : list[int]
        Segment indices selected by sampling_rules.py for this exec step.
    keyframes_by_seg : dict[int, list[int]]
        Mapping from segment idx to list of keyframe frame indices
        (relative to the segment's start_frame).
    task_goal : str
        Task goal string (used for VideoPlaceButton).
    max_frames : int
        Frame budget (default 32).

    Returns
    -------
    list[int]
        Sorted absolute frame indices to load, length <= max_frames.
    """
    # Find which exec segment this step belongs to
    current_seg = None
    for seg in segments:
        if seg["start_frame"] <= step_idx <= seg["end_frame"]:
            current_seg = seg
            break
    if current_seg is None or current_seg["phase"] != "exec":
        # Fallback: uniform sampling of past
        return _even_sampling(step_idx, max_frames)

    exec_label = current_seg["label"]

    # Skip "all tasks completed" or "complete"
    if _is_completed(exec_label):
        return _even_sampling(step_idx, max_frames)

    strategy, priority_fn = get_density_rule(task_name, exec_label)

    # Resolve source segments as dicts
    seg_lookup = {s["idx"]: s for s in segments}
    source_segs = [seg_lookup[i] for i in source_seg_indices if i in seg_lookup]

    if strategy == "equal_weight_keyframes":
        return _strategy_equal_weight_keyframes(
            step_idx, source_segs, keyframes_by_seg, max_frames
        )
    elif strategy == "uniform_linspace":
        return _even_sampling(step_idx, max_frames)
    elif strategy == "priority_keyframes_then_equal":
        return _strategy_priority_then_equal(
            step_idx, source_segs, segments, keyframes_by_seg,
            priority_fn, max_frames
        )
    elif strategy == "priority_keyframes_then_uniform":
        return _strategy_priority_then_uniform(
            step_idx, source_segs, segments, keyframes_by_seg,
            priority_fn, max_frames
        )
    elif strategy == "exec_keyframes_then_uniform_demo":
        return _strategy_exec_keyframes_uniform_demo(
            step_idx, source_segs, keyframes_by_seg, max_frames
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _even_sampling(step_idx: int, max_frames: int) -> list[int]:
    """Uniform linspace over [0, step_idx].  Matches the baseline."""
    if step_idx < max_frames:
        return list(range(step_idx + 1))
    return np.linspace(0, step_idx, max_frames, dtype=np.int64).tolist()



def _strategy_equal_weight_keyframes(
    step_idx: int,
    source_segs: list[dict],
    keyframes_by_seg: dict[int, list[int]],
    max_frames: int,
) -> list[int]:
    """Strategy A: equal-weight keyframes across all chosen source segments."""
    if not source_segs:
        return _even_sampling(step_idx, max_frames)

    # Collect keyframes per source segment (only those before step_idx)
    per_seg_kf: list[list[int]] = []
    for seg in source_segs:
        kfs = [k for k in _collect_segment_keyframes(seg, keyframes_by_seg)
               if k <= step_idx]
        per_seg_kf.append(kfs)

    # Equal budget per segment
    n_segs = len(source_segs)
    per_seg_budget = max_frames // n_segs
    remainder = max_frames % n_segs

    selected: list[int] = []
    for i, kfs in enumerate(per_seg_kf):
        budget = per_seg_budget + (1 if i < remainder else 0)
        if not kfs:
            # No keyframes for this segment — fill with uniform from segment range
            seg = source_segs[i]
            end = min(seg["end_frame"], step_idx)
            selected.extend(_uniform_indices(seg["start_frame"], end, budget))
        elif len(kfs) <= budget:
            selected.extend(kfs)
            # Fill remaining budget with uniform from this segment
            remaining = budget - len(kfs)
            if remaining > 0:
                seg = source_segs[i]
                end = min(seg["end_frame"], step_idx)
                kf_set = set(kfs)
                pool = [f for f in _uniform_indices(seg["start_frame"], end, budget)
                        if f not in kf_set]
                selected.extend(pool[:remaining])
        else:
            # More keyframes than budget — subsample
            subsample_idx = np.linspace(0, len(kfs) - 1, budget, dtype=np.int64)
            selected.extend(kfs[int(j)] for j in subsample_idx)

    return sorted(set(selected))[:max_frames]


def _strategy_priority_then_equal(
    step_idx: int,
    source_segs: list[dict],
    all_segments: list[dict],
    keyframes_by_seg: dict[int, list[int]],
    priority_fn,
    max_frames: int,
) -> list[int]:
    """Strategy C: all keyframes from priority segment(s), fill remaining
    with equal-weight keyframes from other chosen segments."""
    if not source_segs:
        return _even_sampling(step_idx, max_frames)

    # Split into priority and other
    priority_segs = [s for s in source_segs if priority_fn(s, all_segments)]
    other_segs = [s for s in source_segs if not priority_fn(s, all_segments)]

    # Collect all priority keyframes
    priority_kfs: list[int] = []
    for seg in priority_segs:
        priority_kfs.extend(
            k for k in _collect_segment_keyframes(seg, keyframes_by_seg)
            if k <= step_idx
        )
    priority_kfs = sorted(set(priority_kfs))

    # If no keyframes found for priority segments, use uniform from their ranges
    if not priority_kfs and priority_segs:
        for seg in priority_segs:
            end = min(seg["end_frame"], step_idx)
            priority_kfs.extend(_uniform_indices(seg["start_frame"], end, max_frames // max(len(priority_segs), 1)))
        priority_kfs = sorted(set(priority_kfs))

    # If priority keyframes exceed budget, subsample them
    if len(priority_kfs) >= max_frames:
        subsample_idx = np.linspace(0, len(priority_kfs) - 1, max_frames, dtype=np.int64)
        return sorted(priority_kfs[int(j)] for j in subsample_idx)

    selected = list(priority_kfs)
    remaining_budget = max_frames - len(selected)

    # Distribute remaining across other segments with equal weight
    if other_segs and remaining_budget > 0:
        per_seg_budget = remaining_budget // len(other_segs)
        remainder = remaining_budget % len(other_segs)
        existing = set(selected)

        for i, seg in enumerate(other_segs):
            budget = per_seg_budget + (1 if i < remainder else 0)
            kfs = [k for k in _collect_segment_keyframes(seg, keyframes_by_seg)
                   if k <= step_idx and k not in existing]
            if not kfs:
                end = min(seg["end_frame"], step_idx)
                new = [f for f in _uniform_indices(seg["start_frame"], end, budget)
                       if f not in existing]
                selected.extend(new[:budget])
            elif len(kfs) <= budget:
                selected.extend(kfs)
                remaining = budget - len(kfs)
                if remaining > 0:
                    end = min(seg["end_frame"], step_idx)
                    kf_set = existing | set(kfs)
                    pool = [f for f in _uniform_indices(seg["start_frame"], end, budget)
                            if f not in kf_set]
                    selected.extend(pool[:remaining])
            else:
                subsample_idx = np.linspace(0, len(kfs) - 1, budget, dtype=np.int64)
                selected.extend(kfs[int(j)] for j in subsample_idx)

    return sorted(set(selected))[:max_frames]


def _strategy_priority_then_uniform(
    step_idx: int,
    source_segs: list[dict],
    all_segments: list[dict],
    keyframes_by_seg: dict[int, list[int]],
    priority_fn,
    max_frames: int,
) -> list[int]:
    """Strategy D: all keyframes from priority segment(s), fill remaining
    with uniformly spaced frames (linspace) from the full past."""
    if not source_segs:
        return _even_sampling(step_idx, max_frames)

    # Collect priority keyframes
    priority_segs = [s for s in source_segs if priority_fn(s, all_segments)]
    priority_kfs: list[int] = []
    for seg in priority_segs:
        priority_kfs.extend(
            k for k in _collect_segment_keyframes(seg, keyframes_by_seg)
            if k <= step_idx
        )
    priority_kfs = sorted(set(priority_kfs))

    # If no keyframes found for priority segments, use uniform from their ranges
    if not priority_kfs and priority_segs:
        for seg in priority_segs:
            end = min(seg["end_frame"], step_idx)
            priority_kfs.extend(_uniform_indices(seg["start_frame"], end, max_frames // max(len(priority_segs), 1)))
        priority_kfs = sorted(set(priority_kfs))

    if len(priority_kfs) >= max_frames:
        subsample_idx = np.linspace(0, len(priority_kfs) - 1, max_frames, dtype=np.int64)
        return sorted(priority_kfs[int(j)] for j in subsample_idx)

    selected = set(priority_kfs)
    remaining_budget = max_frames - len(selected)

    # Fill remaining with uniform linspace from [0, step_idx]
    if remaining_budget > 0:
        uniform = _uniform_indices(0, step_idx, max_frames)
        filler = [f for f in uniform if f not in selected]
        if len(filler) > remaining_budget:
            subsample_idx = np.linspace(0, len(filler) - 1, remaining_budget, dtype=np.int64)
            filler = [filler[int(j)] for j in subsample_idx]
        selected.update(filler)

    return sorted(selected)[:max_frames]


def _strategy_exec_keyframes_uniform_demo(
    step_idx: int,
    source_segs: list[dict],
    keyframes_by_seg: dict[int, list[int]],
    max_frames: int,
) -> list[int]:
    """Strategy E: equal-weight keyframes from exec sources, fill remaining
    with uniform linspace across demo sources."""
    if not source_segs:
        return _even_sampling(step_idx, max_frames)

    exec_sources = [s for s in source_segs if s["phase"] == "exec"]
    demo_sources = [s for s in source_segs if s["phase"] == "demo"]

    # Collect exec keyframes
    exec_kfs: list[int] = []
    for seg in exec_sources:
        exec_kfs.extend(
            k for k in _collect_segment_keyframes(seg, keyframes_by_seg)
            if k <= step_idx
        )
    exec_kfs = sorted(set(exec_kfs))

    if len(exec_kfs) >= max_frames:
        subsample_idx = np.linspace(0, len(exec_kfs) - 1, max_frames, dtype=np.int64)
        return sorted(exec_kfs[int(j)] for j in subsample_idx)

    selected = set(exec_kfs)
    remaining_budget = max_frames - len(selected)

    # Fill from demo segments uniformly
    if demo_sources and remaining_budget > 0:
        demo_start = min(s["start_frame"] for s in demo_sources)
        demo_end = min(max(s["end_frame"] for s in demo_sources), step_idx)
        uniform = _uniform_indices(demo_start, demo_end, remaining_budget)
        filler = [f for f in uniform if f not in selected]
        selected.update(filler[:remaining_budget])

    return sorted(selected)[:max_frames]


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def compute_episode_frame_indices(
    task_name: str,
    segments: list[dict],
    keyframes_by_seg: dict[int, list[int]],
    task_goal: str = "",
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> dict[int, list[int]]:
    """Compute frame indices for every exec step in the episode.

    Returns a dict mapping each exec segment index to the list of
    frame indices to sample when executing that subtask.

    This combines sampling_rules (which segments to look at) with
    density rules (how to sample frames from those segments).
    """
    source_map = get_sampling_sources(task_name, segments, task_goal=task_goal)
    result: dict[int, list[int]] = {}

    for seg in segments:
        if seg["phase"] != "exec" or _is_completed(seg["label"]):
            continue

        seg_idx = seg["idx"]
        source_seg_indices = source_map.get(seg_idx, [])

        # Use the midpoint of the segment as the "current step" for sampling
        # (in practice, during training, step_idx varies within the segment)
        mid_step = (seg["start_frame"] + seg["end_frame"]) // 2

        indices = compute_frame_indices(
            task_name=task_name,
            step_idx=mid_step,
            segments=segments,
            source_seg_indices=source_seg_indices,
            keyframes_by_seg=keyframes_by_seg,
            task_goal=task_goal,
            max_frames=max_frames,
        )
        result[seg_idx] = indices

    return result
