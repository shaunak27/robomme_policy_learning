"""Programmatic sampling rule functions per task.

Each function takes a list of segment dicts and returns a mapping
from exec segment index to list of past segment indices to sample from.

Segment dict keys: "idx", "phase" ("demo"/"exec"), "label" (lowercase subtask name).
Some functions accept an optional task_goal kwarg for goal-dependent logic.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
}


def _ordinal_of(label: str) -> int | None:
    """Extract ordinal number from label, e.g. 'pick up the second red cube' -> 2."""
    for word, n in ORDINALS.items():
        if word in label:
            return n
    return None


def _is_completed(label: str) -> bool:
    return "all tasks completed" in label or label == "complete"


def _exec_segs(segments: list[dict]) -> list[dict]:
    return [s for s in segments if s["phase"] == "exec" and not _is_completed(s["label"])]


def _demo_segs(segments: list[dict]) -> list[dict]:
    return [s for s in segments if s["phase"] == "demo"]


def _all_past(seg_idx: int, segments: list[dict]) -> list[int]:
    """All segment indices before seg_idx (both demo and exec, excluding completed)."""
    return [s["idx"] for s in segments if s["idx"] < seg_idx and not _is_completed(s["label"])]


def _past_exec(seg_idx: int, segments: list[dict]) -> list[int]:
    return [s["idx"] for s in segments if s["idx"] < seg_idx and s["phase"] == "exec" and not _is_completed(s["label"])]


def _past_demo(seg_idx: int, segments: list[dict]) -> list[int]:
    return [s["idx"] for s in segments if s["idx"] < seg_idx and s["phase"] == "demo"]


def _matching_demo(label: str, segments: list[dict]) -> list[int]:
    """Find demo segments whose label best matches the given exec label."""
    # Exact match first
    exact = [s["idx"] for s in segments if s["phase"] == "demo" and s["label"] == label]
    if exact:
        return exact
    # Fuzzy: shared significant words
    exec_words = set(re.findall(r'\b\w+\b', label)) - {"the", "a", "an", "to", "of", "from", "by", "on", "onto", "it", "its"}
    best = []
    best_score = 0
    for s in segments:
        if s["phase"] != "demo":
            continue
        demo_words = set(re.findall(r'\b\w+\b', s["label"])) - {"the", "a", "an", "to", "of", "from", "by", "on", "onto", "it", "its"}
        overlap = len(exec_words & demo_words)
        if overlap > best_score:
            best_score = overlap
            best = [s["idx"]]
        elif overlap == best_score and overlap > 0:
            best.append(s["idx"])
    return best


# ---------------------------------------------------------------------------
# Task rule functions
# ---------------------------------------------------------------------------

def rule_binfill(segments: list[dict]) -> dict[int, list[int]]:
    """BinFill: All pick/place subtasks need ALL prior pick and place segments
    (order matters, color-agnostic). Press button needs all prior tuples.
    Put into bin needs no memory."""
    result = {}
    for seg in _exec_segs(segments):
        label = seg["label"]
        if "press the button" in label:
            result[seg["idx"]] = _past_exec(seg["idx"], segments)
        elif "pick up" in label:
            # All prior pick/place segments regardless of color or ordinal
            prior = [s["idx"] for s in segments
                     if s["phase"] == "exec" and s["idx"] < seg["idx"]
                     and ("pick up" in s["label"] or "put" in s["label"])
                     and not _is_completed(s["label"])]
            result[seg["idx"]] = prior
        elif "put" in label and "bin" in label:
            result[seg["idx"]] = []
        else:
            result[seg["idx"]] = []
    return result


def rule_buttonunmask(segments: list[dict]) -> dict[int, list[int]]:
    """ButtonUnmask: press button = no memory. pick up container = all past."""
    result = {}
    for seg in _exec_segs(segments):
        if "press" in seg["label"]:
            result[seg["idx"]] = []
        elif "put down" in seg["label"] or "put it down" in seg["label"]:
            result[seg["idx"]] = []
        elif "pick up" in seg["label"]:
            result[seg["idx"]] = _all_past(seg["idx"], segments)
        else:
            result[seg["idx"]] = []
    return result


def rule_buttonunmaskswap(segments: list[dict]) -> dict[int, list[int]]:
    """ButtonUnmaskSwap: first button = no memory, second button = first button,
    put down = no memory, pick up = all past."""
    result = {}
    for seg in _exec_segs(segments):
        label = seg["label"]
        if "first button" in label:
            result[seg["idx"]] = []
        elif "second button" in label:
            # Needs first button
            first_btn = [s["idx"] for s in segments if s["phase"] == "exec" and "first button" in s["label"] and s["idx"] < seg["idx"]]
            result[seg["idx"]] = first_btn
        elif "put down" in label or "put it down" in label:
            result[seg["idx"]] = []
        elif "pick up" in label:
            result[seg["idx"]] = _all_past(seg["idx"], segments)
        else:
            result[seg["idx"]] = []
    return result


def rule_insertpeg(segments: list[dict]) -> dict[int, list[int]]:
    """InsertPeg: each exec subtask needs its corresponding video demo segment."""
    result = {}
    for seg in _exec_segs(segments):
        result[seg["idx"]] = _matching_demo(seg["label"], segments)
    return result


def rule_movecube(segments: list[dict]) -> dict[int, list[int]]:
    """MoveCube: all exec subtasks need corresponding + succeeding video segment.
    No need of previous exec segments."""
    result = {}
    demo = _demo_segs(segments)
    for seg in _exec_segs(segments):
        matching = _matching_demo(seg["label"], segments)
        # Also add the succeeding demo segment
        sources = set(matching)
        for m in matching:
            next_demo = [d["idx"] for d in demo if d["idx"] == m + 1]
            sources.update(next_demo)
        result[seg["idx"]] = sorted(sources)
    return result


def rule_patternlock(segments: list[dict]) -> dict[int, list[int]]:
    """PatternLock: each exec subtask needs corresponding demo subtask AND
    the next immediate demo subtask + all previous exec subtasks."""
    result = {}
    demo = _demo_segs(segments)
    for seg in _exec_segs(segments):
        sources = set()
        # Find corresponding demo segment(s)
        matching = _matching_demo(seg["label"], segments)
        sources.update(matching)
        # Add the next immediate demo segment after each match
        for m_idx in matching:
            next_demos = [d["idx"] for d in demo if d["idx"] == m_idx + 1]
            sources.update(next_demos)
        # All previous exec
        sources.update(_past_exec(seg["idx"], segments))
        result[seg["idx"]] = sorted(sources)
    return result


def rule_pickhighlight(segments: list[dict]) -> dict[int, list[int]]:
    """PickHighlight: all pick subtasks need 'press the button' subtask.
    Place cube = no memory. Nth highlighted pick needs all prior picks."""
    result = {}
    # Find button press segments
    button_segs = [s["idx"] for s in segments if s["phase"] == "exec" and "press" in s["label"]]

    for seg in _exec_segs(segments):
        label = seg["label"]
        if "press" in label:
            result[seg["idx"]] = []
        elif "place" in label and "table" in label:
            result[seg["idx"]] = []
        elif "pick up" in label or "highlighted" in label:
            sources = set(button_segs)
            # Nth pick needs all prior pick segments
            n = _ordinal_of(label)
            if n is not None and n > 1:
                prior_picks = [s["idx"] for s in segments
                               if s["phase"] == "exec" and s["idx"] < seg["idx"]
                               and ("pick up" in s["label"] or "highlighted" in s["label"])
                               and not _is_completed(s["label"])]
                sources.update(prior_picks)
            # Filter to only past
            result[seg["idx"]] = sorted(s for s in sources if s < seg["idx"])
        else:
            result[seg["idx"]] = []
    return result


def rule_pickxtimes(segments: list[dict]) -> dict[int, list[int]]:
    """PickXtimes: pick Nth time needs all prior (pick, place) tuples.
    Place tasks need no memory. Press button needs all prior tuples."""
    result = {}
    for seg in _exec_segs(segments):
        label = seg["label"]
        if "place" in label:
            result[seg["idx"]] = []
        elif "pick" in label:
            n = _ordinal_of(label)
            if n is not None and n > 1:
                # All prior pick/place tuples
                result[seg["idx"]] = _past_exec(seg["idx"], segments)
            else:
                result[seg["idx"]] = []
        elif "press" in label or "button" in label:
            # Press button needs all prior pick/place tuples
            result[seg["idx"]] = _past_exec(seg["idx"], segments)
        else:
            result[seg["idx"]] = []
    return result


def rule_routestick(segments: list[dict]) -> dict[int, list[int]]:
    """RouteStick: all subtasks need the whole video demo.
    Sample corresponding segment more densely. Add past execution."""
    result = {}
    all_demo = [s["idx"] for s in _demo_segs(segments)]
    for seg in _exec_segs(segments):
        sources = set(all_demo)
        sources.update(_past_exec(seg["idx"], segments))
        result[seg["idx"]] = sorted(sources)
    return result


def rule_stopcube(segments: list[dict]) -> dict[int, list[int]]:
    """StopCube: sample uniformly (keyframes) at every step = all past."""
    result = {}
    for seg in _exec_segs(segments):
        result[seg["idx"]] = _all_past(seg["idx"], segments)
    return result


def rule_swingxtimes(segments: list[dict]) -> dict[int, list[int]]:
    """SwingXtimes: all subtasks need all of the past (uniform)."""
    result = {}
    for seg in _exec_segs(segments):
        result[seg["idx"]] = _all_past(seg["idx"], segments)
    return result


def rule_videoplacebutton(segments: list[dict], task_goal: str = "") -> dict[int, list[int]]:
    """VideoPlaceButton: extract ONE demo segment for all exec subtasks.
    That segment is the 'drop the cube onto target' immediately before or
    after 'press the button' in demo, depending on keyword in task goal."""
    result = {}
    demo = _demo_segs(segments)

    # Find the button press in demo
    button_idx = None
    for d in demo:
        if "press" in d["label"] and "button" in d["label"]:
            button_idx = d["idx"]
            break

    # Determine before/after from task goal
    use_before = "before" in task_goal.lower()

    target_seg_idx = None
    if button_idx is not None:
        if use_before:
            # Find "drop the cube onto target" immediately before button press
            before = [d for d in demo if d["idx"] < button_idx and "drop" in d["label"] and "target" in d["label"]]
            # Take the one closest to (immediately before) button
            if before:
                target_seg_idx = max(b["idx"] for b in before)
        else:
            # Find "drop the cube onto target" immediately after button press
            after = [d for d in demo if d["idx"] > button_idx and "drop" in d["label"] and "target" in d["label"]]
            # Take the one closest to (immediately after) button
            if after:
                target_seg_idx = min(a["idx"] for a in after)

    for seg in _exec_segs(segments):
        if target_seg_idx is not None:
            result[seg["idx"]] = [target_seg_idx]
        else:
            result[seg["idx"]] = [d["idx"] for d in demo if "drop" in d["label"] and "target" in d["label"]][:1]
    return result


def rule_videoplaceorder(segments: list[dict]) -> dict[int, list[int]]:
    """VideoPlaceOrder: all exec subtasks should look at all video demo subtasks
    with labels 'drop the cube onto target', 'press the button', 'static',
    and 'drop the cube onto table'."""
    result = {}
    relevant_demo = [
        s["idx"] for s in _demo_segs(segments)
        if any(kw in s["label"] for kw in ["drop the cube onto target", "press the button", "static", "drop the cube onto table"])
    ]
    for seg in _exec_segs(segments):
        result[seg["idx"]] = relevant_demo
    return result


def rule_videorepick(segments: list[dict]) -> dict[int, list[int]]:
    """VideoRepick: all pick tasks need entire video. Nth pick needs prior
    (pick, put) tuples. Put it down = uniform memory across everything.
    Press button = all demo + exec segments till now."""
    result = {}
    all_demo = [s["idx"] for s in _demo_segs(segments)]

    for seg in _exec_segs(segments):
        label = seg["label"]
        if "press" in label or "button" in label:
            # All demo and exec segments till now
            result[seg["idx"]] = _all_past(seg["idx"], segments)
        elif "put" in label or "down" in label:
            # Uniform memory across everything
            result[seg["idx"]] = _all_past(seg["idx"], segments)
        elif "pick" in label:
            sources = set(all_demo)
            n = _ordinal_of(label)
            if n is not None and n > 1:
                sources.update(_past_exec(seg["idx"], segments))
            result[seg["idx"]] = sorted(sources)
        else:
            result[seg["idx"]] = _all_past(seg["idx"], segments)
    return result


def rule_videounmask(segments: list[dict]) -> dict[int, list[int]]:
    """VideoUnmask: pick up container = keyframes from video demo + all prior
    pick/put tuples from exec. Put down = uniform across demo AND past exec."""
    result = {}
    all_demo = [s["idx"] for s in _demo_segs(segments)]

    for seg in _exec_segs(segments):
        label = seg["label"]
        if "put down" in label or "put it down" in label:
            # Uniform across demo and past exec segments
            result[seg["idx"]] = _all_past(seg["idx"], segments)
        elif "pick up" in label:
            sources = set(all_demo)
            sources.update(_past_exec(seg["idx"], segments))
            result[seg["idx"]] = sorted(sources)
        else:
            result[seg["idx"]] = _all_past(seg["idx"], segments)
    return result


def rule_videounmaskswap(segments: list[dict]) -> dict[int, list[int]]:
    """VideoUnmaskSwap: pick up container = video demo + all prior
    (pick up, put down) exec tuples. Put down = uniform across demo + past exec."""
    result = {}
    all_demo = [s["idx"] for s in _demo_segs(segments)]

    for seg in _exec_segs(segments):
        label = seg["label"]
        if "put down" in label or "put it down" in label:
            # Uniform across demo and past exec segments
            result[seg["idx"]] = _all_past(seg["idx"], segments)
        elif "pick up" in label:
            sources = set(all_demo)
            # All prior pick/put exec tuples
            sources.update(_past_exec(seg["idx"], segments))
            result[seg["idx"]] = sorted(sources)
        else:
            result[seg["idx"]] = _all_past(seg["idx"], segments)
    return result


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

TASK_RULES = {
    "BinFill": rule_binfill,
    "ButtonUnmask": rule_buttonunmask,
    "ButtonUnmaskSwap": rule_buttonunmaskswap,
    "InsertPeg": rule_insertpeg,
    "MoveCube": rule_movecube,
    "PatternLock": rule_patternlock,
    "PickHighlight": rule_pickhighlight,
    "PickXtimes": rule_pickxtimes,
    "RouteStick": rule_routestick,
    "StopCube": rule_stopcube,
    "SwingXtimes": rule_swingxtimes,
    "VideoPlaceButton": rule_videoplacebutton,
    "VideoPlaceOrder": rule_videoplaceorder,
    "VideoRepick": rule_videorepick,
    "VideoUnmask": rule_videounmask,
    "VideoUnmaskSwap": rule_videounmaskswap,
}


def get_sampling_sources(task_name: str, segments: list[dict], task_goal: str = "") -> dict[int, list[int]]:
    """Look up and apply the rule function for the given task."""
    func = TASK_RULES.get(task_name)
    if func is None:
        raise ValueError(f"No sampling rule for task {task_name}")
    # Pass task_goal for functions that need it (e.g. VideoPlaceButton)
    import inspect
    if "task_goal" in inspect.signature(func).parameters:
        return func(segments, task_goal=task_goal)
    return func(segments)
