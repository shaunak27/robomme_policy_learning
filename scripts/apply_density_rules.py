"""Apply segment-aware density rules and visualize frame selections.

For each episode, computes which 32 frame indices to load for each exec
subtask, using the sampling_rules (which segments) + sampling_density
(how to sample frames from those segments).

Outputs JSON consumed by the taxonomy viewer for visual verification.

Usage:
    # Sample (2 episodes per task):
    python scripts/apply_density_rules.py --episodes-per-task 2

    # All episodes:
    python scripts/apply_density_rules.py --episodes-per-task 100

    # Specific tasks:
    python scripts/apply_density_rules.py --tasks BinFill PatternLock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    get_episode_indices,
    get_task_goal,
    get_timestep_indices,
)
from mme_vla_suite.shared.sampling_rules import TASK_RULES, get_sampling_sources
from mme_vla_suite.shared.sampling_density import (
    compute_frame_indices,
    _is_completed,
    DEFAULT_MAX_FRAMES,
)

H5_DIR = Path("/coc/testnvme/shalbe3/robomme_data/robomme_data_h5")
TOPREWARD_DIR = Path("/coc/testnvme/shalbe3/robomme_data/topreward_full")
OUTPUT_PATH = PROJECT_ROOT / "annotation_tool" / "density_results.json"


# ---------------------------------------------------------------------------
# Segment extraction (reused from apply_rules_programmatic.py)
# ---------------------------------------------------------------------------

def extract_segments(ep_data, ts_indices, exec_start):
    segments = []
    current_label = None
    seg_start = None
    for t in ts_indices:
        is_demo = t < exec_start
        raw = ep_data[f"timestep_{t}"]["info"]["simple_subgoal"][()].decode()
        label = raw.strip().lower()
        phase = "demo" if is_demo else "exec"
        key = (phase, label)
        if key != current_label:
            if current_label is not None:
                segments.append({
                    "idx": len(segments),
                    "phase": current_label[0],
                    "label": current_label[1],
                    "start_frame": seg_start,
                    "end_frame": t - 1,
                    "num_frames": t - seg_start,
                })
            current_label = key
            seg_start = t
    if current_label is not None:
        segments.append({
            "idx": len(segments),
            "phase": current_label[0],
            "label": current_label[1],
            "start_frame": seg_start,
            "end_frame": ts_indices[-1],
            "num_frames": ts_indices[-1] - seg_start + 1,
        })
    return segments


# ---------------------------------------------------------------------------
# Topreward keyframe loading
# ---------------------------------------------------------------------------

def load_keyframes_for_episode(
    task_name: str, ep_idx: int, segments: list[dict]
) -> dict[int, list[int]]:
    """Load per-segment keyframes from topreward_full.

    Returns dict mapping segment idx -> list of keyframe indices
    (relative to segment start_frame).
    """
    keyframes_by_seg: dict[int, list[int]] = {}

    for seg in segments:
        start = seg["start_frame"]
        end = seg["end_frame"]

        # Build the expected topreward directory name
        if seg["phase"] == "demo":
            dir_name = f"{task_name}_ep{ep_idx}_vdemo_f{start}-{end}"
        else:
            dir_name = f"{task_name}_ep{ep_idx}_f{start}-{end}"

        topreward_path = TOPREWARD_DIR / dir_name / "selected_keyframes.json"

        if topreward_path.exists():
            with open(topreward_path) as f:
                data = json.load(f)
            keyframes_by_seg[seg["idx"]] = data["selected_keyframes"]
        else:
            # Try matching by start_frame only (handle slight end_frame mismatches)
            if seg["phase"] == "demo":
                pattern = f"{task_name}_ep{ep_idx}_vdemo_f{start}-*"
            else:
                pattern = f"{task_name}_ep{ep_idx}_f{start}-*"
            matches = sorted(TOPREWARD_DIR.glob(pattern))
            if matches:
                kf_path = matches[0] / "selected_keyframes.json"
                if kf_path.exists():
                    with open(kf_path) as f:
                        data = json.load(f)
                    keyframes_by_seg[seg["idx"]] = data["selected_keyframes"]

    return keyframes_by_seg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--output", "-o", default=str(OUTPUT_PATH))
    parser.add_argument("--tasks", nargs="*", help="Subset of tasks")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    args = parser.parse_args()

    h5_files = sorted(H5_DIR.glob("record_dataset_*.h5"))
    results = {}
    stats = {"total": 0, "success": 0, "errors": 0, "missing_keyframes": 0}

    for h5_path in h5_files:
        task_name = h5_path.stem.replace("record_dataset_", "")

        if args.tasks and task_name not in args.tasks:
            continue

        if task_name not in TASK_RULES:
            print(f"[SKIP] {task_name}: no rule function")
            continue

        print(f"\n{'='*60}")
        print(f"{task_name}")
        print(f"{'='*60}")

        with h5py.File(str(h5_path), "r") as f:
            ep_indices = get_episode_indices(f)
            n_sample = min(args.episodes_per_task, len(ep_indices))
            sampled = ep_indices[:n_sample]

            for ep_idx in sampled:
                ep = f[f"episode_{ep_idx}"]
                task_goal = get_task_goal(ep)
                ts_indices = get_timestep_indices(ep)
                exec_start = first_execution_step(ep)
                segments = extract_segments(ep, ts_indices, exec_start)

                stats["total"] += 1

                # Load keyframes
                keyframes_by_seg = load_keyframes_for_episode(
                    task_name, ep_idx, segments
                )
                segs_with_kf = len(keyframes_by_seg)
                segs_total = len(segments)
                if segs_with_kf == 0:
                    stats["missing_keyframes"] += 1
                    print(f"  ep{ep_idx}: WARNING - no keyframes found")

                # Get source segment mapping
                try:
                    source_map = get_sampling_sources(
                        task_name, segments, task_goal=task_goal
                    )
                except Exception as e:
                    print(f"  ep{ep_idx}: ERROR in sampling_rules - {e}")
                    stats["errors"] += 1
                    continue

                # Compute frame indices per exec segment
                frame_selections = {}
                for seg in segments:
                    if seg["phase"] != "exec" or _is_completed(seg["label"]):
                        continue

                    seg_idx = seg["idx"]
                    source_seg_indices = source_map.get(seg_idx, [])

                    # Use midpoint of segment as representative step
                    mid_step = (seg["start_frame"] + seg["end_frame"]) // 2

                    try:
                        indices = compute_frame_indices(
                            task_name=task_name,
                            step_idx=mid_step,
                            segments=segments,
                            source_seg_indices=source_seg_indices,
                            keyframes_by_seg=keyframes_by_seg,
                            task_goal=task_goal,
                            max_frames=args.max_frames,
                        )
                        frame_selections[str(seg_idx)] = {
                            "frame_indices": indices,
                            "num_frames": len(indices),
                            "source_segments": source_seg_indices,
                            "step_idx": mid_step,
                        }
                    except Exception as e:
                        print(f"  ep{ep_idx} seg{seg_idx}: ERROR - {e}")
                        stats["errors"] += 1
                        frame_selections[str(seg_idx)] = {
                            "frame_indices": [],
                            "num_frames": 0,
                            "source_segments": source_seg_indices,
                            "step_idx": mid_step,
                            "error": str(e),
                        }

                stats["success"] += 1

                ep_key = f"{task_name}_ep{ep_idx}"
                results[ep_key] = {
                    "task": task_name,
                    "episode_idx": ep_idx,
                    "task_goal": task_goal,
                    "segments": segments,
                    "keyframes_by_seg": {
                        str(k): v for k, v in keyframes_by_seg.items()
                    },
                    "sampling_map": {
                        str(k): v for k, v in source_map.items()
                    },
                    "frame_selections": frame_selections,
                    "max_frames": args.max_frames,
                }

                # Print summary
                exec_segs = [
                    s for s in segments
                    if s["phase"] == "exec"
                    and not _is_completed(s["label"])
                ]
                print(
                    f"  ep{ep_idx}: {len(exec_segs)} exec segs, "
                    f"{segs_with_kf}/{segs_total} segs have keyframes, "
                    f"{len(frame_selections)} frame selections"
                )
                for seg_idx_str, sel in sorted(
                    frame_selections.items(), key=lambda x: int(x[0])
                ):
                    seg = segments[int(seg_idx_str)]
                    n = sel["num_frames"]
                    src = sel["source_segments"]
                    print(
                        f"    [{seg_idx_str}] {seg['label'][:45]:45s} "
                        f"-> {n:2d} frames from {len(src)} sources"
                    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f_out:
        json.dump(results, f_out, indent=2)

    print(f"\n{'='*60}")
    print(f"Results: {stats['success']}/{stats['total']} episodes OK")
    print(f"Errors: {stats['errors']}")
    print(f"Missing keyframes: {stats['missing_keyframes']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
