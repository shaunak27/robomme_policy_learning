"""Precompute segments.json and density_keyframes.json per episode.

Saves the metadata needed by density-based frame sampling into each
episode's feature directory (both main features/ and per_task/ copies).

For each of the 1600 episodes:
  - segments.json: ordered list of {idx, phase, label, start_frame, end_frame, num_frames}
  - density_keyframes.json: {seg_idx: [relative_keyframe_indices]} from topreward_full

Usage:
    python scripts/precompute_density_metadata.py
    python scripts/precompute_density_metadata.py --tasks BinFill PatternLock
    python scripts/precompute_density_metadata.py --episodes-per-task 5
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
    get_timestep_indices,
)

H5_DIR = Path("/coc/testnvme/shalbe3/robomme_data/robomme_data_h5")
TOPREWARD_DIR = Path("/coc/testnvme/shalbe3/robomme_data/topreward_full")
PREPROCESSED_DIR = Path("/coc/testnvme/shalbe3/robomme_data/robomme_preprocessed_data")

# Global episode offset per task (alphabetical order, 100 episodes each)
TASK_ORDER = [
    "BinFill", "ButtonUnmask", "ButtonUnmaskSwap", "InsertPeg",
    "MoveCube", "PatternLock", "PickHighlight", "PickXtimes",
    "RouteStick", "StopCube", "SwingXtimes", "VideoPlaceButton",
    "VideoPlaceOrder", "VideoRepick", "VideoUnmask", "VideoUnmaskSwap",
]
TASK_GLOBAL_OFFSET = {task: i * 100 for i, task in enumerate(TASK_ORDER)}


def extract_segments(ep_data: h5py.Group, ts_indices: list[int], exec_start: int) -> list[dict]:
    """Extract ordered segments from episode data."""
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
            # Fallback: match by start_frame only
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


def save_to_feature_dir(feature_dir: Path, segments: list[dict], keyframes: dict[int, list[int]]):
    """Save segments.json and density_keyframes.json to a feature directory."""
    if not feature_dir.exists():
        print(f"  WARNING: feature dir missing: {feature_dir}")
        return False

    with open(feature_dir / "segments.json", "w") as f:
        json.dump(segments, f)

    # Convert int keys to str for JSON
    kf_str_keys = {str(k): v for k, v in keyframes.items()}
    with open(feature_dir / "density_keyframes.json", "w") as f:
        json.dump(kf_str_keys, f)

    return True


def main():
    parser = argparse.ArgumentParser(description="Precompute density sampling metadata")
    parser.add_argument("--tasks", nargs="*", help="Subset of tasks (default: all)")
    parser.add_argument("--episodes-per-task", type=int, default=100)
    args = parser.parse_args()

    tasks = args.tasks or TASK_ORDER
    total_saved = 0
    total_skipped = 0

    for task_name in tasks:
        if task_name not in TASK_GLOBAL_OFFSET:
            print(f"[SKIP] Unknown task: {task_name}")
            continue

        h5_path = H5_DIR / f"record_dataset_{task_name}.h5"
        if not h5_path.exists():
            print(f"[SKIP] H5 not found: {h5_path}")
            continue

        global_offset = TASK_GLOBAL_OFFSET[task_name]
        print(f"\n{task_name} (global offset={global_offset})")

        with h5py.File(str(h5_path), "r") as f:
            ep_indices = get_episode_indices(f)
            n_sample = min(args.episodes_per_task, len(ep_indices))

            for ep_idx in ep_indices[:n_sample]:
                ep = f[f"episode_{ep_idx}"]
                ts_indices = get_timestep_indices(ep)
                exec_start = first_execution_step(ep)
                segments = extract_segments(ep, ts_indices, exec_start)
                keyframes = load_keyframes_for_episode(task_name, ep_idx, segments)

                # Save to per_task features
                per_task_dir = PREPROCESSED_DIR / "per_task" / task_name / "features" / f"episode_{ep_idx}"
                ok1 = save_to_feature_dir(per_task_dir, segments, keyframes)

                # Save to main features
                global_ep = global_offset + ep_idx
                main_dir = PREPROCESSED_DIR / "features" / f"episode_{global_ep}"
                ok2 = save_to_feature_dir(main_dir, segments, keyframes)

                if ok1 or ok2:
                    total_saved += 1
                else:
                    total_skipped += 1

        print(f"  {n_sample} episodes processed")

    print(f"\nDone: {total_saved} episodes saved, {total_skipped} skipped")


if __name__ == "__main__":
    main()
