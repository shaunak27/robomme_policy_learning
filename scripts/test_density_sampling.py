"""Quick test: verify density_sampling works end-to-end for a few samples.

Loads a few pkl samples, computes density-based frame indices, and
loads features via the existing mem_buffer pipeline.

Usage:
    python scripts/test_density_sampling.py
    python scripts/test_density_sampling.py --samples 0 100000 400000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mme_vla_suite.shared.sampling_density import compute_frame_indices
from mme_vla_suite.shared.sampling_rules import get_sampling_sources
from mme_vla_suite.shared.mem_buffer import MemoryBuffer
from mme_vla_suite.shared.data_utils import even_sampling_indices

PREPROCESSED_DIR = Path("/coc/testnvme/shalbe3/robomme_data/robomme_preprocessed_data")

_TASK_ORDER = [
    "BinFill", "ButtonUnmask", "ButtonUnmaskSwap", "InsertPeg",
    "MoveCube", "PatternLock", "PickHighlight", "PickXtimes",
    "RouteStick", "StopCube", "SwingXtimes", "VideoPlaceButton",
    "VideoPlaceOrder", "VideoRepick", "VideoUnmask", "VideoUnmaskSwap",
]


def episode_to_task(global_ep_idx: int) -> str:
    return _TASK_ORDER[global_ep_idx // 100]


def load_token_emb(path, step_idx):
    with open(path, "rb") as f:
        return np.load(f, allow_pickle=True).item(), step_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", type=int, default=[0, 1000, 50000, 200000, 400000])
    args = parser.parse_args()

    feature_dir = PREPROCESSED_DIR / "features"
    token_per_image = 16
    token_budget = 512
    max_frames = token_budget // token_per_image  # 32

    mem_buffer = MemoryBuffer(
        num_views=1,
        img_emb_dim=2048,
        pos_emb_dim=768,
        state_emb_dim=8,
    )

    for sample_idx in args.samples:
        pkl_path = PREPROCESSED_DIR / "data" / f"{sample_idx}.pkl"
        if not pkl_path.exists():
            print(f"[SKIP] sample {sample_idx}: pkl not found")
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        epis_idx = data["epis_idx"].item()
        step_idx = data["step_idx"].item()
        prompt = data["prompt"]
        task_name = episode_to_task(epis_idx)

        print(f"\n{'='*60}")
        print(f"Sample {sample_idx}: ep={epis_idx}, step={step_idx}, task={task_name}")
        print(f"  prompt: {prompt[:80]}")
        print(f"  subgoal: {data['simple_subgoal'][:80]}")

        # Check precomputed metadata exists
        ep_dir = feature_dir / f"episode_{epis_idx}"
        seg_path = ep_dir / "segments.json"
        kf_path = ep_dir / "density_keyframes.json"

        if not seg_path.exists() or not kf_path.exists():
            print(f"  [SKIP] Precomputed metadata not found")
            continue

        with open(seg_path) as f:
            segments = json.load(f)
        with open(kf_path) as f:
            raw = json.load(f)
        keyframes_by_seg = {int(k): v for k, v in raw.items()}

        # Get source segments
        source_map = get_sampling_sources(task_name, segments, task_goal=prompt)

        # Find current segment
        current_seg = None
        for seg in segments:
            if seg["start_frame"] <= step_idx <= seg["end_frame"]:
                current_seg = seg
                break

        if current_seg:
            print(f"  current seg: [{current_seg['idx']}] {current_seg['phase']} \"{current_seg['label'][:50]}\"")
            source_seg_indices = source_map.get(current_seg["idx"], [])
        else:
            print(f"  current seg: NONE (step_idx={step_idx} not in any segment)")
            source_seg_indices = []

        # Compute density-based frame indices
        density_indices = compute_frame_indices(
            task_name=task_name,
            step_idx=step_idx,
            segments=segments,
            source_seg_indices=source_seg_indices,
            keyframes_by_seg=keyframes_by_seg,
            task_goal=prompt,
            max_frames=max_frames,
        )

        # Compare with baseline uniform
        baseline_indices = even_sampling_indices(step_idx, max_frames)

        print(f"  density:  {len(density_indices)} frames, indices={density_indices[:10]}{'...' if len(density_indices) > 10 else ''}")
        print(f"  baseline: {len(baseline_indices)} frames, indices={baseline_indices[:10]}{'...' if len(baseline_indices) > 10 else ''}")

        # Try loading features for density indices
        def gather_fn(indices_to_load, epis_idx):
            feats = {}
            for idx in indices_to_load:
                path = os.path.join(feature_dir, f"episode_{epis_idx}", f"token_emb_{idx}.npy")
                if os.path.exists(path):
                    feats[idx] = load_token_emb(path, idx)[0]
                else:
                    print(f"  WARNING: token_emb_{idx}.npy not found")
                    feats[idx] = {
                        "image_emb_4x4": np.zeros((1, 16, 2048), dtype=np.float32),
                        "pos_emb_4x4": np.zeros((1, 16, 768), dtype=np.float32),
                        "state_emb": np.zeros((8,), dtype=np.float32),
                    }
            return feats

        try:
            img_emb, pos_emb, state_emb, mask = mem_buffer.prepare_frame_sampling_with_indices(
                density_indices, token_budget, token_per_image,
                gather_fn, epis_idx=epis_idx,
            )
            print(f"  output: img_emb={img_emb.shape}, mask_sum={mask.sum()}/{mask.shape[0]}")
            print(f"  OK")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
