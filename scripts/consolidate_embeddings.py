"""Consolidate per-frame token embeddings into per-episode arrays.

For each episode directory that contains token_emb_{t}.npy files, this script
creates three consolidated files:

    detail_img_4x4.npy   — (T, 16, 2048) bfloat16
    detail_pos_4x4.npy   — (T, 16, 768)  float32
    detail_state.npy      — (T, 8)        float32

If global_emb.npy is missing (e.g., in per-task subdirectories), it is also
created by mean-pooling the 8x8 image embeddings from the per-frame files.

Processes both the top-level features/ directory and any per_task/*/features/
subdirectories.

Usage:
    python scripts/consolidate_embeddings.py \
        --dataset_path data/robomme_preprocessed_data \
        --workers 16
"""

import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (stored as uint16) to float32 by left-shifting 16 bits."""
    return np.frombuffer(
        (arr_u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32
    ).reshape(arr_u16.shape)


def consolidate_episode(ep_dir: Path) -> str | None:
    """Consolidate a single episode directory. Returns error message or None."""
    # Check if already consolidated
    if (ep_dir / "detail_img_4x4.npy").exists():
        return None

    # Discover frame count from token_emb files
    frame_files = sorted(ep_dir.glob("token_emb_*.npy"),
                         key=lambda p: int(re.search(r"token_emb_(\d+)", p.stem).group(1)))
    if not frame_files:
        return f"Skipped {ep_dir.name}: no token_emb files"

    # Get the frame indices
    frame_indices = [int(re.search(r"token_emb_(\d+)", p.stem).group(1)) for p in frame_files]
    T = max(frame_indices) + 1

    # If global_emb.npy exists, use it for T; otherwise create it
    global_path = ep_dir / "global_emb.npy"
    has_global = global_path.exists()
    if has_global:
        existing_global = np.load(global_path)
        T = existing_global.shape[0]

    img_list = []
    pos_list = []
    state_list = []
    global_list = [] if not has_global else None

    for t in range(T):
        frame_path = ep_dir / f"token_emb_{t}.npy"
        if not frame_path.exists():
            return f"ERROR {ep_dir.name}: missing token_emb_{t}.npy (expected T={T})"

        data = np.load(str(frame_path), allow_pickle=True).item()
        # image_emb_4x4: (1, 16, 2048) bf16 → squeeze view dim → (16, 2048) f32
        # Cast to float32 because numpy can't natively handle bfloat16.
        img_4x4 = data["image_emb_4x4"].squeeze(0)
        if img_4x4.dtype != np.float32:
            img_4x4 = np.frombuffer(img_4x4.tobytes(), dtype=np.uint16).reshape(img_4x4.shape)
            img_4x4 = _bf16_to_f32(img_4x4)
        img_list.append(img_4x4)
        pos_list.append(data["pos_emb_4x4"].squeeze(0))
        state_list.append(data["state_emb"])

        if global_list is not None:
            # Mean-pool from 8x8 grid for global embedding
            img_8x8 = data["image_emb_8x8"]  # (1, 64, 2048) bf16
            if img_8x8.dtype != np.float32:
                img_8x8 = np.frombuffer(img_8x8.tobytes(), dtype=np.uint16).reshape(img_8x8.shape)
                img_8x8 = _bf16_to_f32(img_8x8)
            global_vec = img_8x8.mean(axis=(0, 1))
            global_list.append(global_vec)

    img = np.stack(img_list, axis=0)      # (T, 16, 2048) bf16
    pos = np.stack(pos_list, axis=0)      # (T, 16, 768)  f32
    state = np.stack(state_list, axis=0)  # (T, 8)        f32

    np.save(ep_dir / "detail_img_4x4.npy", img)
    np.save(ep_dir / "detail_pos_4x4.npy", pos)
    np.save(ep_dir / "detail_state.npy", state)

    if global_list is not None:
        global_emb = np.stack(global_list, axis=0)  # (T, 2048)
        np.save(global_path, global_emb)

    return None


def collect_episode_dirs(dataset_path: Path) -> list[Path]:
    """Collect all episode directories from top-level and per-task subdirectories."""
    dirs = []

    # Top-level features/
    top_features = dataset_path / "features"
    if top_features.exists():
        dirs.extend(
            d for d in top_features.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        )

    # Per-task features/
    per_task = dataset_path / "per_task"
    if per_task.exists():
        for task_dir in sorted(per_task.iterdir()):
            task_features = task_dir / "features"
            if task_features.exists():
                dirs.extend(
                    d for d in task_features.iterdir()
                    if d.is_dir() and d.name.startswith("episode_")
                )

    return sorted(dirs, key=lambda d: (str(d.parent), int(d.name.split("_")[1])))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Consolidate per-frame embeddings")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    ep_dirs = collect_episode_dirs(Path(args.dataset_path))
    logger.info("Found %d episode directories (top-level + per-task)", len(ep_dirs))

    done = 0
    errors = []
    skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(consolidate_episode, d): d for d in ep_dirs}
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result is not None:
                if result.startswith("ERROR"):
                    errors.append(result)
                    logger.error(result)
                else:
                    skipped += 1
            if done % 200 == 0:
                logger.info("Progress: %d / %d episodes", done, len(ep_dirs))

    logger.info("Done. %d processed, %d skipped, %d errors.", done - skipped - len(errors), skipped, len(errors))
    for e in errors:
        logger.error(e)


if __name__ == "__main__":
    main()
