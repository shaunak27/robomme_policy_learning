"""Pre-compute per-episode global embeddings for RL selector training.

For each episode, reads the cached 8x8 SigLIP token embeddings (64 tokens,
the highest-resolution grid stored by build_robomme_dataset) and mean-pools
them into a single 2048-dim vector per timestep.  The result is saved as
``features/episode_{idx}/global_emb.npy`` with shape (episode_length, 2048).

Note: SigLIP So400m/14 natively produces 16x16=256 patches from 224x224 images.
The stored 8x8 grid is already spatially downsampled from those 256 patches
during dataset building.  Using all 256 tokens would require re-running SigLIP
on raw images, which is expensive; the 8x8 mean-pool is a good proxy for a
single global embedding vector.

Usage:
    python scripts/precompute_global_emb.py \
        --dataset_path /path/to/preprocessed_dataset
"""

import argparse
from pathlib import Path

import numpy as np
import tqdm


def process_episode(ep_dir: Path) -> None:
    """Create global_emb.npy for one episode directory."""
    out_path = ep_dir / "global_emb.npy"
    if out_path.exists():
        return  # already done

    # Find all token_emb files
    token_files = sorted(
        ep_dir.glob("token_emb_*.npy"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not token_files:
        return

    global_embs = []
    for tf in token_files:
        data = np.load(str(tf), allow_pickle=True).item()
        # Use the full 8x8 grid (64 tokens) — the highest resolution
        # stored by build_robomme_dataset — for the most discriminative
        # global embedding.
        img_8x8 = data["image_emb_8x8"]  # (num_views, 64, 2048)
        global_vec = img_8x8.mean(axis=(0, 1))  # (2048,)
        global_embs.append(global_vec.astype(np.float32))

    global_embs = np.stack(global_embs, axis=0)  # (T, 2048)
    np.save(str(out_path), global_embs)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute global embeddings for RL selector"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the preprocessed dataset (output of build_robomme_dataset.py)",
    )
    args = parser.parse_args()

    feature_dir = Path(args.dataset_path) / "features"
    if not feature_dir.exists():
        raise FileNotFoundError(
            f"{feature_dir} does not exist. "
            "Run build_robomme_dataset.py first to create the preprocessed dataset."
        )

    episode_dirs = sorted(
        feature_dir.glob("episode_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    print(f"Found {len(episode_dirs)} episodes in {feature_dir}")

    for ep_dir in tqdm.tqdm(episode_dirs, desc="Computing global embeddings"):
        process_episode(ep_dir)

    print("Done.")


if __name__ == "__main__":
    main()
