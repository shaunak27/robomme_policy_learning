"""Precompute SigLIP text embeddings for per-episode instruction prompts.

Each episode has a unique prompt (e.g. different cube colors/counts).
Collects all unique prompts across episodes, batch-encodes them once,
then saves per-episode embeddings at:
    data/robomme_preprocessed_data/per_task/<task>/features/episode_<ep>/instruction_emb.npy

Usage:
    python scripts/precompute_instruction_emb.py
    python scripts/precompute_instruction_emb.py --dataset_path data/robomme_preprocessed_data
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, SiglipModel


SIGLIP_MODEL = "google/siglip-so400m-patch14-384"  # 64-token text context
BATCH_SIZE = 64


def collect_episode_prompts(task_data_dir: Path, feature_dir: Path) -> dict[int, str]:
    """Map episode index -> prompt string by sampling one pkl per episode."""
    ep_prompts = {}
    # Iterate episode dirs to find which episodes exist
    for ep_dir in sorted(feature_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue
        ep_idx = int(ep_dir.name.split("_")[1])
        # Find any pkl sample from this episode
        # Scan data files until we find one with this epis_idx
        if ep_idx not in ep_prompts:
            # Use a heuristic: episode data is roughly contiguous
            # Just scan a few files to find a sample from this episode
            ep_prompts[ep_idx] = None

    # Scan all pkl files and collect prompts per episode
    files = sorted(os.listdir(task_data_dir), key=lambda f: int(f.replace(".pkl", "")))
    needed = set(ep_prompts.keys())
    for f in files:
        if not needed:
            break
        with open(task_data_dir / f, "rb") as fh:
            data = pickle.load(fh)
        ep = int(data["epis_idx"][0]) if hasattr(data["epis_idx"], "__len__") else int(data["epis_idx"])
        if ep in needed:
            ep_prompts[ep] = data["prompt"]
            needed.discard(ep)

    return {k: v for k, v in ep_prompts.items() if v is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="data/robomme_preprocessed_data",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    per_task_dir = dataset_path / "per_task"

    print(f"Loading SigLIP text encoder: {SIGLIP_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL)
    model = SiglipModel.from_pretrained(SIGLIP_MODEL)
    model.eval()

    tasks = sorted(
        d.name for d in per_task_dir.iterdir()
        if d.is_dir() and (d / "data").exists()
    )

    total_saved = 0

    for task in tasks:
        data_dir = per_task_dir / task / "data"
        feature_dir = per_task_dir / task / "features"
        ep_prompts = collect_episode_prompts(data_dir, feature_dir)

        # Deduplicate prompts for efficient encoding
        unique_prompts = list(set(ep_prompts.values()))
        prompt_to_emb = {}

        # Batch encode unique prompts
        for i in range(0, len(unique_prompts), BATCH_SIZE):
            batch_prompts = unique_prompts[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
            )
            with torch.no_grad():
                text_out = model.text_model(**inputs)
                embs = text_out.pooler_output.cpu().numpy()  # (batch, 1152)
            for j, prompt in enumerate(batch_prompts):
                prompt_to_emb[prompt] = embs[j].astype(np.float32)

        # Save per-episode
        for ep_idx, prompt in sorted(ep_prompts.items()):
            out_path = feature_dir / f"episode_{ep_idx}" / "instruction_emb.npy"
            np.save(out_path, prompt_to_emb[prompt])
            total_saved += 1

        print(
            f"  {task}: {len(ep_prompts)} episodes, "
            f"{len(unique_prompts)} unique prompts"
        )

    print(f"\nDone. Saved {total_saved} instruction embeddings across {len(tasks)} tasks.")


if __name__ == "__main__":
    main()
