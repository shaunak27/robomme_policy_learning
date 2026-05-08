"""Fast dataset for fused RL selector training (v2).

Key differences from dataset.py:
- Loads consolidated per-episode arrays (detail_img_4x4.npy etc.) instead of
  hundreds of individual token_emb_{t}.npy files.  3 file reads per episode.
- Pre-loads ALL candidate detail embeddings into the batch so the fused train
  step can gather selected frames on-device via vmap indexing (no CPU I/O).
- Computes uniform baseline frame packing from consolidated arrays.
- Applies VLA transforms in __getitem__ so the DataLoader yields
  ready-to-shard batches with no per-step CPU work in the training loop.

Memory budget per sample (N=512 candidates, 16 tokens, 2048 dim):
    cand_detail_img:  512 * 16 * 2048 * 2 bytes (bf16) ≈ 33 MB
    cand_detail_pos:  512 * 16 *  768 * 4 bytes (f32)  ≈ 25 MB
    cand_detail_state: 512 * 8  *    4 bytes (f32)     ≈  16 KB
    Total per sample ≈ 58 MB → batch of 64 ≈ 3.6 GB, sharded across GPUs.
"""

import json
import logging
import math
import os
import pickle
from pathlib import Path

import numpy as np
from openpi.training.data_loader import Dataset

from mme_vla_suite.selector.config import SelectorConfig
from mme_vla_suite.shared.data_utils import even_sampling_indices, right_padding_token_emb

logger = logging.getLogger(__name__)


class SelectorDatasetV2(Dataset):
    """Yields fully-prepared samples for the fused RL selector train step."""

    def __init__(
        self,
        dataset_path: str,
        config: SelectorConfig,
        vla_transform=None,
        norm_stats: dict | None = None,
        use_quantiles: bool = False,
    ):
        self.config = config
        self.dataset_path = dataset_path
        self.feature_dir = Path(dataset_path) / "features"

        stats_path = os.path.join(dataset_path, "meta", "stats.json")
        self.stats = json.load(open(stats_path))

        self.vla_transform = vla_transform
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles

        # Episode-level caches (populated lazily)
        self._episode_lengths: dict[int, int] = {}
        self._consolidated_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self):
        if "execution_samples" in self.stats:
            return self.stats["execution_samples"]
        return self.stats["total_samples"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_episode_length(self, epis_idx: int) -> int:
        if epis_idx not in self._episode_lengths:
            global_path = self.feature_dir / f"episode_{epis_idx}" / "global_emb.npy"
            if global_path.exists():
                self._episode_lengths[epis_idx] = np.load(global_path).shape[0]
            else:
                # Fallback: use consolidated detail array
                detail_path = self.feature_dir / f"episode_{epis_idx}" / "detail_img_4x4.npy"
                self._episode_lengths[epis_idx] = np.load(detail_path).shape[0]
        return self._episode_lengths[epis_idx]

    def _load_consolidated(self, epis_idx: int):
        """Load consolidated detail arrays for an episode.

        Returns (detail_img, detail_pos, detail_state) with shapes:
            detail_img:   (T, 16, 2048) bf16
            detail_pos:   (T, 16, 768)  f32
            detail_state: (T, 8)        f32
        """
        if epis_idx in self._consolidated_cache:
            return self._consolidated_cache[epis_idx]

        ep_dir = self.feature_dir / f"episode_{epis_idx}"
        img = np.load(ep_dir / "detail_img_4x4.npy")
        pos = np.load(ep_dir / "detail_pos_4x4.npy")
        state = np.load(ep_dir / "detail_state.npy")

        # Don't cache too aggressively — let OS page cache handle it
        return img, pos, state

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return state
        ns = self.norm_stats
        if self.use_quantiles:
            return (state - ns.q01) / (ns.q99 - ns.q01 + 1e-6) * 2.0 - 1.0
        return (state - ns.mean) / (ns.std + 1e-6)

    # ------------------------------------------------------------------
    # Frame packing (same logic as dataset.py but from consolidated arrays)
    # ------------------------------------------------------------------

    def _pack_uniform_baseline(
        self,
        detail_img: np.ndarray,   # (T, 16, 2048)
        detail_pos: np.ndarray,   # (T, 16, 768)
        detail_state: np.ndarray, # (T, 8)
        step_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pack uniformly-sampled frames into VLA memory format."""
        token_per_image = self.config.token_per_image  # 16
        num_views = self.config.num_views              # 1
        max_frames = self.config.token_budget // (token_per_image * num_views)

        indices = even_sampling_indices(step_idx, max_frames)

        # Gather from consolidated arrays
        img = detail_img[indices]         # (L, 16, 2048)
        pos = detail_pos[indices]         # (L, 16, 768)
        state = detail_state[indices]     # (L, 8)
        mask = np.ones(len(indices), dtype=np.bool_)

        # Right-pad to max_frames
        img, pos, state, mask = right_padding_token_emb(img, pos, state, mask, max_frames)

        # Reshape to flat token sequence
        img_dim = self.config.candidate_emb_dim  # 2048
        pos_dim = 768
        img_flat = img.reshape(-1, img_dim)
        pos_flat = pos.reshape(-1, pos_dim)
        mask_flat = np.repeat(mask, num_views * token_per_image)
        state_flat = np.repeat(state, num_views * token_per_image, axis=0)

        return img_flat, pos_flat, state_flat, mask_flat

    # ------------------------------------------------------------------
    # Candidate detail tensors (padded to max_candidates)
    # ------------------------------------------------------------------

    def _build_candidate_tensors(
        self,
        detail_img: np.ndarray,   # (T, 16, 2048)
        detail_pos: np.ndarray,   # (T, 16, 768)
        detail_state: np.ndarray, # (T, 8)
        global_emb: np.ndarray,   # (T, 2048)
        step_idx: int,
    ) -> dict:
        """Build padded candidate tensors for the selector and on-device gather.

        When there are more history frames than max_candidates, we uniformly
        subsample to fit the budget.  This keeps memory bounded while preserving
        temporal diversity across the episode.

        Returns dict with cand_embs, cand_times, cand_mask, cand_detail_img,
        cand_detail_pos, cand_detail_state — all padded to max_candidates.
        """
        N = self.config.max_candidates
        K = self.config.num_frames_to_select
        num_cands = step_idx + 1  # frames 0..step_idx

        # Slice history up to current step
        raw_global = global_emb[:num_cands]         # (num_cands, 2048)
        raw_img = detail_img[:num_cands]             # (num_cands, 16, 2048)
        raw_pos = detail_pos[:num_cands]             # (num_cands, 16, 768)
        raw_state = detail_state[:num_cands]         # (num_cands, 8)

        # Tile if fewer candidates than K (prevents selector from picking invalid slots)
        if num_cands < K:
            reps = (K // num_cands) + 1
            raw_global = np.tile(raw_global, (reps, 1))[:K]
            raw_img = np.tile(raw_img, (reps, 1, 1))[:K]
            raw_pos = np.tile(raw_pos, (reps, 1, 1))[:K]
            raw_state = np.tile(raw_state, (reps, 1))[:K]
            num_cands = K

        # Subsample if more candidates than budget
        if num_cands > N:
            sub_idx = np.linspace(0, num_cands - 1, N, dtype=np.int32)
            raw_global = raw_global[sub_idx]
            raw_img = raw_img[sub_idx]
            raw_pos = raw_pos[sub_idx]
            raw_state = raw_state[sub_idx]
            # Remap original timestep indices for age calculation
            orig_times = sub_idx.astype(np.float32)
            n = N
        else:
            orig_times = np.arange(num_cands, dtype=np.float32)
            n = num_cands

        # Padded candidate global embeddings (for selector scoring)
        cand_embs = np.zeros((N, self.config.candidate_emb_dim), dtype=np.float32)
        cand_embs[:n] = raw_global[:n].astype(np.float32)

        # Candidate timestamps (normalised age based on original frame index)
        cand_times = np.zeros((N, 1), dtype=np.float32)
        for t in range(n):
            orig_t = orig_times[t % len(orig_times)]
            cand_times[t, 0] = (step_idx - orig_t) / max(step_idx, 1)

        # Candidate validity mask
        cand_mask = np.zeros(N, dtype=np.bool_)
        cand_mask[:n] = True

        # Padded candidate detail embeddings (for on-device gather after selection)
        cand_detail_img = np.zeros((N, 16, 2048), dtype=np.float32)
        cand_detail_img[:n] = raw_img[:n].astype(np.float32)

        cand_detail_pos = np.zeros((N, 16, 768), dtype=np.float32)
        cand_detail_pos[:n] = raw_pos[:n].astype(np.float32)

        cand_detail_state = np.zeros((N, 8), dtype=np.float32)
        cand_detail_state[:n] = raw_state[:n].astype(np.float32)

        return {
            "cand_embs": cand_embs,
            "cand_times": cand_times,
            "cand_mask": cand_mask,
            "cand_detail_img": cand_detail_img,
            "cand_detail_pos": cand_detail_pos,
            "cand_detail_state": cand_detail_state,
        }

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        # Load base sample (actions, images, prompts, etc.)
        with open(os.path.join(self.dataset_path, "data", f"{idx}.pkl"), "rb") as f:
            data = pickle.load(f)

        epis_idx = int(data["epis_idx"].item()) if hasattr(data["epis_idx"], "item") else int(data["epis_idx"])
        step_idx = int(data["step_idx"].item()) if hasattr(data["step_idx"], "item") else int(data["step_idx"])
        episode_length = self._get_episode_length(epis_idx)

        # Load consolidated detail arrays and global embeddings
        detail_img, detail_pos, detail_state = self._load_consolidated(epis_idx)
        global_emb = np.load(self.feature_dir / f"episode_{epis_idx}" / "global_emb.npy")

        # Build candidate tensors (padded, for selector + on-device gather)
        cand = self._build_candidate_tensors(
            detail_img, detail_pos, detail_state, global_emb, step_idx,
        )

        # Current front-view embedding (for selector query)
        front_view_emb = global_emb[min(step_idx, len(global_emb) - 1)].astype(np.float32)

        # Proprio + progress
        proprio = data["state"].astype(np.float32)
        progress = np.array([step_idx / max(episode_length - 1, 1)], dtype=np.float32)

        # Uniform baseline (pre-packed, from consolidated arrays)
        uni_img, uni_pos, uni_state, uni_mask = self._pack_uniform_baseline(
            detail_img, detail_pos, detail_state, step_idx,
        )

        # Build the VLA-only dict for the transform pipeline
        action_horizon = 20
        vla_input = {
            "image": data["image"],
            "wrist_image": data["wrist_image"],
            "state": data["state"],
            "actions": data["actions"][:action_horizon],
            "prompt": data["prompt"],
            "simple_subgoal": data.get("simple_subgoal", ""),
            "grounded_subgoal": data.get("grounded_subgoal", ""),
            # Placeholders required by repack transform
            "static_image_emb": None,
            "static_pos_emb": None,
            "static_state_emb": None,
            "static_mask": None,
            "recur_image_emb": None,
            "recur_pos_emb": None,
            "recur_state_emb": None,
            "recur_mask": None,
        }

        # Apply VLA transforms (tokenization, image preprocessing, normalization)
        if self.vla_transform is not None:
            vla_input = self.vla_transform(vla_input)

        # Merge transformed VLA data with selector-specific fields.
        # Strip None values (memory placeholders) — they're injected on-device.
        sample = {k: v for k, v in vla_input.items() if v is not None}
        sample.update({
            # Selector inputs
            "cand_embs": cand["cand_embs"],
            "cand_times": cand["cand_times"],
            "cand_mask": cand["cand_mask"],
            "front_view_emb": front_view_emb,
            "proprio": proprio,
            "progress": progress,
            # Candidate details for on-device gather
            "cand_detail_img": cand["cand_detail_img"],
            "cand_detail_pos": cand["cand_detail_pos"],
            "cand_detail_state": cand["cand_detail_state"],
            # Pre-packed uniform baseline
            "uniform_static_image_emb": uni_img.astype(np.float32),
            "uniform_static_pos_emb": uni_pos.astype(np.float32),
            "uniform_static_state_emb": self._normalize_state(uni_state).astype(np.float32),
            "uniform_static_mask": uni_mask,
        })

        return sample


class MultiTaskDatasetV2:
    """Concatenates multiple SelectorDatasetV2 instances."""

    def __init__(self, sub_datasets: list[SelectorDatasetV2]):
        self._datasets = sub_datasets
        self._cumulative: list[int] = []
        total = 0
        for ds in sub_datasets:
            total += len(ds)
            self._cumulative.append(total)

    def __len__(self):
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, idx: int) -> dict:
        for i, cum in enumerate(self._cumulative):
            if idx < cum:
                local = idx - (self._cumulative[i - 1] if i > 0 else 0)
                return self._datasets[i][local]
        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
