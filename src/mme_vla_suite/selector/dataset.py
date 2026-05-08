"""Dataset for RL selector training.

Each sample provides:
- All candidate frame global embeddings (mean-pooled from cached 8x8 SigLIP
  grid — the highest resolution stored by build_robomme_dataset — one per timestep)
- Current front-view global embedding
- Instruction prompt (for tokenisation by the VLA)
- Proprio state
- Episode progress
- Detailed 4x4 token embeddings for selected frames (gathered after selection)
- Actions (for FM loss computation)

Note: SigLIP So400m/14 natively produces 16x16=256 patches from 224x224 images.
The stored 8x8 (64 tokens), 4x4 (16 tokens), and 2x2 (4 tokens) grids are all
spatially downsampled from this.  Global embeddings use the 8x8 grid for best
fidelity; the VLA memory uses 4x4 for token-budget reasons.

The global embeddings are pre-computed by scripts/precompute_global_emb.py
and stored as ``features/episode_{idx}/global_emb.npy`` with shape
(episode_length, 2048).
"""

import json
import logging
import os
import math
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openpi.training.data_loader import Dataset

from mme_vla_suite.selector.config import SelectorConfig
from mme_vla_suite.shared.data_utils import even_sampling_indices, right_padding_token_emb

logger = logging.getLogger(__name__)


def _load_npy(path: str) -> dict:
    with open(path, "rb") as f:
        return np.load(f, allow_pickle=True).item()


class SelectorDataset(Dataset):
    """Yields samples for RL selector training.

    Each __getitem__ returns a dict with padded arrays ready for the selector
    and the frozen VLA.
    """

    def __init__(
        self,
        dataset_path: str,
        config: SelectorConfig,
        norm_stats: dict | None = None,
        use_quantiles: bool = False,
    ):
        self.config = config
        self.dataset_path = dataset_path

        stats_path = os.path.join(dataset_path, "meta", "stats.json")
        self.stats = json.load(open(stats_path))
        self.feature_dir = Path(dataset_path) / "features"

        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles

        # Cache episode lengths for progress computation
        self._episode_lengths: dict[int, int] = {}

    def __len__(self):
        if "execution_samples" in self.stats:
            return self.stats["execution_samples"]
        return self.stats["total_samples"]

    def _get_episode_length(self, epis_idx: int) -> int:
        if epis_idx not in self._episode_lengths:
            ep_dir = self.feature_dir / f"episode_{epis_idx}"
            global_path = ep_dir / "global_emb.npy"
            if global_path.exists():
                arr = np.load(global_path)
                self._episode_lengths[epis_idx] = arr.shape[0]
            else:
                # Fallback: count token_emb_*.npy files
                count = len(list(ep_dir.glob("token_emb_*.npy")))
                self._episode_lengths[epis_idx] = count
        return self._episode_lengths[epis_idx]

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return state
        ns = self.norm_stats
        if self.use_quantiles:
            return (state - ns.q01) / (ns.q99 - ns.q01 + 1e-6) * 2.0 - 1.0
        return (state - ns.mean) / (ns.std + 1e-6)

    def _load_global_embs(self, epis_idx: int, max_step: int) -> np.ndarray:
        """Load pre-computed global embeddings for frames 0..max_step.

        Returns (num_frames, 2048) float32.
        """
        global_path = self.feature_dir / f"episode_{epis_idx}" / "global_emb.npy"
        if global_path.exists():
            all_emb = np.load(global_path)  # (episode_len, 2048)
            return all_emb[: max_step + 1]

        # Fallback: load individual files and mean-pool from 8x8 grid
        embs = []
        for t in range(max_step + 1):
            path = self.feature_dir / f"episode_{epis_idx}" / f"token_emb_{t}.npy"
            data = _load_npy(str(path))
            img_8x8 = data["image_emb_8x8"]  # (v, 64, 2048)
            global_vec = img_8x8.mean(axis=(0, 1))  # (2048,)
            embs.append(global_vec)
        return np.stack(embs, axis=0)

    def _gather_detailed_embs(
        self, epis_idx: int, indices: list[int]
    ) -> dict[int, dict]:
        """Load full 4x4 token embeddings for specific frame indices."""
        result = {}
        paths = []
        for idx in indices:
            paths.append(
                str(self.feature_dir / f"episode_{epis_idx}" / f"token_emb_{idx}.npy")
            )
        max_workers = min(32, max(4, len(paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_load_npy, p): i for p, i in zip(paths, indices)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                result[idx] = fut.result()
        return result

    def _pack_frame_sampling(
        self,
        history_feats: dict[int, dict],
        indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pack selected frame embeddings into the standard perceptual memory format.

        Returns (static_image_emb, static_pos_emb, static_state_emb, static_mask)
        all shaped for the VLA's token budget.
        """
        token_per_image = self.config.token_per_image
        num_views = self.config.num_views
        token_budget = self.config.token_budget
        max_frames = token_budget // (token_per_image * num_views)

        spatial_size = str(int(math.sqrt(token_per_image)))
        spatial_key = f"{spatial_size}x{spatial_size}"

        img_dim = self.config.candidate_emb_dim  # 2048
        pos_dim = 768
        state_dim = self.config.proprio_dim  # 8

        sampled_img = np.stack(
            [history_feats[i][f"image_emb_{spatial_key}"] for i in indices], axis=0
        )
        sampled_pos = np.stack(
            [history_feats[i][f"pos_emb_{spatial_key}"] for i in indices], axis=0
        )
        sampled_state = np.stack(
            [history_feats[i]["state_emb"] for i in indices], axis=0
        )
        mask = np.ones(sampled_img.shape[0], dtype=np.bool_)

        sampled_img, sampled_pos, sampled_state, mask = right_padding_token_emb(
            sampled_img, sampled_pos, sampled_state, mask, max_frames
        )

        img_emb = sampled_img.reshape(-1, img_dim)
        pos_emb = sampled_pos.reshape(-1, pos_dim)
        mask = np.repeat(mask, num_views * token_per_image)
        state_emb = np.repeat(sampled_state, num_views * token_per_image, axis=0)

        return img_emb, pos_emb, state_emb, mask

    def __getitem__(self, idx: int) -> dict:
        # Load base sample
        with open(os.path.join(self.dataset_path, "data", f"{idx}.pkl"), "rb") as f:
            data = pickle.load(f)

        epis_idx = int(data["epis_idx"].item()) if hasattr(data["epis_idx"], "item") else int(data["epis_idx"])
        step_idx = int(data["step_idx"].item()) if hasattr(data["step_idx"], "item") else int(data["step_idx"])
        episode_length = self._get_episode_length(epis_idx)
        action_horizon = 20
        actions = data["actions"][:action_horizon]

        # ---- Candidate global embeddings (for selector scoring) ----
        N = self.config.max_candidates
        global_embs_raw = self._load_global_embs(epis_idx, step_idx)
        num_cands = global_embs_raw.shape[0]

        cand_embs = np.zeros((N, self.config.candidate_emb_dim), dtype=np.float32)
        cand_times = np.zeros((N, 1), dtype=np.float32)
        cand_mask = np.zeros(N, dtype=np.bool_)

        K = self.config.num_frames_to_select
        # If fewer candidates than K, tile to ensure at least K valid entries
        if num_cands < K:
            reps = (K // num_cands) + 1
            global_embs_raw = np.tile(global_embs_raw, (reps, 1))[:max(num_cands * reps, K)]
            num_cands = global_embs_raw.shape[0]

        n = min(num_cands, N)
        cand_embs[:n] = global_embs_raw[:n]
        cand_mask[:n] = True

        # Normalised relative time: (step_idx - t) / step_idx → age in [0, 1]
        for t in range(n):
            cand_times[t, 0] = (step_idx - (t % (step_idx + 1))) / max(step_idx, 1)

        # ---- Current front-view global embedding (for query) ----
        front_view_emb = global_embs_raw[step_idx] if step_idx < num_cands else global_embs_raw[-1]

        # ---- Proprio + progress ----
        proprio = data["state"].astype(np.float32)
        progress = np.array([step_idx / max(episode_length - 1, 1)], dtype=np.float32)

        # ---- Uniform frame sampling (for baseline FM loss) ----
        max_frames = self.config.token_budget // (self.config.token_per_image * self.config.num_views)
        uniform_indices = even_sampling_indices(step_idx, max_frames)
        uniform_feats = self._gather_detailed_embs(epis_idx, uniform_indices)
        uniform_img, uniform_pos, uniform_state, uniform_mask = self._pack_frame_sampling(
            uniform_feats, uniform_indices
        )

        result = {
            # Selector inputs
            "cand_embs": cand_embs,
            "cand_times": cand_times,
            "cand_mask": cand_mask,
            "front_view_emb": front_view_emb,
            "proprio": proprio,
            "progress": progress,
            # For packing selected frames after selector runs
            "epis_idx": epis_idx,
            "step_idx": step_idx,
            "num_candidates": num_cands,
            # Pre-packed uniform baseline
            "uniform_static_image_emb": uniform_img,
            "uniform_static_pos_emb": uniform_pos,
            "uniform_static_state_emb": self._normalize_state(uniform_state),
            "uniform_static_mask": uniform_mask,
            # VLA inputs (keys match pkl format, VLA transforms add observation/ prefix)
            "image": data["image"],
            "wrist_image": data["wrist_image"],
            "state": data["state"],
            "actions": actions,
            "prompt": data["prompt"],
            # Keys required by repack transform (not used by selector, but must exist)
            "simple_subgoal": data.get("simple_subgoal", ""),
            "grounded_subgoal": data.get("grounded_subgoal", ""),
            "static_image_emb": None,
            "static_pos_emb": None,
            "static_state_emb": None,
            "static_mask": None,
            "recur_image_emb": None,
            "recur_pos_emb": None,
            "recur_state_emb": None,
            "recur_mask": None,
        }

        return result

    def gather_and_pack_selected(
        self,
        epis_idx: int,
        selected_indices: np.ndarray,
        ds_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """After the selector picks frames, load their detailed embeddings
        and pack into VLA format.

        Args:
            epis_idx: episode index
            selected_indices: (K,) int32 array of chosen timesteps

        Returns:
            (static_image_emb, static_pos_emb, static_state_emb, static_mask)
        """
        # Map tiled indices back to real frame indices
        ep_len = self._get_episode_length(epis_idx)
        indices_list = sorted(set(int(i) % ep_len for i in selected_indices))
        feats = self._gather_detailed_embs(epis_idx, indices_list)
        img, pos, state, mask = self._pack_frame_sampling(feats, indices_list)
        state = self._normalize_state(state)
        return img, pos, state, mask
