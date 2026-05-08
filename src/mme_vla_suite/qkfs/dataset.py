"""Dataset for QKFS training.

Each sample provides:
- Query inputs: instruction emb, current obs emb, recent frame embs, proprio
- History inputs: all past frame global embeddings, proprios, timestamps
- Target distributions: P_target (source-subtask) and T_j (within-subtask)
  pre-computed and mapped to candidate frame positions.

Loads consolidated per-episode arrays (global_emb.npy, detail_state.npy)
and segment/keyframe annotations.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
from openpi.training.data_loader import Dataset

from mme_vla_suite.qkfs.config import QKFSConfig
from mme_vla_suite.qkfs.targets import (
    build_target_distributions,
    load_topreward_intervals,
)

logger = logging.getLogger(__name__)


class QKFSDataset(Dataset):
    """Yields samples for QKFS supervised training."""

    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        config: QKFSConfig,
        topreward_dir: str = "",
    ):
        self.config = config
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.feature_dir = Path(dataset_path) / "features"
        self.topreward_dir = topreward_dir

        stats_path = os.path.join(dataset_path, "meta", "stats.json")
        self.stats = json.load(open(stats_path))

        # Caches
        self._global_embs: dict[int, np.ndarray] = {}
        self._states: dict[int, np.ndarray] = {}
        self._segments: dict[int, list[dict]] = {}
        self._density_kfs: dict[int, dict] = {}
        self._topreward: dict[int, dict] = {}
        self._episode_lengths: dict[int, int] = {}

    def __len__(self):
        if "execution_samples" in self.stats:
            return self.stats["execution_samples"]
        return self.stats["total_samples"]

    def _get_episode_data(self, epis_idx: int):
        """Load and cache per-episode arrays."""
        if epis_idx not in self._global_embs:
            ep_dir = self.feature_dir / f"episode_{epis_idx}"

            # Global embeddings
            global_path = ep_dir / "global_emb.npy"
            self._global_embs[epis_idx] = np.load(global_path)  # (T, 2048)

            # States
            state_path = ep_dir / "detail_state.npy"
            if state_path.exists():
                self._states[epis_idx] = np.load(state_path)  # (T, 8)
            else:
                # Fallback: load from individual files
                T = self._global_embs[epis_idx].shape[0]
                states = np.zeros((T, self.config.proprio_dim), dtype=np.float32)
                for t in range(T):
                    tok_path = ep_dir / f"token_emb_{t}.npy"
                    if tok_path.exists():
                        data = np.load(str(tok_path), allow_pickle=True).item()
                        if "state_emb" in data:
                            states[t] = data["state_emb"]
                self._states[epis_idx] = states

            self._episode_lengths[epis_idx] = self._global_embs[epis_idx].shape[0]

            # Segments
            seg_path = ep_dir / "segments.json"
            if seg_path.exists():
                self._segments[epis_idx] = json.load(open(seg_path))
            else:
                self._segments[epis_idx] = []

            # Density keyframes
            dk_path = ep_dir / "density_keyframes.json"
            if dk_path.exists():
                self._density_kfs[epis_idx] = json.load(open(dk_path))
            else:
                self._density_kfs[epis_idx] = {}

            # TOPReward intervals
            if self.topreward_dir and self._segments[epis_idx]:
                self._topreward[epis_idx] = load_topreward_intervals(
                    self.topreward_dir, self.task_name, epis_idx,
                    self._segments[epis_idx],
                )
            else:
                self._topreward[epis_idx] = {}

    def __getitem__(self, idx: int) -> dict:
        # Load base sample
        with open(os.path.join(self.dataset_path, "data", f"{idx}.pkl"), "rb") as f:
            data = pickle.load(f)

        epis_idx = int(data["epis_idx"].item()) if hasattr(data["epis_idx"], "item") else int(data["epis_idx"])
        step_idx = int(data["step_idx"].item()) if hasattr(data["step_idx"], "item") else int(data["step_idx"])

        # Load episode-level data
        self._get_episode_data(epis_idx)

        global_embs = self._global_embs[epis_idx]   # (T, 2048)
        states = self._states[epis_idx]               # (T, 8)
        segments = self._segments[epis_idx]
        density_kfs = self._density_kfs[epis_idx]
        topreward_intervals = self._topreward.get(epis_idx, {})
        episode_length = self._episode_lengths[epis_idx]

        N = self.config.max_candidates
        R = self.config.num_recent_frames
        emb_dim = self.config.frame_emb_dim
        proprio_dim = self.config.proprio_dim

        # ---- Current observation embedding ----
        current_obs_emb = global_embs[min(step_idx, len(global_embs) - 1)]  # (2048,)

        # ---- Instruction embedding (use current obs as proxy; real one from VLA at train time) ----
        # We'll use the global_emb of the first frame as instruction proxy
        # The real instruction embedding will be computed by the VLA's LLM
        instruction_emb = global_embs[0]  # (2048,) — placeholder, overridden in training

        # ---- Recent R frames ----
        recent_embs = np.zeros((R, emb_dim), dtype=np.float32)
        recent_mask = np.zeros(R, dtype=np.bool_)
        start_recent = max(0, step_idx - R)
        actual_recent = step_idx - start_recent
        if actual_recent > 0:
            recent_embs[:actual_recent] = global_embs[start_recent:step_idx]
            recent_mask[:actual_recent] = True

        # ---- Proprio ----
        proprio = data["state"].astype(np.float32)

        # ---- Candidate history frames (all past frames) ----
        num_past = step_idx  # frames 0..step_idx-1
        cand_embs = np.zeros((N, emb_dim), dtype=np.float32)
        cand_proprios = np.zeros((N, proprio_dim), dtype=np.float32)
        cand_times = np.zeros(N, dtype=np.int32)
        cand_mask = np.zeros(N, dtype=np.bool_)
        cand_frame_indices = np.zeros(N, dtype=np.int32)  # absolute frame index

        if num_past > 0:
            if num_past <= N:
                # All past frames fit
                cand_embs[:num_past] = global_embs[:num_past]
                cand_proprios[:num_past] = states[:num_past]
                cand_times[:num_past] = np.arange(num_past)
                cand_mask[:num_past] = True
                cand_frame_indices[:num_past] = np.arange(num_past)
            else:
                # Subsample: uniform spread across past
                indices = np.linspace(0, num_past - 1, N, dtype=np.int64)
                cand_embs[:] = global_embs[indices]
                cand_proprios[:] = states[indices]
                cand_times[:] = indices.astype(np.int32)
                cand_mask[:] = True
                cand_frame_indices[:] = indices.astype(np.int32)

        n_valid = int(cand_mask.sum())

        # ---- Build target distributions ----
        S = self.config.max_segments  # fixed size to avoid JIT retrace

        # Initialize target arrays
        target_frame_dist = np.zeros(N, dtype=np.float32)
        target_seg_assignment = np.zeros(N, dtype=np.int32)
        target_seg_weights = np.zeros(S, dtype=np.float32)

        if len(segments) > S:
            logger.warning("Episode %d has %d segments > max_segments=%d, clamping",
                           epis_idx, len(segments), S)

        if segments and step_idx > 0:
            targets = build_target_distributions(
                step_idx=step_idx,
                segments=segments,
                task_name=self.task_name,
                task_goal=data["prompt"],
                density_keyframes=density_kfs,
                topreward_intervals=topreward_intervals,
                sigma=self.config.sigma,
                max_frames=self.config.num_frames_to_select,
                epsilon=self.config.epsilon,
            )

            if targets is not None:
                # Map segment assignment for each candidate
                seg_lookup = {s["idx"]: s for s in segments}
                for i in range(n_valid):
                    frame_idx = int(cand_frame_indices[i])
                    for seg in segments:
                        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
                            target_seg_assignment[i] = min(seg["idx"], S - 1)
                            break

                # Map P_target to array
                for seg_idx, weight in targets["P_target"].items():
                    if seg_idx < S:
                        target_seg_weights[seg_idx] = weight

                # Map T_j to candidate positions
                for seg_idx, T_j in targets["T_j"].items():
                    if T_j is None or len(T_j) == 0:
                        continue
                    seg = seg_lookup.get(seg_idx)
                    if seg is None:
                        continue
                    for i in range(n_valid):
                        frame_idx = int(cand_frame_indices[i])
                        rel_idx = frame_idx - seg["start_frame"]
                        if 0 <= rel_idx < len(T_j):
                            if target_seg_assignment[i] == seg_idx:
                                target_frame_dist[i] = T_j[rel_idx]
                has_target = True
            else:
                has_target = False
        else:
            has_target = False

        return {
            # Query inputs
            "instruction_emb": instruction_emb,
            "current_obs_emb": current_obs_emb,
            "recent_embs": recent_embs,
            "recent_mask": recent_mask,
            "proprio": proprio,
            # History inputs
            "cand_embs": cand_embs,
            "cand_proprios": cand_proprios,
            "cand_times": cand_times,
            "cand_mask": cand_mask,
            "cand_frame_indices": cand_frame_indices,
            # Targets
            "target_frame_dist": target_frame_dist,
            "target_seg_assignment": target_seg_assignment,
            "target_seg_weights": target_seg_weights,
            "has_target": np.array(has_target, dtype=np.bool_),
            # Metadata
            "epis_idx": epis_idx,
            "step_idx": step_idx,
            "prompt": data["prompt"],
        }


class MultiTaskQKFSDataset(Dataset):
    """Wraps multiple per-task QKFSDatasets for cross-task training."""

    def __init__(self, datasets: list[QKFSDataset]):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> dict:
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                if i == 0:
                    local_idx = idx
                else:
                    local_idx = idx - self.cumulative_sizes[i - 1]
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
