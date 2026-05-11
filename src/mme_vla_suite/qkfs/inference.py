"""QKFS inference: diversity-aware top-k frame selection.

At eval timestep t:
1. Build query from instruction, current obs, recent frames, proprio.
2. Score all past frames.
3. Reserve last `reserve_recent` frames.
4. Fill remaining budget with diversity-aware greedy selection:
   adjusted_score[i] = score[i]
       - alpha * max_visual_similarity(h_i, h_j for j in selected)
       - beta * max_temporal_closeness(i, j for j in selected)
5. Return selected frames sorted chronologically.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from mme_vla_suite.qkfs.config import QKFSConfig
from mme_vla_suite.qkfs.model import QKFS

logger = logging.getLogger(__name__)


def load_qkfs(checkpoint_path: str) -> tuple[QKFS, QKFSConfig]:
    """Load a trained QKFS model from checkpoint."""
    ckpt_dir = Path(checkpoint_path)

    with open(ckpt_dir / "config.pkl", "rb") as f:
        config = pickle.load(f)

    model = QKFS(config, rngs=nnx.Rngs(jax.random.key(0)))

    with open(ckpt_dir / "qkfs_params.pkl", "rb") as f:
        saved_state = pickle.load(f)

    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(
        jax.tree.map(lambda x: jnp.asarray(x), saved_state.to_pure_dict())
    )
    model = nnx.merge(graphdef, state)
    model.eval()

    logger.info("Loaded QKFS from %s", checkpoint_path)
    return model, config


def diversity_aware_topk(
    scores: np.ndarray,          # (N,) raw scores
    h: np.ndarray,               # (N, d) history token embeddings
    frame_indices: np.ndarray,   # (N,) absolute frame indices
    cand_mask: np.ndarray,       # (N,) bool
    k: int,                      # total frames to select
    reserve_recent: int,         # how many recent frames to always include
    alpha: float,                # visual similarity penalty weight
    beta: float,                 # temporal closeness penalty weight
    tau: float,                  # temporal closeness decay scale
) -> np.ndarray:
    """Select k frames using diversity-aware greedy selection.

    Returns sorted absolute frame indices of selected frames.
    """
    N = scores.shape[0]
    valid_indices = np.where(cand_mask)[0]

    if len(valid_indices) == 0:
        return np.array([], dtype=np.int32)

    if len(valid_indices) <= k:
        return np.sort(frame_indices[valid_indices])

    # Step 1: Reserve the most recent frames
    valid_frame_ids = frame_indices[valid_indices]
    sorted_by_time = np.argsort(valid_frame_ids)[::-1]  # most recent first

    selected_cand_indices = []
    n_recent = min(reserve_recent, len(valid_indices))
    for i in range(n_recent):
        selected_cand_indices.append(valid_indices[sorted_by_time[i]])

    remaining_budget = k - len(selected_cand_indices)
    if remaining_budget <= 0:
        sel_frame_ids = frame_indices[np.array(selected_cand_indices)]
        return np.sort(sel_frame_ids)

    # Step 2: Greedy diversity-aware selection for remaining budget
    selected_set = set(selected_cand_indices)

    # Precompute normalized embeddings for cosine similarity
    h_norm = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)

    for _ in range(remaining_budget):
        best_idx = -1
        best_adjusted = -np.inf

        for ci in valid_indices:
            if ci in selected_set:
                continue

            # Base score
            adj = scores[ci]

            if selected_cand_indices:
                sel_arr = np.array(selected_cand_indices)

                # Visual similarity penalty: max cosine similarity to any selected
                sim = h_norm[ci] @ h_norm[sel_arr].T  # (num_selected,)
                max_sim = np.max(sim)
                adj -= alpha * max(max_sim, 0.0)

                # Temporal closeness penalty
                time_diffs = np.abs(
                    frame_indices[ci].astype(np.float64) -
                    frame_indices[sel_arr].astype(np.float64)
                )
                temporal_close = np.exp(-time_diffs / tau)
                max_temporal = np.max(temporal_close)
                adj -= beta * max_temporal

            if adj > best_adjusted:
                best_adjusted = adj
                best_idx = ci

        if best_idx < 0:
            break

        selected_cand_indices.append(best_idx)
        selected_set.add(best_idx)

    sel_frame_ids = frame_indices[np.array(selected_cand_indices)]
    return np.sort(sel_frame_ids)


def select_frames_qkfs(
    model: QKFS,
    config: QKFSConfig,
    instruction_emb: np.ndarray,   # (instruction_emb_dim,) — SigLIP text embedding
    current_obs_emb: np.ndarray,   # (frame_emb_dim,)
    all_past_embs: np.ndarray,     # (T, frame_emb_dim) — all past global embeddings
    all_past_proprios: np.ndarray,  # (T, proprio_dim)
    current_proprio: np.ndarray,   # (proprio_dim,)
) -> np.ndarray:
    """Run QKFS to select frames at eval time.

    Args:
        model: trained QKFS model
        config: QKFS config
        instruction_emb: pooled instruction embedding from VLA
        current_obs_emb: current frame's global SigLIP embedding
        all_past_embs: global embeddings for frames 0..t-1
        all_past_proprios: proprio states for frames 0..t-1
        current_proprio: current proprioceptive state

    Returns:
        selected_indices: sorted absolute frame indices to use as memory
    """
    T = all_past_embs.shape[0]

    if T == 0:
        return np.array([], dtype=np.int32)

    if T <= config.num_frames_to_select:
        return np.arange(T, dtype=np.int32)

    N = config.max_candidates
    R = config.num_recent_frames

    # Exclude the most recent R frames from candidates — they are already
    # encoded by the query encoder.  Including them would let the model
    # take a dot-product shortcut instead of learning keyframe relevance.
    cand_end = max(0, T - R)
    if cand_end == 0:
        # All frames are "recent", nothing to score
        return np.arange(T, dtype=np.int32)

    # Subsample if more past frames than max candidates
    if cand_end > N:
        cand_indices_abs = np.linspace(0, cand_end - 1, N, dtype=np.int64)
    else:
        cand_indices_abs = np.arange(cand_end)

    n_cands = len(cand_indices_abs)

    # Prepare candidate arrays (padded to N)
    cand_embs = np.zeros((N, config.frame_emb_dim), dtype=np.float32)
    cand_proprios = np.zeros((N, config.proprio_dim), dtype=np.float32)
    cand_times = np.zeros(N, dtype=np.int32)
    cand_mask = np.zeros(N, dtype=np.bool_)

    cand_embs[:n_cands] = all_past_embs[cand_indices_abs]
    cand_proprios[:n_cands] = all_past_proprios[cand_indices_abs]
    cand_times[:n_cands] = cand_indices_abs.astype(np.int32)
    cand_mask[:n_cands] = True

    # Prepare recent frames
    recent_embs = np.zeros((R, config.frame_emb_dim), dtype=np.float32)
    recent_mask = np.zeros(R, dtype=np.bool_)
    start_recent = max(0, T - R)
    actual_recent = T - start_recent
    if actual_recent > 0:
        recent_embs[:actual_recent] = all_past_embs[start_recent:T]
        recent_mask[:actual_recent] = True

    # Add batch dimension
    batch = {
        "instruction_emb": jnp.asarray(instruction_emb[None]),
        "current_obs_emb": jnp.asarray(current_obs_emb[None]),
        "recent_embs": jnp.asarray(recent_embs[None]),
        "recent_mask": jnp.asarray(recent_mask[None]),
        "proprio": jnp.asarray(current_proprio[None]),
        "cand_embs": jnp.asarray(cand_embs[None]),
        "cand_proprios": jnp.asarray(cand_proprios[None]),
        "cand_times": jnp.asarray(cand_times[None]),
        "cand_mask": jnp.asarray(cand_mask[None]),
    }

    # Forward pass
    q = model.encode_query(
        batch["instruction_emb"], batch["current_obs_emb"],
        batch["recent_embs"], batch["recent_mask"],
        batch["proprio"], deterministic=True,
    )
    h = model.encode_history(
        batch["cand_embs"], batch["cand_proprios"], batch["cand_times"],
    )
    logits = model.score_frames(q, h, batch["cand_mask"], deterministic=True)

    # Get numpy arrays
    scores = np.array(jax.device_get(logits[0]))  # (N,)
    h_np = np.array(jax.device_get(h[0]))          # (N, d)
    frame_indices = np.array(cand_times)            # (N,) absolute indices
    mask_np = np.array(cand_mask)                   # (N,)

    # Diversity-aware top-k selection
    selected = diversity_aware_topk(
        scores=scores,
        h=h_np,
        frame_indices=frame_indices,
        cand_mask=mask_np,
        k=config.num_frames_to_select,
        reserve_recent=config.reserve_recent,
        alpha=config.alpha,
        beta=config.beta,
        tau=config.tau,
    )

    return selected
