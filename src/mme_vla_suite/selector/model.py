"""RL-based frame selector network.

Architecture:
- Query encoder: 2-layer MLP over [front_view_emb, instruction_emb, proprio, progress].
- Candidate projection: linear from candidate_emb_dim → query_dim.
- Scorer MLP: [q, cand_proj, q*cand_proj, cand_time] → scalar score.
- Value head: MLP on query → scalar baseline for PPO.

Selection uses parallel top-K via Gumbel-Softmax (training) or
deterministic argmax (eval).  Log-probs are computed via the
Plackett-Luce decomposition for PPO compatibility.
"""

import functools
from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from mme_vla_suite.selector.config import SelectorConfig


class SelectionOutput(NamedTuple):
    selected_indices: jnp.ndarray   # (K,) int32
    log_probs: jnp.ndarray          # (K,) per-step log-prob (Plackett-Luce)
    total_log_prob: jnp.ndarray     # scalar
    entropy: jnp.ndarray            # scalar (mean per-step entropy)
    value: jnp.ndarray              # scalar baseline


class FrameSelector(nnx.Module):
    """Selects K frames from a variable-length candidate set."""

    def __init__(self, config: SelectorConfig, rngs: nnx.Rngs):
        self.config = config
        K = config.num_frames_to_select
        q_dim = config.query_dim

        # --- instruction projection (from pooled LLM embeddings) ---
        self.instruction_proj = nnx.Linear(config.front_view_emb_dim, config.instruction_emb_dim, rngs=rngs)

        # --- query encoder (2-layer MLP) ---
        query_input_dim = (
            config.front_view_emb_dim
            + config.instruction_emb_dim
            + config.proprio_dim
            + (1 if config.use_episode_progress else 0)
        )
        self.query_fc1 = nnx.Linear(query_input_dim, q_dim, rngs=rngs)
        self.query_fc2 = nnx.Linear(q_dim, q_dim, rngs=rngs)

        # --- candidate projection ---
        self.cand_proj = nnx.Linear(config.candidate_emb_dim, q_dim, rngs=rngs)

        # --- scorer MLP ---
        # input: [q, cand_proj, q * cand_proj, cand_time]  →  3*q_dim + 1
        scorer_in = 3 * q_dim + 1
        self.scorer_fc1 = nnx.Linear(scorer_in, config.scorer_hidden_dim, rngs=rngs)
        self.scorer_fc2 = nnx.Linear(config.scorer_hidden_dim, 1, rngs=rngs)

        # --- value head ---
        self.value_fc1 = nnx.Linear(q_dim, config.scorer_hidden_dim, rngs=rngs)
        self.value_fc2 = nnx.Linear(config.scorer_hidden_dim, 1, rngs=rngs)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def encode_query(
        self,
        front_view_emb: jnp.ndarray,     # (b, front_view_emb_dim)
        instruction_emb: jnp.ndarray,     # (b, front_view_emb_dim)  -- pooled LLM tokens
        proprio: jnp.ndarray,             # (b, proprio_dim)
        progress: jnp.ndarray | None,     # (b, 1)  normalized ∈ [0,1]
    ) -> jnp.ndarray:
        """Returns query vector (b, query_dim)."""
        instr_proj = nnx.relu(self.instruction_proj(instruction_emb))  # (b, instr_dim)
        parts = [front_view_emb, instr_proj, proprio]
        if progress is not None:
            parts.append(progress)
        x = jnp.concatenate(parts, axis=-1)
        x = nnx.relu(self.query_fc1(x))
        q = nnx.relu(self.query_fc2(x))
        return q

    def score_candidates(
        self,
        q: jnp.ndarray,            # (b, q_dim)  or (q_dim,)
        cand_embs: jnp.ndarray,    # (b, N, cand_dim) or (N, cand_dim)
        cand_times: jnp.ndarray,   # (b, N, 1) or (N, 1)
    ) -> jnp.ndarray:
        """Returns raw logits (b, N) or (N,)."""
        cand_proj = self.cand_proj(cand_embs)               # (..., N, q_dim)
        if q.ndim == cand_proj.ndim - 1:
            q_exp = jnp.expand_dims(q, axis=-2)             # (..., 1, q_dim)
        else:
            q_exp = q
        q_broad = jnp.broadcast_to(q_exp, cand_proj.shape)  # (..., N, q_dim)
        interaction = q_broad * cand_proj                    # (..., N, q_dim)
        scorer_in = jnp.concatenate([q_broad, cand_proj, interaction, cand_times], axis=-1)
        h = nnx.relu(self.scorer_fc1(scorer_in))
        logits = self.scorer_fc2(h).squeeze(-1)              # (..., N)
        return logits

    def predict_value(self, q: jnp.ndarray) -> jnp.ndarray:
        """Baseline value V(q) → scalar per batch element."""
        h = nnx.relu(self.value_fc1(q))
        return self.value_fc2(h).squeeze(-1)

    # ------------------------------------------------------------------
    # Parallel selection (Gumbel-top-K for training, argmax for eval)
    # ------------------------------------------------------------------

    def _select_gumbel_topk(
        self,
        rng: jnp.ndarray,
        logits: jnp.ndarray,       # (N,)
        cand_mask: jnp.ndarray,    # (N,) bool
    ) -> jnp.ndarray:
        """Sample K indices without replacement using Gumbel-top-k.

        Returns (K,) int32 selected indices.
        """
        K = self.config.num_frames_to_select
        # Mask invalid candidates
        logits = jnp.where(cand_mask, logits, -1e9)
        # Add Gumbel noise for stochastic selection
        gumbel_noise = jax.random.gumbel(rng, logits.shape)
        perturbed = logits + gumbel_noise
        perturbed = jnp.where(cand_mask, perturbed, -1e9)
        # Top-K selection
        _, indices = jax.lax.top_k(perturbed, K)
        return indices

    def _plackett_luce_log_probs(
        self,
        logits: jnp.ndarray,       # (N,)
        cand_mask: jnp.ndarray,    # (N,) bool
        selected: jnp.ndarray,     # (K,) int32
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute per-selection log-probs under the Plackett-Luce model.

        For each selection k, the log-prob is:
            log p(selected[k]) = logits[selected[k]] - logsumexp(remaining logits)

        This is equivalent to the sequential categorical decomposition
        but computed in parallel via cumulative masking.

        Returns (log_probs (K,), entropies (K,)).
        """
        K = selected.shape[0]
        N = logits.shape[0]

        # Sort selected indices to process in order
        order = jnp.argsort(selected)
        sorted_sel = selected[order]

        masked_logits = jnp.where(cand_mask, logits, -1e9)

        def step_fn(avail_logits, k):
            idx = sorted_sel[k]
            log_p = jax.nn.log_softmax(avail_logits)
            lp = log_p[idx]
            # Clamp to prevent -inf when selecting masked/exhausted candidates
            lp = jnp.maximum(lp, -20.0)

            # Entropy of current distribution
            p = jnp.exp(log_p)
            valid = avail_logits > -1e8
            ent = -jnp.sum(jnp.where(valid, p * log_p, 0.0))
            # Clamp entropy to be non-negative (numerical noise)
            ent = jnp.maximum(ent, 0.0)

            # Remove selected from available
            new_logits = avail_logits.at[idx].set(-1e9)
            return new_logits, (lp, ent)

        _, (sorted_lps, sorted_ents) = jax.lax.scan(
            step_fn, masked_logits, jnp.arange(K)
        )

        # Unsort back to original selection order
        inv_order = jnp.argsort(order)
        log_probs = sorted_lps[inv_order]
        entropies = sorted_ents[inv_order]

        return log_probs, entropies

    # ------------------------------------------------------------------
    # Batched forward (for training)
    # ------------------------------------------------------------------

    def forward_batched(
        self,
        rng: jnp.ndarray,
        front_view_emb: jnp.ndarray,     # (B, front_view_emb_dim)
        instruction_emb: jnp.ndarray,     # (B, front_view_emb_dim)
        proprio: jnp.ndarray,             # (B, proprio_dim)
        progress: jnp.ndarray | None,     # (B, 1)
        cand_embs: jnp.ndarray,           # (B, N, cand_dim)
        cand_times: jnp.ndarray,          # (B, N, 1)
        cand_mask: jnp.ndarray,           # (B, N) bool
    ) -> SelectionOutput:
        """Batched forward: scores all candidates, selects K via Gumbel-top-k."""
        B = front_view_emb.shape[0]

        # Encode queries (batched)
        q = self.encode_query(front_view_emb, instruction_emb, proprio, progress)  # (B, q_dim)

        # Score all candidates (batched)
        logits = self.score_candidates(q, cand_embs, cand_times)  # (B, N)

        # Select K per sample via Gumbel-top-k
        rngs = jax.random.split(rng, B)
        selected = jax.vmap(self._select_gumbel_topk)(rngs, logits, cand_mask)  # (B, K)

        # Compute Plackett-Luce log-probs (vmapped over batch)
        log_probs, entropies = jax.vmap(self._plackett_luce_log_probs)(
            logits, cand_mask, selected
        )  # both (B, K)

        # Value prediction (batched)
        values = self.predict_value(q)  # (B,)

        total_log_prob = jnp.sum(log_probs, axis=-1)  # (B,)
        # Clamp total log-prob to prevent -inf (can happen when K > num_valid_candidates)
        total_log_prob = jnp.maximum(total_log_prob, -100.0)

        return SelectionOutput(
            selected_indices=selected,              # (B, K)
            log_probs=log_probs,                    # (B, K)
            total_log_prob=total_log_prob,           # (B,)
            entropy=jnp.mean(entropies, axis=-1),   # (B,)
            value=values,                            # (B,)
        )

    def evaluate_batched(
        self,
        front_view_emb: jnp.ndarray,     # (B, front_view_emb_dim)
        instruction_emb: jnp.ndarray,     # (B, front_view_emb_dim)
        proprio: jnp.ndarray,             # (B, proprio_dim)
        progress: jnp.ndarray | None,     # (B, 1)
        cand_embs: jnp.ndarray,           # (B, N, cand_dim)
        cand_times: jnp.ndarray,          # (B, N, 1)
        cand_mask: jnp.ndarray,           # (B, N) bool
        chosen_indices: jnp.ndarray,      # (B, K) int32
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Batched recompute for PPO update.

        Returns (total_log_prob (B,), mean_entropy (B,), value (B,)).
        """
        q = self.encode_query(front_view_emb, instruction_emb, proprio, progress)
        logits = self.score_candidates(q, cand_embs, cand_times)

        log_probs, entropies = jax.vmap(self._plackett_luce_log_probs)(
            logits, cand_mask, chosen_indices
        )
        values = self.predict_value(q)

        total_lp = jnp.maximum(jnp.sum(log_probs, axis=-1), -100.0)
        return total_lp, jnp.mean(entropies, axis=-1), values

    # ------------------------------------------------------------------
    # Single-sample forward (kept for inference compatibility)
    # ------------------------------------------------------------------

    def forward_single(
        self,
        rng: jnp.ndarray,
        front_view_emb: jnp.ndarray,     # (front_view_emb_dim,)
        instruction_emb: jnp.ndarray,     # (front_view_emb_dim,)
        proprio: jnp.ndarray,             # (proprio_dim,)
        progress: jnp.ndarray | None,     # (1,)
        cand_embs: jnp.ndarray,           # (N, cand_dim)
        cand_times: jnp.ndarray,          # (N, 1)
        cand_mask: jnp.ndarray,           # (N,) bool
    ) -> SelectionOutput:
        # Add batch dim, run batched, squeeze
        out = self.forward_batched(
            rng,
            front_view_emb[None], instruction_emb[None], proprio[None],
            progress[None] if progress is not None else None,
            cand_embs[None], cand_times[None], cand_mask[None],
        )
        return SelectionOutput(
            selected_indices=out.selected_indices[0],
            log_probs=out.log_probs[0],
            total_log_prob=out.total_log_prob[0],
            entropy=out.entropy[0],
            value=out.value[0],
        )

    def evaluate_single(
        self,
        front_view_emb: jnp.ndarray,
        instruction_emb: jnp.ndarray,
        proprio: jnp.ndarray,
        progress: jnp.ndarray | None,
        cand_embs: jnp.ndarray,
        cand_times: jnp.ndarray,
        cand_mask: jnp.ndarray,
        chosen_indices: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Recompute log-probs and value for PPO update (single sample)."""
        return self.evaluate_batched(
            front_view_emb[None], instruction_emb[None], proprio[None],
            progress[None] if progress is not None else None,
            cand_embs[None], cand_times[None], cand_mask[None],
            chosen_indices[None],
        )

    def select_eval(
        self,
        front_view_emb: jnp.ndarray,
        instruction_emb: jnp.ndarray,
        proprio: jnp.ndarray,
        progress: jnp.ndarray | None,
        cand_embs: jnp.ndarray,
        cand_times: jnp.ndarray,
        cand_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Deterministic top-k selection for evaluation. Returns sorted indices."""
        q = self.encode_query(
            front_view_emb[None], instruction_emb[None], proprio[None],
            progress[None] if progress is not None else None,
        )[0]
        logits = self.score_candidates(q, cand_embs, cand_times)
        logits = jnp.where(cand_mask, logits, -1e9)
        _, indices = jax.lax.top_k(logits, self.config.num_frames_to_select)
        return jnp.sort(indices)