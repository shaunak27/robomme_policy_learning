"""Query-conditioned KeyFrame Selector (QKFS) model.

Architecture:
1. QueryEncoder: Transformer that encodes [instruction, current obs, last R frames, proprio]
   into a query vector q_t.
2. FrameEncoder: MLP that projects each past frame (global_emb + proprio + gripper + temporal)
   into a history token h_i.
3. Selector: Cross-attention transformer from q_t to H_t, producing one scalar score
   s_ti per past frame.

The model outputs a probability distribution p_ti = softmax(s_ti) over past frames.
Training uses KL divergence against target distributions derived from subtask annotations.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from mme_vla_suite.qkfs.config import QKFSConfig


class TransformerBlock(nnx.Module):
    """Standard pre-norm transformer block with self-attention."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int,
                 dropout_rate: float, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None,
                 deterministic: bool = True) -> jnp.ndarray:
        # Self-attention
        h = self.norm1(x)
        h = self.attn(h, mask=mask)
        if not deterministic:
            h = self.dropout(h)
        x = x + h

        # FFN
        h = self.norm2(x)
        h = nnx.gelu(self.fc1(h))
        if not deterministic:
            h = self.dropout(h)
        h = self.fc2(h)
        if not deterministic:
            h = self.dropout(h)
        x = x + h
        return x


class CrossAttentionBlock(nnx.Module):
    """Pre-norm cross-attention block: query attends to key-value (history)."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int,
                 dropout_rate: float, rngs: nnx.Rngs):
        self.norm_q = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(dim, rngs=rngs)
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            decode=False,
            rngs=rngs,
        )
        self.norm_ff = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, q: jnp.ndarray, kv: jnp.ndarray,
                 kv_mask: jnp.ndarray | None = None,
                 deterministic: bool = True) -> jnp.ndarray:
        # Cross-attention
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        # mask shape for cross-attn: (batch, 1, q_len, kv_len) or (batch, kv_len)
        h = self.cross_attn(q_norm, kv_norm, mask=kv_mask)
        if not deterministic:
            h = self.dropout(h)
        q = q + h

        # FFN
        h = self.norm_ff(q)
        h = nnx.gelu(self.fc1(h))
        if not deterministic:
            h = self.dropout(h)
        h = self.fc2(h)
        if not deterministic:
            h = self.dropout(h)
        q = q + h
        return q


class QueryEncoder(nnx.Module):
    """Encodes the current query context into a query vector.

    Input tokens:
    - instruction token(s): projected instruction embedding
    - current obs token: projected current frame global embedding
    - recent frame tokens: projected last R frame embeddings
    - proprio token: projected proprioceptive state

    Output: q_t of shape (hidden_dim,) via mean-pooling over token outputs.
    """

    def __init__(self, config: QKFSConfig, rngs: nnx.Rngs):
        d = config.hidden_dim

        # Input projections
        self.instruction_proj = nnx.Linear(config.instruction_emb_dim, d, rngs=rngs)
        self.obs_proj = nnx.Linear(config.frame_emb_dim, d, rngs=rngs)
        self.recent_proj = nnx.Linear(config.frame_emb_dim, d, rngs=rngs)
        self.proprio_proj = nnx.Linear(config.proprio_dim, d, rngs=rngs)

        # Learnable type embeddings (4 types: instruction, current_obs, recent, proprio)
        self.type_embeddings = nnx.Embed(4, d, rngs=rngs)

        # Positional embedding for recent frames (relative position within R)
        self.recent_pos_emb = nnx.Embed(config.num_recent_frames, d, rngs=rngs)

        # Transformer layers
        self.layers = [
            TransformerBlock(d, config.num_heads, config.mlp_dim,
                             config.dropout_rate, rngs)
            for _ in range(config.num_query_layers)
        ]
        self.final_norm = nnx.LayerNorm(d, rngs=rngs)

    def __call__(
        self,
        instruction_emb: jnp.ndarray,   # (B, instruction_emb_dim)
        current_obs_emb: jnp.ndarray,   # (B, frame_emb_dim)
        recent_embs: jnp.ndarray,       # (B, R, frame_emb_dim)
        recent_mask: jnp.ndarray,       # (B, R) bool
        proprio: jnp.ndarray,           # (B, proprio_dim)
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Returns query vector (B, hidden_dim)."""
        B = instruction_emb.shape[0]
        R = recent_embs.shape[1]

        # Project each input type
        instr_tok = self.instruction_proj(instruction_emb)[:, None, :]   # (B, 1, d)
        obs_tok = self.obs_proj(current_obs_emb)[:, None, :]             # (B, 1, d)
        recent_toks = self.recent_proj(recent_embs)                       # (B, R, d)
        proprio_tok = self.proprio_proj(proprio)[:, None, :]             # (B, 1, d)

        # Add type embeddings
        instr_tok = instr_tok + self.type_embeddings(jnp.zeros((B, 1), dtype=jnp.int32))
        obs_tok = obs_tok + self.type_embeddings(jnp.ones((B, 1), dtype=jnp.int32))
        recent_toks = recent_toks + self.type_embeddings(
            jnp.full((B, R), 2, dtype=jnp.int32)
        )
        proprio_tok = proprio_tok + self.type_embeddings(
            jnp.full((B, 1), 3, dtype=jnp.int32)
        )

        # Add positional embedding for recent frames
        recent_pos_ids = jnp.arange(R)[None, :]  # (1, R)
        recent_toks = recent_toks + self.recent_pos_emb(
            jnp.broadcast_to(recent_pos_ids, (B, R))
        )

        # Concatenate all tokens: [instr, obs, recent_0..R-1, proprio]
        tokens = jnp.concatenate([instr_tok, obs_tok, recent_toks, proprio_tok], axis=1)
        # (B, 2 + R + 1, d)

        # Build attention mask: mask out padded recent frames
        fixed_mask = jnp.ones((B, 3), dtype=jnp.bool_)  # instr + obs + proprio
        # Insert recent mask between obs and proprio
        full_mask = jnp.concatenate(
            [fixed_mask[:, :2], recent_mask, fixed_mask[:, 2:3]], axis=1
        )  # (B, 2 + R + 1)

        # Convert to self-attention mask: (B, 1, T, T)
        # Each token can attend to all non-masked tokens
        attn_mask = full_mask[:, None, None, :]  # (B, 1, 1, T) broadcast over query dim

        # Apply transformer
        x = tokens
        for layer in self.layers:
            x = layer(x, mask=attn_mask, deterministic=deterministic)
        x = self.final_norm(x)

        # Mean-pool over non-masked tokens
        mask_expanded = full_mask[:, :, None].astype(x.dtype)  # (B, T, 1)
        q = jnp.sum(x * mask_expanded, axis=1) / jnp.maximum(
            mask_expanded.sum(axis=1), 1.0
        )  # (B, d)
        return q


class FrameEncoder(nnx.Module):
    """Encodes each past frame into a history token.

    Input per frame: global_emb (2048) + proprio (8) + temporal_emb
    Output: h_i of shape (hidden_dim,)
    """

    def __init__(self, config: QKFSConfig, rngs: nnx.Rngs):
        d = config.hidden_dim
        input_dim = config.frame_emb_dim + config.proprio_dim

        self.proj = nnx.Linear(input_dim, d, rngs=rngs)
        self.norm = nnx.LayerNorm(d, rngs=rngs)
        self.fc = nnx.Linear(d, d, rngs=rngs)

        # Temporal embedding: learned embedding for relative age
        # Discretize relative age into bins
        self.temporal_emb = nnx.Embed(config.max_episode_length, d, rngs=rngs)

    def __call__(
        self,
        frame_embs: jnp.ndarray,    # (B, N, frame_emb_dim)
        frame_proprios: jnp.ndarray, # (B, N, proprio_dim)
        frame_times: jnp.ndarray,   # (B, N) int32 — absolute timestep indices
    ) -> jnp.ndarray:
        """Returns history tokens (B, N, hidden_dim)."""
        # Concatenate frame features
        x = jnp.concatenate([frame_embs, frame_proprios], axis=-1)  # (B, N, emb+proprio)
        x = nnx.gelu(self.proj(x))

        # Add temporal embedding
        # Clamp times to valid range
        times_clamped = jnp.clip(frame_times, 0, self.temporal_emb.num_embeddings - 1)
        t_emb = self.temporal_emb(times_clamped)  # (B, N, d)
        x = x + t_emb

        x = self.norm(x)
        x = nnx.gelu(self.fc(x))
        return x  # (B, N, d)


class QKFS(nnx.Module):
    """Query-conditioned KeyFrame Selector.

    Computes per-frame scores s_ti via cross-attention from query q_t to
    history tokens H = {h_i}, then converts to probabilities p_ti = softmax(s).
    """

    def __init__(self, config: QKFSConfig, rngs: nnx.Rngs):
        self.config = config
        d = config.hidden_dim

        self.query_encoder = QueryEncoder(config, rngs)
        self.frame_encoder = FrameEncoder(config, rngs)

        # Selector: cross-attention layers from query to history
        self.selector_layers = [
            CrossAttentionBlock(d, config.num_heads, config.mlp_dim,
                                config.dropout_rate, rngs)
            for _ in range(config.num_selector_layers)
        ]
        self.selector_norm = nnx.LayerNorm(d, rngs=rngs)

        # Score projection: maps each cross-attended query-frame pair to scalar
        self.score_proj = nnx.Linear(d, 1, rngs=rngs)

    def encode_query(
        self,
        instruction_emb: jnp.ndarray,   # (B, instruction_emb_dim)
        current_obs_emb: jnp.ndarray,   # (B, frame_emb_dim)
        recent_embs: jnp.ndarray,       # (B, R, frame_emb_dim)
        recent_mask: jnp.ndarray,       # (B, R) bool
        proprio: jnp.ndarray,           # (B, proprio_dim)
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Encode query context -> (B, hidden_dim)."""
        return self.query_encoder(
            instruction_emb, current_obs_emb, recent_embs, recent_mask,
            proprio, deterministic=deterministic,
        )

    def encode_history(
        self,
        frame_embs: jnp.ndarray,     # (B, N, frame_emb_dim)
        frame_proprios: jnp.ndarray,  # (B, N, proprio_dim)
        frame_times: jnp.ndarray,    # (B, N) int32
    ) -> jnp.ndarray:
        """Encode past frames -> (B, N, hidden_dim)."""
        return self.frame_encoder(frame_embs, frame_proprios, frame_times)

    def score_frames(
        self,
        q: jnp.ndarray,             # (B, hidden_dim)
        h: jnp.ndarray,             # (B, N, hidden_dim)
        cand_mask: jnp.ndarray,     # (B, N) bool
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Score each past frame given query. Returns logits (B, N)."""
        # Expand query to sequence dim for cross-attention: (B, 1, d)
        q_seq = q[:, None, :]

        # Cross-attention mask: (B, 1, 1, N)
        kv_mask = cand_mask[:, None, None, :]

        # Apply cross-attention layers
        x = q_seq
        for layer in self.selector_layers:
            x = layer(x, h, kv_mask=kv_mask, deterministic=deterministic)
        x = self.selector_norm(x)  # (B, 1, d)

        # Now compute per-frame scores via dot-product between
        # the cross-attended query and each history token
        # x: (B, 1, d), h: (B, N, d)
        scores = jnp.sum(x * h, axis=-1)  # (B, N) — dot product
        # Alternative: use learned score projection on concatenated features
        # But dot-product is simpler and works well

        # Mask out invalid frames
        scores = jnp.where(cand_mask, scores, -1e9)
        return scores

    def forward(
        self,
        instruction_emb: jnp.ndarray,   # (B, instruction_emb_dim)
        current_obs_emb: jnp.ndarray,   # (B, frame_emb_dim)
        recent_embs: jnp.ndarray,       # (B, R, frame_emb_dim)
        recent_mask: jnp.ndarray,       # (B, R) bool
        proprio: jnp.ndarray,           # (B, proprio_dim)
        frame_embs: jnp.ndarray,        # (B, N, frame_emb_dim)
        frame_proprios: jnp.ndarray,    # (B, N, proprio_dim)
        frame_times: jnp.ndarray,       # (B, N) int32
        cand_mask: jnp.ndarray,         # (B, N) bool
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Full forward pass. Returns log-probabilities (B, N)."""
        q = self.encode_query(
            instruction_emb, current_obs_emb, recent_embs, recent_mask,
            proprio, deterministic=deterministic,
        )
        h = self.encode_history(frame_embs, frame_proprios, frame_times)
        logits = self.score_frames(q, h, cand_mask, deterministic=deterministic)

        # Convert to log-probabilities
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Re-mask after softmax
        log_probs = jnp.where(cand_mask, log_probs, -1e9)
        return log_probs
