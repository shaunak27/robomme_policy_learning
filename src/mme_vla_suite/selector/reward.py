"""Reward computation for the RL frame selector.

    reward = FM_loss_uniform - FM_loss_selector - redundancy_penalty

FM losses are computed under the *same* random noise and timestep to reduce
variance.  The VLA is frozen; only the selector's choice of frames changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from mme_vla_suite.selector.config import SelectorConfig

if TYPE_CHECKING:
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
    from mme_vla_suite.models.integration.history_observation import HistAugObservation


def compute_fm_loss_deterministic(
    model: HistoryPi0,
    observation: HistAugObservation,
    actions: jnp.ndarray,
    noise: jnp.ndarray,
    time: jnp.ndarray,
):
    """Compute flow-matching loss with pre-sampled noise and timestep.

    Matches the logic in HistoryPi0.compute_loss but uses provided
    noise/time so that selector and uniform comparisons share randomness.

    Returns per-sample mean loss (scalar when unbatched, (B,) when batched).
    """
    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask

    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, stats = (
        model.embed_prefix(observation)
    )
    suffix_tokens, suffix_mask, suffix_ar_mask, suffix_na_mask, adarms_cond = (
        model.embed_suffix(observation, x_t, time)
    )

    if model.integration_type == "expert":
        mem_tokens, mem_input_mask, mem_ar_mask, mem_na_mask, _ = (
            model.embed_memory(observation)
        )
        mem_ar_mask = jnp.array(mem_ar_mask)
        mem_na_mask = jnp.array(mem_na_mask)
        input_mask = jnp.concatenate([mem_input_mask, prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([mem_ar_mask, prefix_ar_mask, suffix_ar_mask], axis=0)
        na_mask = jnp.concatenate([mem_na_mask, prefix_na_mask, suffix_na_mask], axis=0)
    else:
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        na_mask = jnp.concatenate([prefix_na_mask, suffix_na_mask], axis=0)

    if model.use_history and model.representation_type != "symbolic":
        attn_mask = make_attn_mask(input_mask, ar_mask, na_mask)
    else:
        attn_mask = make_attn_mask(input_mask, ar_mask)

    positions = jnp.cumsum(input_mask, axis=1) - 1

    if model.integration_type == "expert":
        (_, _, suffix_out), _ = model.PaliGemma.llm(
            [mem_tokens, prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, None, adarms_cond],
        )
    elif model.integration_type == "modulation":
        mem_seq, mem_mask, _, _, _ = model.embed_memory(observation)
        (_, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            mem_seq=[None, mem_seq],
            mem_mask=[None, mem_mask],
        )
    else:
        (_, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    loss_per_sample = jnp.mean(jnp.square(v_t - u_t), axis=(-2, -1))  # (B,)
    return loss_per_sample


def redundancy_penalty(
    selected_indices: jnp.ndarray,   # (K,) int32
    threshold: int = 2,
) -> jnp.ndarray:
    """Penalise near-duplicate timestamp selections.

    Counts pairs of selected frames whose absolute timestep difference
    is < threshold, normalised by the number of selections.
    """
    K = selected_indices.shape[0]
    # (K, K) pairwise distances
    diffs = jnp.abs(selected_indices[:, None] - selected_indices[None, :])
    # upper triangle only (exclude diagonal)
    mask = jnp.triu(jnp.ones((K, K), dtype=jnp.bool_), k=1)
    close_pairs = jnp.sum(jnp.where(mask, diffs < threshold, 0))
    return close_pairs / K


def compute_reward(
    config: SelectorConfig,
    fm_loss_uniform: jnp.ndarray,    # (B,)
    fm_loss_selector: jnp.ndarray,   # (B,)
    selected_indices: jnp.ndarray,   # (B, K) int32
) -> jnp.ndarray:
    """Per-sample reward: how much better the selector is than uniform.

    reward = FM_loss_uniform - FM_loss_selector - coef * redundancy
    """
    red_pen = jax.vmap(
        lambda idx: redundancy_penalty(idx, config.redundancy_time_threshold)
    )(selected_indices)

    reward = (
        fm_loss_uniform
        - fm_loss_selector
        - config.redundancy_penalty_coef * red_pen
    )
    return reward
