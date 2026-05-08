"""PPO training loop for the frame selector.

With gamma=0 this is effectively a contextual bandit: each (episode, step) is
an independent decision and the advantage simplifies to  A = r - V(s).
"""

import dataclasses
import functools
import logging
from typing import Any

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from mme_vla_suite.selector.config import SelectorConfig
from mme_vla_suite.selector.model import FrameSelector

logger = logging.getLogger(__name__)


# ---- Rollout buffer (plain dict-of-arrays) --------------------------------

@dataclasses.dataclass
class RolloutBatch:
    """Collected experience for one PPO update cycle."""
    # Per-sample query inputs
    front_view_emb: jnp.ndarray     # (B, front_dim)
    instruction_emb: jnp.ndarray    # (B, front_dim)
    proprio: jnp.ndarray            # (B, proprio_dim)
    progress: jnp.ndarray | None    # (B, 1)

    # Per-sample candidate info (padded to max_candidates)
    cand_embs: jnp.ndarray          # (B, N, cand_dim)
    cand_times: jnp.ndarray         # (B, N, 1)
    cand_mask: jnp.ndarray          # (B, N) bool

    # Rollout outputs
    selected_indices: jnp.ndarray   # (B, K) int32
    old_log_probs: jnp.ndarray      # (B,)  total log-prob under collection policy
    old_values: jnp.ndarray         # (B,)  V(s) at collection time
    rewards: jnp.ndarray            # (B,)
    advantages: jnp.ndarray         # (B,)  = rewards - old_values
    returns: jnp.ndarray            # (B,)  = rewards (gamma=0)


def compute_advantages(rewards: jnp.ndarray, values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """With gamma=0, advantage = reward - V(s), return = reward."""
    advantages = rewards - values
    returns = rewards
    return advantages, returns


# ---- PPO loss (batched) ------------------------------------------------------

def ppo_loss_batch(
    selector: FrameSelector,
    config: SelectorConfig,
    batch: RolloutBatch,
):
    """Mean PPO loss over a batch.  Returns (scalar_loss, info_dict)."""
    # Batched evaluate — no vmap needed, evaluate_batched handles it
    new_log_probs, entropies, values = selector.evaluate_batched(
        batch.front_view_emb,
        batch.instruction_emb,
        batch.proprio,
        batch.progress,
        batch.cand_embs,
        batch.cand_times,
        batch.cand_mask,
        batch.selected_indices,
    )  # all (B,)

    # Policy loss (clipped surrogate)
    log_ratio = new_log_probs - batch.old_log_probs
    # Clamp log-ratio to prevent exp overflow/NaN
    log_ratio = jnp.clip(log_ratio, -10.0, 10.0)
    ratio = jnp.exp(log_ratio)
    surr1 = ratio * batch.advantages
    surr2 = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * batch.advantages
    policy_loss = -jnp.minimum(surr1, surr2)

    # Value loss
    value_loss = jnp.square(values - batch.returns)

    # Entropy bonus (clamp to prevent negative entropy from numerical issues)
    entropies = jnp.maximum(entropies, 0.0)
    entropy_loss = -entropies

    total = (
        policy_loss
        + config.value_loss_coef * value_loss
        + config.entropy_coef * entropy_loss
    )

    mean_loss = jnp.mean(total)
    info = {
        "policy_loss": jnp.mean(policy_loss),
        "value_loss": jnp.mean(value_loss),
        "entropy": jnp.mean(entropies),
        "ratio": jnp.mean(ratio),
        "approx_kl": jnp.mean(-log_ratio),  # uses clamped log-ratio
    }
    return mean_loss, info


# ---- Optimizer + update step -----------------------------------------------

def create_selector_optimizer(config: SelectorConfig):
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr),
    )


def ppo_update_step(
    selector: FrameSelector,
    opt_state: optax.OptState,
    tx: optax.GradientTransformation,
    config: SelectorConfig,
    batch: RolloutBatch,
):
    """One gradient step of PPO on the selector.

    Returns (new_opt_state, loss, info).
    """
    trainable = nnx.state(selector, nnx.Param)

    def loss_fn(model):
        return ppo_loss_batch(model, config, batch)

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(selector)

    grad_params = nnx.state(selector, nnx.Param)
    # grads is structured the same as trainable params
    updates, new_opt_state = tx.update(grads, opt_state, grad_params)
    new_params = optax.apply_updates(grad_params, updates)
    nnx.update(selector, new_params)

    info["grad_norm"] = optax.global_norm(grads)
    return new_opt_state, loss, info


def run_ppo_epochs(
    selector: FrameSelector,
    opt_state: optax.OptState,
    tx: optax.GradientTransformation,
    config: SelectorConfig,
    rollout: RolloutBatch,
    rng: jnp.ndarray,
) -> tuple[optax.OptState, dict[str, Any]]:
    """Run config.ppo_epochs passes over the rollout data.

    Shuffles and splits into minibatches each epoch.
    Returns (new_opt_state, aggregated_info).
    """
    B = rollout.rewards.shape[0]
    mb = config.ppo_minibatch_size
    all_infos = []

    for epoch in range(config.ppo_epochs):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, B)

        for start in range(0, B, mb):
            end = min(start + mb, B)
            idx = perm[start:end]

            mini = RolloutBatch(
                front_view_emb=rollout.front_view_emb[idx],
                instruction_emb=rollout.instruction_emb[idx],
                proprio=rollout.proprio[idx],
                progress=rollout.progress[idx] if rollout.progress is not None else None,
                cand_embs=rollout.cand_embs[idx],
                cand_times=rollout.cand_times[idx],
                cand_mask=rollout.cand_mask[idx],
                selected_indices=rollout.selected_indices[idx],
                old_log_probs=rollout.old_log_probs[idx],
                old_values=rollout.old_values[idx],
                rewards=rollout.rewards[idx],
                advantages=rollout.advantages[idx],
                returns=rollout.returns[idx],
            )

            opt_state, loss, info = ppo_update_step(
                selector, opt_state, tx, config, mini
            )
            all_infos.append(info)

    # Aggregate info across all minibatches and epochs
    agg = jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs)), *all_infos)
    return opt_state, agg
