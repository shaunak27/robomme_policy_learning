"""Fused train step for the RL frame selector.

Everything — selector forward, on-device frame gather, VLA FM loss (×2),
reward computation, and PPO update — runs inside a single jax.jit call.
No CPU round-trips mid-step.

The key enabler is that candidate detail embeddings (img/pos/state) are
pre-loaded in the batch tensor, so selected frames can be gathered on-device
via vmap indexing instead of file I/O.
"""

from __future__ import annotations

import functools
import math
from typing import Any, NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from mme_vla_suite.selector.config import SelectorConfig
from mme_vla_suite.selector.model import FrameSelector


# ---- Types ----------------------------------------------------------------

class TrainStepOutput(NamedTuple):
    """Everything the training loop needs for logging."""
    opt_state: Any
    selector_state: nnx.State
    rng: jnp.ndarray
    metrics: dict[str, jnp.ndarray]


class FusedBatch(NamedTuple):
    """Pre-sharded batch from the DataLoader. All arrays are global (B, ...)."""
    # Selector inputs
    cand_embs: jnp.ndarray        # (B, N, 2048)
    cand_times: jnp.ndarray       # (B, N, 1)
    cand_mask: jnp.ndarray        # (B, N) bool
    front_view_emb: jnp.ndarray   # (B, 2048)
    proprio: jnp.ndarray          # (B, 8)
    progress: jnp.ndarray         # (B, 1)
    # Candidate detail embeddings (for on-device gather)
    cand_detail_img: jnp.ndarray    # (B, N, 16, 2048)
    cand_detail_pos: jnp.ndarray    # (B, N, 16, 768)
    cand_detail_state: jnp.ndarray  # (B, N, 8)
    # Pre-packed uniform baseline
    uniform_img: jnp.ndarray      # (B, budget, 2048)
    uniform_pos: jnp.ndarray      # (B, budget, 768)
    uniform_state: jnp.ndarray    # (B, budget, 8)
    uniform_mask: jnp.ndarray     # (B, budget) bool
    # VLA observation fields (already transformed/tokenized)
    vla_obs: Any                  # pytree with images, tokenized_prompt, etc.
    actions: jnp.ndarray          # (B, action_horizon, action_dim)
    # Instruction embedding (from frozen VLA, computed outside JIT)
    instruction_emb: jnp.ndarray  # (B, 2048)


# ---- On-device frame packing ---------------------------------------------

def gather_and_pack_frames(
    selected_indices: jnp.ndarray,   # (B, K) int32
    cand_detail_img: jnp.ndarray,    # (B, N, 16, 2048)
    cand_detail_pos: jnp.ndarray,    # (B, N, 16, 768)
    cand_detail_state: jnp.ndarray,  # (B, N, 8)
    token_per_image: int,
    num_views: int,
    token_budget: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Gather selected frame embeddings and pack into VLA memory format.

    All on-device, no Python loops.

    Returns:
        static_image_emb:  (B, budget, 2048)
        static_pos_emb:    (B, budget, 768)
        static_state_emb:  (B, budget, 8)
        static_mask:       (B, budget) bool
    """
    K = selected_indices.shape[1]
    max_frames = token_budget // (token_per_image * num_views)

    # Gather selected frames: (B, K, 16, dim) via vmap
    def _gather_one(detail, indices):
        return detail[indices]  # (K, 16, dim) or (K, 8)

    sel_img = jax.vmap(_gather_one)(cand_detail_img, selected_indices)     # (B, K, 16, 2048)
    sel_pos = jax.vmap(_gather_one)(cand_detail_pos, selected_indices)     # (B, K, 16, 768)
    sel_state = jax.vmap(_gather_one)(cand_detail_state, selected_indices) # (B, K, 8)

    # Pad to max_frames if K < max_frames
    B = selected_indices.shape[0]
    if K < max_frames:
        pad_k = max_frames - K
        sel_img = jnp.pad(sel_img, ((0, 0), (0, pad_k), (0, 0), (0, 0)))
        sel_pos = jnp.pad(sel_pos, ((0, 0), (0, pad_k), (0, 0), (0, 0)))
        sel_state = jnp.pad(sel_state, ((0, 0), (0, pad_k), (0, 0)))
        frame_mask = jnp.concatenate([
            jnp.ones((B, K), dtype=jnp.bool_),
            jnp.zeros((B, pad_k), dtype=jnp.bool_),
        ], axis=1)
    elif K > max_frames:
        sel_img = sel_img[:, :max_frames]
        sel_pos = sel_pos[:, :max_frames]
        sel_state = sel_state[:, :max_frames]
        frame_mask = jnp.ones((B, max_frames), dtype=jnp.bool_)
    else:
        frame_mask = jnp.ones((B, max_frames), dtype=jnp.bool_)

    # Reshape to flat token sequence: (B, max_frames, 16, dim) → (B, budget, dim)
    static_img = sel_img.reshape(B, -1, sel_img.shape[-1])      # (B, budget, 2048)
    static_pos = sel_pos.reshape(B, -1, sel_pos.shape[-1])      # (B, budget, 768)

    # State: repeat each frame's state across its token_per_image tokens
    # (B, max_frames, 8) → (B, budget, 8)
    static_state = jnp.repeat(sel_state, num_views * token_per_image, axis=1)

    # Mask: repeat per-frame mask to per-token mask
    static_mask = jnp.repeat(frame_mask, num_views * token_per_image, axis=1)

    return static_img, static_pos, static_state, static_mask


# ---- Fused train step (to be JIT'd) --------------------------------------

def make_fused_train_step(
    config: SelectorConfig,
    vla_fm_loss_fn,
):
    """Create the fused train step function.

    Args:
        config: SelectorConfig
        vla_fm_loss_fn: Pure function (vla_state, obs, actions, noise, time) → (B,) loss.
            This should be the output of preparing the VLA for JIT (see train script).

    Returns:
        A function: (selector_gdef, selector_state, opt_state, tx_state, rng, batch)
                     → TrainStepOutput
    """
    clip_eps = config.clip_epsilon
    value_coef = config.value_loss_coef
    entropy_coef = config.entropy_coef
    ppo_epochs = config.ppo_epochs
    ppo_mb = config.ppo_minibatch_size
    redundancy_coef = config.redundancy_penalty_coef
    redundancy_thresh = config.redundancy_time_threshold
    token_per_image = config.token_per_image
    num_views = config.num_views
    token_budget = config.token_budget

    def fused_step(
        selector_graphdef: nnx.GraphDef,
        selector_state: nnx.State,
        opt_state: optax.OptState,
        tx: optax.GradientTransformation,
        rng: jnp.ndarray,
        batch: FusedBatch,
    ) -> TrainStepOutput:
        B = batch.actions.shape[0]
        rng, rollout_rng, noise_rng, time_rng, ppo_rng = jax.random.split(rng, 5)

        # ---- Reconstruct selector from state ----
        selector = nnx.merge(selector_graphdef, selector_state)

        # ---- Selector forward (Gumbel-top-K) ----
        sel_out = selector.forward_batched(
            rollout_rng,
            batch.front_view_emb,
            batch.instruction_emb,
            batch.proprio,
            batch.progress,
            batch.cand_embs,
            batch.cand_times,
            batch.cand_mask,
        )

        # ---- On-device gather & pack selected frames ----
        sel_img, sel_pos, sel_state_emb, sel_mask = gather_and_pack_frames(
            sel_out.selected_indices,
            batch.cand_detail_img,
            batch.cand_detail_pos,
            batch.cand_detail_state,
            token_per_image, num_views, token_budget,
        )

        # ---- Shared noise & timestep for fair comparison ----
        action_shape = batch.actions.shape
        noise = jax.random.normal(noise_rng, action_shape)
        timestep = jax.random.beta(time_rng, 1.5, 1.0, action_shape[:-2]) * 0.999 + 0.001

        # ---- VLA FM loss: selector frames ----
        fm_loss_sel = vla_fm_loss_fn(
            batch.vla_obs, batch.actions, noise, timestep,
            sel_img, sel_pos, sel_state_emb, sel_mask,
        )

        # ---- VLA FM loss: uniform baseline ----
        fm_loss_uni = vla_fm_loss_fn(
            batch.vla_obs, batch.actions, noise, timestep,
            batch.uniform_img, batch.uniform_pos,
            batch.uniform_state, batch.uniform_mask,
        )

        # ---- Reward: FM_uniform - FM_selector - redundancy penalty ----
        def _redundancy(indices):
            K = indices.shape[0]
            diffs = jnp.abs(indices[:, None] - indices[None, :])
            tri = jnp.triu(jnp.ones((K, K), dtype=jnp.bool_), k=1)
            return jnp.sum(jnp.where(tri, diffs < redundancy_thresh, 0)) / K

        red_pen = jax.vmap(_redundancy)(sel_out.selected_indices)
        rewards = fm_loss_uni - fm_loss_sel - redundancy_coef * red_pen

        # ---- Advantages (gamma=0 contextual bandit) ----
        advantages = rewards - sel_out.value
        returns = rewards

        # ---- PPO epochs (in-JIT loop) ----
        def ppo_body(carry, _epoch_idx):
            sel_state_c, opt_state_c, rng_c, agg_metrics = carry
            rng_c, perm_rng = jax.random.split(rng_c)
            perm = jax.random.permutation(perm_rng, B)

            # Minibatch loop via lax.scan
            num_mbs = B // ppo_mb

            def mb_body(inner_carry, mb_idx):
                sel_state_mb, opt_state_mb = inner_carry
                start = mb_idx * ppo_mb
                idx = jax.lax.dynamic_slice(perm, (start,), (ppo_mb,))

                # Slice minibatch
                mb_front = batch.front_view_emb[idx]
                mb_instr = batch.instruction_emb[idx]
                mb_proprio = batch.proprio[idx]
                mb_progress = batch.progress[idx]
                mb_cand_embs = batch.cand_embs[idx]
                mb_cand_times = batch.cand_times[idx]
                mb_cand_mask = batch.cand_mask[idx]
                mb_selected = sel_out.selected_indices[idx]
                mb_old_lp = sel_out.total_log_prob[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # PPO loss + gradient
                selector_mb = nnx.merge(selector_graphdef, sel_state_mb)

                def loss_fn(model):
                    new_lp, ent, val = model.evaluate_batched(
                        mb_front, mb_instr, mb_proprio, mb_progress,
                        mb_cand_embs, mb_cand_times, mb_cand_mask, mb_selected,
                    )
                    ent = jnp.maximum(ent, 0.0)

                    log_ratio = jnp.clip(new_lp - mb_old_lp, -10.0, 10.0)
                    ratio = jnp.exp(log_ratio)
                    surr1 = ratio * mb_advantages
                    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                    policy_loss = -jnp.minimum(surr1, surr2)
                    value_loss = jnp.square(val - mb_returns)
                    entropy_loss = -ent

                    total = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
                    info = {
                        "policy_loss": jnp.mean(policy_loss),
                        "value_loss": jnp.mean(value_loss),
                        "entropy": jnp.mean(ent),
                        "approx_kl": jnp.mean(-log_ratio),
                    }
                    return jnp.mean(total), info

                diff_state = nnx.DiffState(0, nnx.Param)
                (loss, info), grads = nnx.value_and_grad(
                    loss_fn, argnums=diff_state, has_aux=True
                )(selector_mb)

                grad_params = nnx.state(selector_mb, nnx.Param)
                updates, new_opt = tx.update(grads, opt_state_mb, grad_params)
                new_params = optax.apply_updates(grad_params, updates)
                nnx.update(selector_mb, new_params)

                info["grad_norm"] = optax.global_norm(grads)
                new_sel_state = nnx.state(selector_mb)
                return (new_sel_state, new_opt), info

            (sel_state_c, opt_state_c), epoch_infos = jax.lax.scan(
                mb_body, (sel_state_c, opt_state_c), jnp.arange(num_mbs),
            )

            # Aggregate metrics across minibatches
            epoch_agg = jax.tree.map(lambda x: jnp.mean(x, axis=0), epoch_infos)
            new_agg = jax.tree.map(lambda a, e: a + e, agg_metrics, epoch_agg)
            return (sel_state_c, opt_state_c, rng_c, new_agg), None

        init_metrics = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "approx_kl": 0.0, "grad_norm": 0.0,
        }
        (final_sel_state, final_opt_state, _, summed_metrics), _ = jax.lax.scan(
            ppo_body,
            (nnx.state(selector), opt_state, ppo_rng, init_metrics),
            jnp.arange(ppo_epochs),
        )

        # Average over epochs
        avg_metrics = jax.tree.map(lambda x: x / ppo_epochs, summed_metrics)

        # Add rollout-level metrics
        avg_metrics["mean_reward"] = jnp.mean(rewards)
        avg_metrics["mean_fm_selector"] = jnp.mean(fm_loss_sel)
        avg_metrics["mean_fm_uniform"] = jnp.mean(fm_loss_uni)
        avg_metrics["mean_entropy"] = jnp.mean(sel_out.entropy)
        avg_metrics["mean_value"] = jnp.mean(sel_out.value)

        return TrainStepOutput(
            opt_state=final_opt_state,
            selector_state=final_sel_state,
            rng=rng,
            metrics=avg_metrics,
        )

    return fused_step
