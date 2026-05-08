"""Train the RL frame selector with PPO — v2 with consolidated arrays.

Same proven JIT architecture as train_selector.py (separate module_jit calls
for VLA FM loss, run_ppo_epochs for PPO).  The speedup comes from the v2
dataset, which loads consolidated per-episode arrays instead of hundreds of
individual files, and pre-packs uniform baselines + all candidate embeddings
in __getitem__.

Selected-frame gather happens on CPU via numpy indexing into the batch's
pre-loaded candidate tensors — zero file I/O in the training loop.

Prerequisites:
    1. Run scripts/consolidate_embeddings.py to create per-episode arrays.
    2. VLA checkpoint at --vla_checkpoint_path.

Usage (single node):
    python scripts/train_selector_v2.py \
        --vla_checkpoint_path runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999/params \
        --dataset_path data/robomme_preprocessed_data \
        --batch_size 64

Usage (multi-node via SLURM):
    See slurm_train_selector_v2.sh
"""

import argparse
import dataclasses
import logging
import os
import pickle
import time
import types

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax
import wandb
import tqdm_loggable.auto as tqdm

import openpi.shared.nnx_utils as nnx_utils
import openpi.transforms as _transforms
from openpi.training import sharding as _sharding
from openpi.models import model as _model

from mme_vla_suite.selector.config import SelectorConfig
from mme_vla_suite.selector.model import FrameSelector
from mme_vla_suite.selector.ppo import (
    RolloutBatch,
    compute_advantages,
    create_selector_optimizer,
    run_ppo_epochs,
)
from mme_vla_suite.selector.reward import compute_fm_loss_deterministic, compute_reward
from mme_vla_suite.selector.dataset_v2 import SelectorDatasetV2, MultiTaskDatasetV2
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0, HistoryPi0Config
from mme_vla_suite.models.integration.history_observation import (
    HistAugObservation,
    preprocess_observation,
)
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.training.config import (
    RoboMMEDataConfig,
    DataConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> tuple[SelectorConfig, list[str] | None]:
    parser = argparse.ArgumentParser(description="Train RL frame selector (v2)")
    parser.add_argument("--vla_checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="rl_selector_v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_train_steps", type=int, default=50_000)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--redundancy_penalty_coef", type=float, default=0.01)
    parser.add_argument("--neighbor_suppression_radius", type=int, default=0)
    parser.add_argument("--max_candidates", type=int, default=128)
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rl-frame-selector")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Task(s) to train on. Default: all tasks.")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    num_train_steps = args.num_train_steps
    tasks = args.tasks

    if tasks:
        import json
        per_task_dir = os.path.join(dataset_path, "per_task")
        all_stats = json.load(open(os.path.join(dataset_path, "meta", "stats.json")))
        total_all = all_stats.get("execution_samples", all_stats["total_samples"])

        selected_total = 0
        for t in tasks:
            task_stats = os.path.join(per_task_dir, t, "meta", "stats.json")
            if not os.path.exists(task_stats):
                available = sorted(os.listdir(per_task_dir))
                raise ValueError(f"Task '{t}' not found. Available: {available}")
            ts = json.load(open(task_stats))
            selected_total += ts.get("execution_samples", ts["total_samples"])

        data_fraction = selected_total / total_all
        num_train_steps = max(1000, int(args.num_train_steps * data_fraction))
        logger.info("Tasks: %s (%d samples, %.1f%%, %d steps)",
                     tasks, selected_total, data_fraction * 100, num_train_steps)

    # Scale steps with batch size (reference: bs=32)
    num_train_steps = max(1000, int(num_train_steps * 32 / args.batch_size))
    save_interval = max(100, num_train_steps // 5)

    return SelectorConfig(
        vla_checkpoint_path=args.vla_checkpoint_path,
        dataset_path=dataset_path,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        lr=args.lr,
        num_train_steps=num_train_steps,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        redundancy_penalty_coef=args.redundancy_penalty_coef,
        neighbor_suppression_radius=args.neighbor_suppression_radius,
        max_candidates=args.max_candidates,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        log_interval=args.log_interval,
        save_interval=save_interval,
    ), tasks


# ---------------------------------------------------------------------------
# VLA loading & transforms (same as v1)
# ---------------------------------------------------------------------------

def load_frozen_vla(checkpoint_path: str, history_config_name: str) -> HistoryPi0:
    config = HistoryPi0Config(
        pi05=True,
        action_horizon=20,
        use_history=True,
        history_config=history_config_name,
        discrete_state_input=False,
    )
    model = config.create(jax.random.key(0))

    params_shape = nnx.state(model).to_pure_dict()
    loaded = _model.restore_params(checkpoint_path, restore_type=np.ndarray)

    graphdef, state = nnx.split(model)
    from openpi.training.weight_loaders import _merge_params
    merged = _merge_params(loaded, params_shape, missing_regex=".*")
    state.replace_by_pure_dict(merged)
    model = nnx.merge(graphdef, state)
    model.eval()
    return model


def build_vla_transforms(history_config_name: str):
    model_config = HistoryPi0Config(
        pi05=True, action_horizon=20,
        use_history=True, history_config=history_config_name,
        discrete_state_input=False,
    )
    data_config_factory = RoboMMEDataConfig(
        repo_id="robomme",
        base_config=DataConfig(prompt_from_task=True),
    )
    import pathlib
    assets_dirs = pathlib.Path("runs/assets/mme_vla_suite").resolve()
    data_config = data_config_factory.create(assets_dirs, model_config)

    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats or {}, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    return _transforms.compose(transforms_list), data_config


# ---------------------------------------------------------------------------
# Numpy collation for DataLoader
# ---------------------------------------------------------------------------

def numpy_collate(batch: list[dict]) -> dict:
    """Stack list of dicts into a batched dict of numpy arrays."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], np.ndarray):
            result[k] = np.stack(vals, axis=0)
        elif isinstance(vals[0], str):
            result[k] = vals
        elif isinstance(vals[0], (int, float, np.generic)):
            result[k] = np.array(vals)
        elif isinstance(vals[0], dict):
            nested = {}
            for kk in vals[0]:
                inner_vals = [v[kk] for v in vals]
                if isinstance(inner_vals[0], np.ndarray):
                    nested[kk] = np.stack(inner_vals, axis=0)
                elif isinstance(inner_vals[0], (np.generic, int, float, bool)):
                    nested[kk] = np.array(inner_vals)
                else:
                    nested[kk] = inner_vals
            result[k] = nested
        else:
            result[k] = vals
    return result


# ---------------------------------------------------------------------------
# CPU-side gather of selected frames from pre-loaded candidate tensors
# ---------------------------------------------------------------------------

def gather_selected_frames(
    batch: dict,
    selected_indices: np.ndarray,  # (B, K) int32
    config: SelectorConfig,
    norm_stats=None,
    use_quantiles: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gather selected frames from the batch's candidate detail tensors.

    Uses numpy fancy indexing — no file I/O, no JIT.  The candidate tensors
    were pre-loaded by SelectorDatasetV2.__getitem__.

    Returns (sel_img, sel_pos, sel_state, sel_mask) shaped for VLA memory:
        sel_img:   (B, budget, 2048) float32
        sel_pos:   (B, budget, 768)  float32
        sel_state: (B, budget, 8)    float32  (normalized)
        sel_mask:  (B, budget)       bool
    """
    B, K = selected_indices.shape
    token_per_image = config.token_per_image  # 16
    num_views = config.num_views              # 1
    max_frames = config.token_budget // (token_per_image * num_views)

    cand_img = batch["cand_detail_img"]      # (B, N, 16, 2048)
    cand_pos = batch["cand_detail_pos"]      # (B, N, 16, 768)
    cand_state = batch["cand_detail_state"]  # (B, N, 8)

    # Gather per sample: cand[b, selected_indices[b]] -> (B, K, ...)
    b_idx = np.arange(B)[:, None]  # (B, 1) for broadcasting
    sel_img = cand_img[b_idx, selected_indices]      # (B, K, 16, 2048)
    sel_pos = cand_pos[b_idx, selected_indices]      # (B, K, 16, 768)
    sel_state = cand_state[b_idx, selected_indices]  # (B, K, 8)
    sel_mask = np.ones((B, K), dtype=np.bool_)

    # Right-pad to max_frames if K < max_frames
    if K < max_frames:
        pad_frames = max_frames - K
        sel_img = np.pad(sel_img, ((0, 0), (0, pad_frames), (0, 0), (0, 0)))
        sel_pos = np.pad(sel_pos, ((0, 0), (0, pad_frames), (0, 0), (0, 0)))
        sel_state = np.pad(sel_state, ((0, 0), (0, pad_frames), (0, 0)))
        sel_mask = np.pad(sel_mask, ((0, 0), (0, pad_frames)), constant_values=False)

    # Reshape to flat token sequence: (B, max_frames, 16, dim) -> (B, budget, dim)
    sel_img = sel_img.reshape(B, -1, 2048)[:, :config.token_budget]
    sel_pos = sel_pos.reshape(B, -1, 768)[:, :config.token_budget]
    # State: repeat per token
    sel_state = np.repeat(sel_state, num_views * token_per_image, axis=1)[:, :config.token_budget]
    sel_mask = np.repeat(sel_mask, num_views * token_per_image, axis=1)[:, :config.token_budget]

    # Normalize state
    if norm_stats is not None:
        ns = norm_stats
        if use_quantiles:
            sel_state = (sel_state - ns.q01) / (ns.q99 - ns.q01 + 1e-6) * 2.0 - 1.0
        else:
            sel_state = (sel_state - ns.mean) / (ns.std + 1e-6)

    return sel_img.astype(np.float32), sel_pos.astype(np.float32), sel_state.astype(np.float32), sel_mask


# ---------------------------------------------------------------------------
# Build VLA observation (same pattern as v1)
# ---------------------------------------------------------------------------

def make_vla_observation(
    transformed_data: dict,
    static_image_emb: np.ndarray,
    static_pos_emb: np.ndarray,
    static_state_emb: np.ndarray,
    static_mask: np.ndarray,
) -> tuple[HistAugObservation, jnp.ndarray]:
    """Construct a batched HistAugObservation from already-batched data."""
    data = dict(transformed_data)
    data["static_image_emb"] = static_image_emb
    data["static_pos_emb"] = static_pos_emb
    data["static_state_emb"] = static_state_emb
    data["static_mask"] = static_mask

    obs_data = jax.tree.map(lambda x: jnp.asarray(x) if x is not None else x, data)
    actions = obs_data.pop("actions")
    observation = HistAugObservation.from_dict(obs_data)
    return observation, actions


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------

def shard_array(x, sharding):
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return jax.make_array_from_process_local_data(sharding, np.asarray(x))
    return x


def shard_pytree(pytree, sharding):
    return jax.tree.map(lambda x: shard_array(x, sharding), pytree)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Multi-node initialization (no-op for single node)
    if "SLURM_JOB_ID" in os.environ:
        coordinator = os.environ.get("SLURM_STEP_NODELIST", os.environ.get("SLURM_NODELIST", "localhost"))
        if "[" in coordinator:
            prefix = coordinator[:coordinator.index("[")]
            first = coordinator[coordinator.index("[") + 1:].split(",")[0].split("-")[0].rstrip("]")
            coordinator = prefix + first
        jax.distributed.initialize(
            coordinator_address=f"{coordinator}:1234",
            num_processes=int(os.environ.get("SLURM_NTASKS", 1)),
            process_id=int(os.environ.get("SLURM_PROCID", 0)),
        )

    config, tasks = parse_args()
    num_devices = jax.device_count()
    local_devices = jax.local_device_count()
    process_id = jax.process_index()
    num_processes = jax.process_count()

    logger.info("Process %d/%d | %d local devices | %d total devices",
                process_id, num_processes, local_devices, num_devices)

    if config.batch_size % num_devices != 0:
        raise ValueError(f"batch_size={config.batch_size} not divisible by {num_devices} devices")

    rng = jax.random.key(42)

    # ---- Device mesh (batch-parallel, no FSDP — selector is tiny) ----
    mesh = _sharding.make_mesh(num_fsdp_devices=1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(_sharding.BATCH_AXIS),
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(),
    )

    # ---- W&B (only on process 0) ----
    if process_id == 0 and config.wandb_enabled:
        wandb.init(name=config.exp_name, project=config.wandb_project,
                   config=dataclasses.asdict(config))
    else:
        wandb.init(mode="disabled")

    # ---- Load frozen VLA ----
    logger.info("Loading frozen VLA from %s ...", config.vla_checkpoint_path)
    vla = load_frozen_vla(config.vla_checkpoint_path, config.history_config)

    # Replicate VLA state across devices before module_jit captures it
    vla_graphdef, vla_state = nnx.split(vla)
    vla_state = jax.device_put(vla_state, replicated_sharding)
    vla = nnx.merge(vla_graphdef, vla_state)
    logger.info("VLA replicated across %d devices", num_devices)

    # ---- Build transforms ----
    vla_transform, data_config = build_vla_transforms(config.history_config)

    # ---- JIT-compiled VLA helpers (standalone module_jit, NEVER nested) ----

    # FM loss — same signature as v1
    def _fm_loss_method(self, observation, actions, noise, timestep):
        observation = preprocess_observation(None, observation, train=False)
        return compute_fm_loss_deterministic(self, observation, actions, noise, timestep)

    vla._fm_loss = types.MethodType(_fm_loss_method, vla)
    jit_fm_loss = nnx_utils.module_jit(vla._fm_loss)

    # Instruction embedding
    def _instr_emb_method(self, tokenized_prompt, tokenized_prompt_mask):
        emb = self.PaliGemma.llm(tokenized_prompt, method="embed")
        mask = tokenized_prompt_mask[:, :, None]
        return jnp.sum(emb * mask, axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)

    vla._instr_emb = types.MethodType(_instr_emb_method, vla)
    jit_instr_emb = nnx_utils.module_jit(vla._instr_emb)

    # ---- Dataset ----
    norm_stats = data_config.norm_stats.get("state") if data_config.norm_stats else None
    use_quantiles = data_config.use_quantile_norm if data_config else False

    if tasks:
        per_task_dir = os.path.join(config.dataset_path, "per_task")
        sub_datasets = []
        for t in tasks:
            ds = SelectorDatasetV2(
                os.path.join(per_task_dir, t), config,
                vla_transform=vla_transform, norm_stats=norm_stats, use_quantiles=use_quantiles,
            )
            logger.info("Task %s: %d samples", t, len(ds))
            sub_datasets.append(ds)
        dataset = MultiTaskDatasetV2(sub_datasets)
    else:
        dataset = SelectorDatasetV2(
            config.dataset_path, config,
            vla_transform=vla_transform, norm_stats=norm_stats, use_quantiles=use_quantiles,
        )
    logger.info("Dataset size: %d samples", len(dataset))

    # ---- DataLoader ----
    from torch.utils.data import DataLoader as TorchDataLoader

    loader = TorchDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # avoid os.fork() deadlock with JAX
        collate_fn=numpy_collate,
        drop_last=True,
    )
    data_iter = iter(loader)

    # ---- Selector model ----
    selector_rng, rng = jax.random.split(rng)
    selector = FrameSelector(config, rngs=nnx.Rngs(selector_rng))
    tx = create_selector_optimizer(config)
    opt_state = tx.init(nnx.state(selector, nnx.Param))

    param_count = sum(x.size for x in jax.tree.leaves(nnx.state(selector, nnx.Param)))
    logger.info("Selector params: %d", param_count)

    # ---- Training loop ----
    os.makedirs(config.selector_checkpoint_dir, exist_ok=True)
    pbar = tqdm.tqdm(range(config.num_train_steps), dynamic_ncols=True)

    for step in pbar:
        t0 = time.time()
        rng, rollout_rng = jax.random.split(rng)

        # ---- Load batch (VLA transforms already applied in dataset) ----
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        t_data = time.time()

        B = config.batch_size

        # ---- Instruction embeddings (standalone module_jit call) ----
        instr_embs = jit_instr_emb(
            shard_array(np.asarray(batch["tokenized_prompt"]), data_sharding),
            shard_array(np.asarray(batch["tokenized_prompt_mask"]), data_sharding),
        )
        # Gather back for selector (which runs replicated)
        instr_embs = jnp.asarray(jax.device_get(instr_embs))
        t_instr = time.time()

        # ---- Selector forward (replicated, tiny model) ----
        sel_out = selector.forward_batched(
            rollout_rng,
            jnp.asarray(batch["front_view_emb"]),
            instr_embs,
            jnp.asarray(batch["proprio"]),
            jnp.asarray(batch["progress"]),
            jnp.asarray(batch["cand_embs"]),
            jnp.asarray(batch["cand_times"]),
            jnp.asarray(batch["cand_mask"]),
        )
        t_selector = time.time()

        # ---- CPU gather of selected frames from pre-loaded candidates ----
        sel_indices_np = np.array(jax.device_get(sel_out.selected_indices))  # (B, K)
        sel_imgs, sel_poss, sel_states, sel_masks = gather_selected_frames(
            batch, sel_indices_np, config, norm_stats=norm_stats, use_quantiles=use_quantiles,
        )
        t_gather = time.time()

        # ---- Build VLA observations for selector and uniform ----
        # Extract VLA-compatible fields (exclude selector-specific and string keys)
        VLA_KEYS = {"image", "image_mask", "state",
                    "tokenized_prompt", "tokenized_prompt_mask",
                    "token_ar_mask", "token_loss_mask", "actions"}
        vla_batch = {k: batch[k] for k in VLA_KEYS
                     if k in batch and batch[k] is not None}

        obs_sel, actions_sel = make_vla_observation(
            vla_batch, sel_imgs, sel_poss, sel_states, sel_masks,
        )
        obs_uni, actions_uni = make_vla_observation(
            vla_batch,
            batch["uniform_static_image_emb"],
            batch["uniform_static_pos_emb"],
            batch["uniform_static_state_emb"],
            batch["uniform_static_mask"],
        )

        # ---- Shard observations for data-parallel VLA forward ----
        obs_sel = shard_pytree(obs_sel, data_sharding)
        actions_sel = shard_array(np.asarray(actions_sel), data_sharding)
        obs_uni = shard_pytree(obs_uni, data_sharding)
        actions_uni = shard_array(np.asarray(actions_uni), data_sharding)

        # ---- Shared noise & timestep (same for both, reduces variance) ----
        rng, noise_rng, time_rng = jax.random.split(rng, 3)
        action_shape = actions_sel.shape
        noise_np = np.array(jax.random.normal(noise_rng, action_shape))
        noise = shard_array(noise_np, data_sharding)
        timestep_np = np.array(
            jax.random.beta(time_rng, 1.5, 1.0, action_shape[:-2]) * 0.999 + 0.001
        )
        timestep = shard_array(timestep_np, data_sharding)

        # ---- VLA FM losses (standalone module_jit calls — NOT nested) ----
        fm_loss_sel = jit_fm_loss(obs_sel, actions_sel, noise, timestep)
        fm_loss_uni = jit_fm_loss(obs_uni, actions_uni, noise, timestep)

        # Gather back for reward/PPO (which run replicated)
        fm_loss_sel = jnp.asarray(jax.device_get(fm_loss_sel))
        fm_loss_uni = jnp.asarray(jax.device_get(fm_loss_uni))
        t_fm = time.time()

        # ---- Reward ----
        rewards = compute_reward(config, fm_loss_uni, fm_loss_sel, sel_out.selected_indices)

        # ---- PPO update ----
        advantages, returns = compute_advantages(rewards, sel_out.value)

        rollout = RolloutBatch(
            front_view_emb=jnp.asarray(batch["front_view_emb"]),
            instruction_emb=instr_embs,
            proprio=jnp.asarray(batch["proprio"]),
            progress=jnp.asarray(batch["progress"]),
            cand_embs=jnp.asarray(batch["cand_embs"]),
            cand_times=jnp.asarray(batch["cand_times"]),
            cand_mask=jnp.asarray(batch["cand_mask"]),
            selected_indices=sel_out.selected_indices,
            old_log_probs=sel_out.total_log_prob,
            old_values=sel_out.value,
            rewards=rewards,
            advantages=advantages,
            returns=returns,
        )

        rng, ppo_rng = jax.random.split(rng)
        opt_state, ppo_info = run_ppo_epochs(
            selector, opt_state, tx, config, rollout, ppo_rng,
        )
        t_ppo = time.time()

        # ---- Logging ----
        mean_reward = float(jnp.mean(rewards))
        mean_fm_sel = float(jnp.mean(fm_loss_sel))
        mean_fm_uni = float(jnp.mean(fm_loss_uni))
        mean_entropy = float(jnp.mean(sel_out.entropy))

        if step < 3:
            logger.info(
                "Step %d timing: data=%.1fs instr=%.1fs selector=%.1fs "
                "gather=%.1fs fm_loss=%.1fs ppo=%.1fs total=%.1fs",
                step,
                t_data - t0, t_instr - t_data, t_selector - t_instr,
                t_gather - t_selector, t_fm - t_gather,
                t_ppo - t_fm, t_ppo - t0,
            )

        if step % config.log_interval == 0:
            log_dict = {
                "mean_reward": mean_reward,
                "mean_fm_selector": mean_fm_sel,
                "mean_fm_uniform": mean_fm_uni,
                "mean_entropy": mean_entropy,
                "policy_loss": float(ppo_info.get("policy_loss", 0)),
                "value_loss": float(ppo_info.get("value_loss", 0)),
                "approx_kl": float(ppo_info.get("approx_kl", 0)),
                "grad_norm": float(ppo_info.get("grad_norm", 0)),
            }
            wandb.log(log_dict, step=step)
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
            pbar.write(f"Step {step}: {info_str}")

            # Selected-frame statistics
            sel_idx_all = jax.device_get(rollout.selected_indices)
            wandb.log({
                "avg_selected_age": float(np.mean(sel_idx_all)),
                "temporal_diversity": float(np.mean(np.std(sel_idx_all, axis=1))),
            }, step=step)

        # ---- Checkpoint ----
        if step > 0 and step % config.save_interval == 0 and process_id == 0:
            ckpt_path = os.path.join(config.selector_checkpoint_dir, f"step_{step}")
            os.makedirs(ckpt_path, exist_ok=True)
            sel_params = jax.device_get(nnx.state(selector))
            with open(os.path.join(ckpt_path, "selector_params.pkl"), "wb") as f:
                pickle.dump(sel_params, f)
            with open(os.path.join(ckpt_path, "config.pkl"), "wb") as f:
                pickle.dump(config, f)
            logger.info("Saved checkpoint step %d -> %s", step, ckpt_path)

    # ---- Final checkpoint ----
    if process_id == 0:
        ckpt_path = os.path.join(config.selector_checkpoint_dir, "final")
        os.makedirs(ckpt_path, exist_ok=True)
        sel_params = jax.device_get(nnx.state(selector))
        with open(os.path.join(ckpt_path, "selector_params.pkl"), "wb") as f:
            pickle.dump(sel_params, f)
        with open(os.path.join(ckpt_path, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        logger.info("Final checkpoint -> %s", ckpt_path)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
