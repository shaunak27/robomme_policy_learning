"""Train the RL frame selector with PPO.

Stage 1: Freeze the entire VLA, train only selector + critic.
Stage 2 (optional): Unfreeze modulation params and continue.

Usage:
    python scripts/train_selector.py \
        --vla_checkpoint_path runs/ckpts/mme_vla_suite/framesamp_modul/80000/params \
        --dataset_path /path/to/preprocessed_dataset \
        --exp_name rl_selector_v1
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import functools
import logging
import os
import pickle
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax
import wandb
import tqdm_loggable.auto as tqdm

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.transforms as _transforms
from openpi.training import sharding as _sharding

from mme_vla_suite.selector.config import SelectorConfig
from mme_vla_suite.selector.model import FrameSelector
from mme_vla_suite.selector.ppo import (
    RolloutBatch,
    compute_advantages,
    create_selector_optimizer,
    run_ppo_epochs,
)
from mme_vla_suite.selector.reward import (
    compute_fm_loss_deterministic,
    compute_reward,
)
from mme_vla_suite.selector.dataset import SelectorDataset
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0, HistoryPi0Config
from mme_vla_suite.models.integration.history_observation import (
    HistAugObservation,
    preprocess_observation,
)
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.policies.robomme_policy import RoboMMEInputs, RoboMMEOutputs
from mme_vla_suite.training.config import (
    RoboMMEDataConfig,
    DataConfig,
    ModelTransformFactory,
    PaligemmaTokenizer,
)

from openpi.models import model as _model

logger = logging.getLogger(__name__)


def parse_args() -> tuple[SelectorConfig, list[str] | None]:
    parser = argparse.ArgumentParser(description="Train RL frame selector")
    parser.add_argument("--vla_checkpoint_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to preprocessed dataset (output of build_robomme_dataset.py)")
    parser.add_argument("--exp_name", type=str, default="rl_selector_v1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_train_steps", type=int, default=50_000)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--redundancy_penalty_coef", type=float, default=0.01)
    parser.add_argument("--neighbor_suppression_radius", type=int, default=0)
    parser.add_argument("--max_candidates", type=int, default=512)
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rl-frame-selector")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Task(s) to train on (e.g. BinFill MoveCube). "
                             "Default: all tasks. Steps auto-scaled to data proportion.")
    # Stage 2
    parser.add_argument("--stage2_unfreeze_modulation", action="store_true")
    parser.add_argument("--stage2_unfreeze_action_expert_lora", action="store_true")

    args = parser.parse_args()

    # Resolve task filtering: use per_task subdirs when tasks are specified
    dataset_path = args.dataset_path
    num_train_steps = args.num_train_steps
    tasks = args.tasks

    if tasks:
        import json
        # Compute total samples across all tasks for proportional scaling
        per_task_dir = os.path.join(dataset_path, "per_task")
        all_tasks_stats = json.load(open(os.path.join(dataset_path, "meta", "stats.json")))
        total_all = all_tasks_stats.get("execution_samples", all_tasks_stats["total_samples"])

        selected_total = 0
        for t in tasks:
            task_stats_path = os.path.join(per_task_dir, t, "meta", "stats.json")
            if not os.path.exists(task_stats_path):
                available = sorted(os.listdir(per_task_dir))
                raise ValueError(f"Task '{t}' not found. Available: {available}")
            ts = json.load(open(task_stats_path))
            selected_total += ts.get("execution_samples", ts["total_samples"])

        # Scale steps proportionally to data fraction
        data_fraction = selected_total / total_all
        num_train_steps = max(1000, int(args.num_train_steps * data_fraction))
        logger.info("Tasks: %s (%d samples, %.1f%% of total, %d steps)",
                     tasks, selected_total, data_fraction * 100, num_train_steps)

    # Scale steps inversely with batch size (reference: bs=32)
    REFERENCE_BS = 32
    num_train_steps = max(1000, int(num_train_steps * REFERENCE_BS / args.batch_size))
    logger.info("Batch size %d (ref %d) -> num_train_steps=%d",
                args.batch_size, REFERENCE_BS, num_train_steps)

    # Derive save_interval as 1/5 of total steps
    save_interval = max(100, num_train_steps // 5)
    logger.info("save_interval=%d (num_train_steps // 5)", save_interval)

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
        stage2_unfreeze_modulation=args.stage2_unfreeze_modulation,
        stage2_unfreeze_action_expert_lora=args.stage2_unfreeze_action_expert_lora,
    ), tasks


# ---------------------------------------------------------------------------
# VLA loading
# ---------------------------------------------------------------------------

def load_frozen_vla(checkpoint_path: str, history_config_name: str) -> HistoryPi0:
    """Load a HistoryPi0 model from checkpoint and freeze all params."""
    config = HistoryPi0Config(
        pi05=True,
        action_horizon=20,
        use_history=True,
        history_config=history_config_name,
        discrete_state_input=False,
    )
    rng = jax.random.key(0)
    model = config.create(rng)

    # Load checkpoint weights
    params_shape = nnx.state(model).to_pure_dict()
    loaded = _model.restore_params(checkpoint_path, restore_type=np.ndarray)

    # Merge loaded into model
    graphdef, state = nnx.split(model)
    from openpi.training.weight_loaders import _merge_params
    merged = _merge_params(loaded, params_shape, missing_regex=".*")
    state.replace_by_pure_dict(merged)
    model = nnx.merge(graphdef, state)
    model.eval()

    logger.info("Frozen VLA loaded from %s", checkpoint_path)
    return model


# ---------------------------------------------------------------------------
# Data transforms (reuse VLA's pipeline for observation construction)
# ---------------------------------------------------------------------------

def build_vla_transforms(history_config_name: str):
    """Build the transform pipeline that converts raw samples into
    HistAugObservation-compatible dicts.
    """
    model_config = HistoryPi0Config(
        pi05=True,
        action_horizon=20,
        use_history=True,
        history_config=history_config_name,
        discrete_state_input=False,
    )

    data_config_factory = RoboMMEDataConfig(
        repo_id="robomme",
        base_config=DataConfig(prompt_from_task=True),
    )
    # We need a dummy assets_dirs; norm stats are loaded from the asset path.
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
# Instruction embedding (frozen VLA's LLM embed layer)
# ---------------------------------------------------------------------------

def compute_instruction_embedding(
    vla: HistoryPi0,
    tokenized_prompt: jnp.ndarray,       # (B, max_token_len)
    tokenized_prompt_mask: jnp.ndarray,   # (B, max_token_len)
) -> jnp.ndarray:
    """Pool the frozen LLM token embeddings to get instruction vectors (B, 2048)."""
    emb = vla.PaliGemma.llm(tokenized_prompt, method="embed")  # (B, L, width)
    mask = tokenized_prompt_mask[:, :, None]  # (B, L, 1)
    pooled = jnp.sum(emb * mask, axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)  # (B, width)
    return pooled


# ---------------------------------------------------------------------------
# Build VLA observation from raw data + memory embeddings
# ---------------------------------------------------------------------------

def make_vla_observation(
    transformed_data: dict,
    static_image_emb: np.ndarray,
    static_pos_emb: np.ndarray,
    static_state_emb: np.ndarray,
    static_mask: np.ndarray,
) -> tuple[HistAugObservation, jnp.ndarray]:
    """Construct a batched HistAugObservation from already-batched data.

    Returns (observation, actions).
    """
    data = dict(transformed_data)
    data["static_image_emb"] = static_image_emb
    data["static_pos_emb"] = static_pos_emb
    data["static_state_emb"] = static_state_emb
    data["static_mask"] = static_mask

    # Data is already batched (B, ...), just convert to jnp
    obs_data = jax.tree.map(lambda x: jnp.asarray(x) if x is not None else x, data)
    actions = obs_data.pop("actions")
    observation = HistAugObservation.from_dict(obs_data)
    return observation, actions


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

class MultiTaskSelectorDataset:
    """Concatenates multiple SelectorDatasets, routing gather calls to the right one."""

    def __init__(self, sub_datasets: list[SelectorDataset]):
        self._datasets = sub_datasets
        self._cumulative = []
        total = 0
        for ds in sub_datasets:
            total += len(ds)
            self._cumulative.append(total)
        # Map epis_idx -> sub-dataset index (each sub-dataset has local epis_idx starting at 0)
        # We store offset so gather_and_pack_selected can find the right sub-dataset
        self._epis_to_ds: dict[tuple[int, int], SelectorDataset] = {}

    def __len__(self):
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, idx: int) -> dict:
        # Find which sub-dataset this index belongs to
        ds_idx, local_idx = self._locate(idx)
        sample = self._datasets[ds_idx][local_idx]
        # Tag with ds_idx so gather_and_pack_selected knows which sub-dataset to use
        sample["_ds_idx"] = ds_idx
        return sample

    def _locate(self, idx: int) -> tuple[int, int]:
        for i, cum in enumerate(self._cumulative):
            if idx < cum:
                local = idx - (self._cumulative[i - 1] if i > 0 else 0)
                return i, local
        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

    def gather_and_pack_selected(self, epis_idx: int, selected_indices, ds_idx: int = 0):
        return self._datasets[ds_idx].gather_and_pack_selected(epis_idx, selected_indices)


def _shard_pytree(pytree, sharding):
    """Place a pytree of numpy/jax arrays onto devices with the given sharding."""
    def _to_device(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            return jax.make_array_from_process_local_data(sharding, x)
        return x
    return jax.tree.map(_to_device, pytree)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config, tasks = parse_args()
    logger.info("Running on: %s", platform.node())
    logger.info("SelectorConfig: %s", config)

    num_devices = jax.device_count()
    logger.info("JAX devices: %d (%s)", num_devices, jax.devices())
    if config.batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by "
            f"the number of devices {num_devices}."
        )

    rng = jax.random.key(42) #INFO: this is the seed

    # ---- Device mesh (batch-parallel, no FSDP — selector is tiny) ----
    mesh = _sharding.make_mesh(num_fsdp_devices=1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(_sharding.BATCH_AXIS) #INFO: this is the sharding spec for data parallelism across the batch dimension
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec() #INFO: this is the sharding spec for replicated parameters (no sharding)
    )
    logger.info("Mesh: %s", mesh)

    # ---- W&B ----
    if config.wandb_enabled:
        wandb.init(
            name=config.exp_name,
            project=config.wandb_project,
            config=dataclasses.asdict(config),
        )
    else:
        wandb.init(mode="disabled")

    # ---- Load frozen VLA ----
    logger.info("Loading frozen VLA...")
    vla = load_frozen_vla(config.vla_checkpoint_path, config.history_config)
    history_cfg = get_history_config(config.history_config)

    # Replicate VLA state across all devices before module_jit captures it
    graphdef, vla_state = nnx.split(vla)
    vla_state = jax.device_put(vla_state, replicated_sharding)
    vla = nnx.merge(graphdef, vla_state)
    logger.info("VLA state replicated across %d devices", num_devices)

    # ---- Build transforms ----
    vla_transform, data_config = build_vla_transforms(config.history_config)

    # ---- Dataset ----
    norm_stats = data_config.norm_stats.get("state") if data_config.norm_stats else None
    use_quantiles = data_config.use_quantile_norm if data_config else False

    if tasks:
        per_task_dir = os.path.join(config.dataset_path, "per_task")
        sub_datasets = []
        for t in tasks:
            task_path = os.path.join(per_task_dir, t)
            ds = SelectorDataset(task_path, config, norm_stats=norm_stats, use_quantiles=use_quantiles)
            logger.info("Task %s: %d samples", t, len(ds))
            sub_datasets.append(ds)
        dataset = MultiTaskSelectorDataset(sub_datasets)
    else:
        dataset = SelectorDataset(
            config.dataset_path, config, norm_stats=norm_stats, use_quantiles=use_quantiles,
        )
    logger.info("Dataset size: %d", len(dataset))

    # ---- Torch data loader (batched) ----
    from torch.utils.data import DataLoader as TorchDataLoader

    def _numpy_collate(batch):
        """Stack list of dicts into a batched dict of numpy arrays."""
        keys = batch[0].keys()
        result = {}
        for k in keys:
            vals = [b[k] for b in batch]
            if isinstance(vals[0], np.ndarray):
                result[k] = np.stack(vals, axis=0)
            elif isinstance(vals[0], str):
                result[k] = vals
            elif isinstance(vals[0], (int, float)):
                result[k] = np.array(vals)
            else:
                result[k] = vals
        return result

    loader = TorchDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # avoid os.fork() deadlock with JAX
        collate_fn=_numpy_collate,
        drop_last=True,
    )
    data_iter = iter(loader)

    # ---- Selector model ----
    selector_rng, rng = jax.random.split(rng)
    selector = FrameSelector(config, rngs=nnx.Rngs(selector_rng))
    tx = create_selector_optimizer(config)
    opt_state = tx.init(nnx.state(selector, nnx.Param))
    logger.info(
        "Selector param count: %d",
        sum(x.size for x in jax.tree.leaves(nnx.state(selector, nnx.Param))),
    )

    # ---- JIT-compiled VLA FM loss (via module_jit to handle Flax nn.Module params) ----
    def _fm_loss_method(self, observation, actions, noise, timestep):
        """Bound-method-style wrapper so module_jit can freeze VLA state."""
        observation = preprocess_observation(None, observation, train=False)
        return compute_fm_loss_deterministic(self, observation, actions, noise, timestep)

    import types
    vla._fm_loss = types.MethodType(_fm_loss_method, vla)
    jit_fm_loss = nnx_utils.module_jit(vla._fm_loss)

    # ---- JIT-compiled instruction embedding ----
    def _instr_emb_method(self, tokenized_prompt, tokenized_prompt_mask):
        return compute_instruction_embedding(self, tokenized_prompt, tokenized_prompt_mask)

    vla._instr_emb = types.MethodType(_instr_emb_method, vla)
    jit_instr_emb = nnx_utils.module_jit(vla._instr_emb)

    # ---- Training loop ----
    os.makedirs(config.selector_checkpoint_dir, exist_ok=True)

    pbar = tqdm.tqdm(range(config.num_train_steps), dynamic_ncols=True)

    for step in pbar:
        t_step_start = time.time()
        rng, rollout_rng = jax.random.split(rng)

        # ---- Load a batch ----
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        t_data = time.time()

        B = config.batch_size

        # ---- Apply VLA transforms per sample (tokenization etc.) ----
        def _transform_one(b):
            sample_b = {k: (v[b] if isinstance(v, np.ndarray) else v[b]) for k, v in batch.items()}
            return vla_transform(sample_b)
        with ThreadPoolExecutor(max_workers=min(B, 16)) as pool:
            vla_data_list = list(pool.map(_transform_one, range(B)))

        # Stack VLA data into batched arrays (handles nested dicts like "image")
        def _stack_values(vals):
            if vals[0] is None:
                return None
            elif isinstance(vals[0], dict):
                return {k: _stack_values([v[k] for v in vals]) for k in vals[0]}
            elif isinstance(vals[0], np.ndarray):
                return np.stack(vals, axis=0)
            elif isinstance(vals[0], jnp.ndarray):
                return jnp.stack(vals, axis=0)
            elif isinstance(vals[0], (bool, int, float, np.integer, np.floating, np.bool_)):
                return np.array(vals)
            else:
                return vals

        vla_batch = {}
        for k in vla_data_list[0]:
            vals = [vla_data_list[b][k] for b in range(B)]
            vla_batch[k] = _stack_values(vals)
        t_transform = time.time()

        # ---- Batched instruction embeddings from frozen VLA ----
        # Shard VLA inputs across devices for data parallelism
        instr_embs = jit_instr_emb(
            jax.make_array_from_process_local_data(
                data_sharding, np.asarray(vla_batch["tokenized_prompt"])),
            jax.make_array_from_process_local_data(
                data_sharding, np.asarray(vla_batch["tokenized_prompt_mask"])),
        )  # (B, 2048) — sharded across devices
        # Gather back to single device for selector (which is replicated)
        instr_embs = jax.device_get(instr_embs)
        instr_embs = jnp.asarray(instr_embs)
        t_instr = time.time()

        # ---- Batched selector forward (parallel Gumbel-top-K) ----
        # Selector is tiny — runs replicated, no sharding needed
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

        # ---- Load detailed embeddings for selected frames (I/O, per sample) ----
        sel_indices_np = np.array(jax.device_get(sel_out.selected_indices))  # (B, K)
        def _gather_one(b):
            epis_idx = int(batch["epis_idx"][b])
            ds_idx = int(batch["_ds_idx"][b]) if "_ds_idx" in batch else 0
            return dataset.gather_and_pack_selected(epis_idx, sel_indices_np[b], ds_idx=ds_idx)
        with ThreadPoolExecutor(max_workers=min(B, 16)) as pool:
            gather_results = list(pool.map(_gather_one, range(B)))
        sel_imgs = np.stack([r[0] for r in gather_results])
        sel_poss = np.stack([r[1] for r in gather_results])
        sel_states = np.stack([r[2] for r in gather_results])
        sel_masks = np.stack([r[3] for r in gather_results])
        t_gather = time.time()

        # ---- Build batched VLA observations (selector + uniform) ----
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

        # ---- Shard observations across devices for data-parallel VLA forward ----
        obs_sel = _shard_pytree(obs_sel, data_sharding)
        actions_sel = jax.make_array_from_process_local_data(data_sharding, np.asarray(actions_sel))
        obs_uni = _shard_pytree(obs_uni, data_sharding)
        actions_uni = jax.make_array_from_process_local_data(data_sharding, np.asarray(actions_uni))

        # ---- Compute FM losses in one batched call each ----
        rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # Generate noise/timestep as numpy, then shard (actions_sel.shape is global)
        action_shape = actions_sel.shape  # (B, action_horizon, action_dim)
        noise_np = np.array(jax.random.normal(noise_rng, action_shape))
        noise = jax.make_array_from_process_local_data(data_sharding, noise_np)
        timestep_np = np.array(
            jax.random.beta(time_rng, 1.5, 1, action_shape[:-2]) * 0.999 + 0.001
        )
        timestep = jax.make_array_from_process_local_data(data_sharding, timestep_np)

        fm_loss_sel = jit_fm_loss(obs_sel, actions_sel, noise, timestep)  # (B,) sharded
        fm_loss_uni = jit_fm_loss(obs_uni, actions_uni, noise, timestep)  # (B,) sharded
        # Gather FM losses back for reward/PPO (which run replicated)
        fm_loss_sel = jnp.asarray(jax.device_get(fm_loss_sel))
        fm_loss_uni = jnp.asarray(jax.device_get(fm_loss_uni))
        t_fm = time.time()

        # ---- Reward (batched) ----
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
            selector, opt_state, tx, config, rollout, ppo_rng
        )
        t_ppo = time.time()

        # ---- Logging ----
        mean_reward = float(jnp.mean(rewards))
        mean_fm_sel = float(jnp.mean(fm_loss_sel))
        mean_fm_uni = float(jnp.mean(fm_loss_uni))
        mean_entropy = float(jnp.mean(sel_out.entropy))

        if step < 3:
            logger.info(
                "Step %d timing: data=%.1fs transform=%.1fs instr=%.1fs "
                "selector=%.1fs gather=%.1fs fm_loss=%.1fs ppo=%.1fs total=%.1fs",
                step,
                t_data - t_step_start, t_transform - t_data, t_instr - t_transform,
                t_selector - t_instr, t_gather - t_selector, t_fm - t_gather,
                t_ppo - t_fm, t_ppo - t_step_start,
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

            # Log selected timestamp statistics
            sel_idx_all = jax.device_get(rollout.selected_indices)  # (B, K)
            ages = []
            diversities = []
            for i in range(sel_idx_all.shape[0]):
                valid = sel_idx_all[i]
                ages.append(np.mean(valid))
                diversities.append(np.std(valid))
            wandb.log({
                "avg_selected_age": np.mean(ages),
                "temporal_diversity": np.mean(diversities),
            }, step=step)

        # ---- Save checkpoint ----
        if step > 0 and step % config.save_interval == 0:
            ckpt_path = os.path.join(config.selector_checkpoint_dir, f"step_{step}")
            os.makedirs(ckpt_path, exist_ok=True)
            selector_state = jax.device_get(nnx.state(selector))
            with open(os.path.join(ckpt_path, "selector_params.pkl"), "wb") as f:
                pickle.dump(selector_state, f)
            with open(os.path.join(ckpt_path, "config.pkl"), "wb") as f:
                pickle.dump(config, f)
            logger.info("Saved checkpoint at step %d to %s", step, ckpt_path)

    # ---- Save final checkpoint ----
    ckpt_path = os.path.join(config.selector_checkpoint_dir, "final")
    os.makedirs(ckpt_path, exist_ok=True)
    selector_state = jax.device_get(nnx.state(selector))
    with open(os.path.join(ckpt_path, "selector_params.pkl"), "wb") as f:
        pickle.dump(selector_state, f)
    with open(os.path.join(ckpt_path, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    logger.info("Saved final checkpoint to %s", ckpt_path)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
