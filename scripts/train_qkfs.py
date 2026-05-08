"""Train the Query-conditioned KeyFrame Selector (QKFS).

Supervised training with two KL-divergence losses:
- L_src: teaches which past subtasks to attend to
- L_frame: teaches which frames within relevant subtasks matter

No policy loss — pi0.5 is not finetuned.

Prerequisites:
    1. Preprocessed dataset with segments.json, density_keyframes.json,
       global_emb.npy per episode.
    2. TOPReward labels in --topreward_dir (optional but recommended).

Usage (single GPU):
    python scripts/train_qkfs.py \
        --dataset_path data/robomme_preprocessed_data \
        --topreward_dir data/topreward_full \
        --tasks BinFill PatternLock \
        --batch_size 64

Usage (SLURM):
    See slurm_train_qkfs.sh
"""

import argparse
import dataclasses
import logging
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import optax
import wandb
import tqdm_loggable.auto as tqdm

from openpi.training import sharding as _sharding

from mme_vla_suite.qkfs.config import QKFSConfig
from mme_vla_suite.qkfs.model import QKFS
from mme_vla_suite.qkfs.dataset import QKFSDataset, MultiTaskQKFSDataset
from mme_vla_suite.qkfs.targets import compute_qkfs_loss_jax

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------

def shard_array(x, sharding):
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return jax.make_array_from_process_local_data(sharding, np.asarray(x))
    return x


def shard_batch(batch: dict, sharding) -> dict:
    """Shard all arrays in a batch dict along the batch axis.

    Filters out non-array metadata (strings, Python lists) that would
    crash JAX's JIT tracing.
    """
    result = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            result[k] = shard_array(v, sharding)
        # Skip non-array metadata (prompt strings, etc.) — JAX can't trace them
    return result


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> tuple[QKFSConfig, list[str] | None, int]:
    parser = argparse.ArgumentParser(description="Train QKFS selector")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--topreward_dir", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="qkfs_v1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_train_steps", type=int, default=50_000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lambda_frame", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_query_layers", type=int, default=2)
    parser.add_argument("--num_selector_layers", type=int, default=2)
    parser.add_argument("--num_recent_frames", type=int, default=4)
    parser.add_argument("--max_candidates", type=int, default=512)
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="qkfs-selector")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Task(s) to train on. Default: all tasks.")
    args = parser.parse_args()

    num_train_steps = args.num_train_steps

    if args.tasks:
        import json
        per_task_dir = os.path.join(args.dataset_path, "per_task")
        all_stats = json.load(open(os.path.join(args.dataset_path, "meta", "stats.json")))
        total_all = all_stats.get("execution_samples", all_stats["total_samples"])

        selected_total = 0
        for t in args.tasks:
            task_stats = os.path.join(per_task_dir, t, "meta", "stats.json")
            if not os.path.exists(task_stats):
                available = sorted(os.listdir(per_task_dir))
                raise ValueError(f"Task '{t}' not found. Available: {available}")
            ts = json.load(open(task_stats))
            selected_total += ts.get("execution_samples", ts["total_samples"])

        data_fraction = selected_total / total_all
        num_train_steps = max(1000, int(args.num_train_steps * data_fraction))
        logger.info("Tasks: %s (%d samples, %.1f%%, %d steps)",
                     args.tasks, selected_total, data_fraction * 100, num_train_steps)

    save_interval = args.save_interval if args.save_interval != 5000 else max(100, num_train_steps // 5)

    # Scale LR with sqrt of batch size (reference: bs=64)
    lr = args.lr * (args.batch_size / 64) ** 0.5

    return QKFSConfig(
        dataset_path=args.dataset_path,
        topreward_dir=args.topreward_dir,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        lr=lr,
        weight_decay=args.weight_decay,
        num_train_steps=num_train_steps,
        warmup_steps=args.warmup_steps,
        lambda_frame=args.lambda_frame,
        sigma=args.sigma,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_query_layers=args.num_query_layers,
        num_selector_layers=args.num_selector_layers,
        num_recent_frames=args.num_recent_frames,
        max_candidates=args.max_candidates,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        log_interval=args.log_interval,
        save_interval=save_interval,
    ), args.tasks, args.num_workers


# ---------------------------------------------------------------------------
# Collation
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
        else:
            result[k] = vals
    return result


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _loss_fn(model: QKFS, batch: dict, num_segments: int,
             epsilon: float, lambda_frame: float):
    """Compute QKFS loss (called inside JIT)."""
    log_probs = model.forward(
        instruction_emb=batch["instruction_emb"],
        current_obs_emb=batch["current_obs_emb"],
        recent_embs=batch["recent_embs"],
        recent_mask=batch["recent_mask"],
        proprio=batch["proprio"],
        frame_embs=batch["cand_embs"],
        frame_proprios=batch["cand_proprios"],
        frame_times=batch["cand_times"],
        cand_mask=batch["cand_mask"],
        deterministic=False,
    )

    loss, info = compute_qkfs_loss_jax(
        log_probs=log_probs,
        candidate_frame_indices=batch["cand_frame_indices"],
        cand_mask=batch["cand_mask"],
        target_frame_dist=batch["target_frame_dist"],
        target_seg_assignment=batch["target_seg_assignment"],
        target_seg_weights=batch["target_seg_weights"],
        num_segments=num_segments,
        epsilon=epsilon,
        lambda_frame=lambda_frame,
    )

    # Scale loss to compute mean over valid samples only.
    # Invalid samples already contribute 0 to the mean, so we scale UP
    # by (B / count_valid) to correct the denominator.
    has_target = batch["has_target"].astype(jnp.float32)
    count_valid = jnp.maximum(has_target.sum(), 1.0)
    masked_loss = loss * (has_target.shape[0] / count_valid)

    return masked_loss, info


def make_train_step(tx, num_segments: int, epsilon: float, lambda_frame: float):
    """Create a JIT-compiled train step with constants captured as closures.

    `@nnx.jit` traces all non-NNX arguments, so constants like num_segments
    (needed concrete by jax.nn.one_hot), epsilon, lambda_frame, and the optax
    transform must live in the closure, not the function signature.
    """
    @nnx.jit
    def train_step(model: QKFS, opt_state, batch: dict):
        """One training step: forward, backward, update."""
        diff_state = nnx.DiffState(0, nnx.Param)

        def loss_wrapper(model):
            return _loss_fn(model, batch, num_segments, epsilon, lambda_frame)

        (loss, info), grads = nnx.value_and_grad(
            loss_wrapper, argnums=diff_state, has_aux=True
        )(model)

        grad_params = nnx.state(model, nnx.Param)
        updates, new_opt_state = tx.update(grads, opt_state, grad_params)
        new_params = optax.apply_updates(grad_params, updates)
        nnx.update(model, new_params)

        grad_norm = optax.global_norm(grads)
        info["grad_norm"] = grad_norm
        return new_opt_state, loss, info

    return train_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Multi-node initialization
    if "SLURM_JOB_ID" in os.environ:
        coordinator = os.environ.get("SLURM_STEP_NODELIST",
                                      os.environ.get("SLURM_NODELIST", "localhost"))
        if "[" in coordinator:
            prefix = coordinator[:coordinator.index("[")]
            first = coordinator[coordinator.index("[") + 1:].split(",")[0].split("-")[0].rstrip("]")
            coordinator = prefix + first
        jax.distributed.initialize(
            coordinator_address=f"{coordinator}:1234",
            num_processes=int(os.environ.get("SLURM_NTASKS", 1)),
            process_id=int(os.environ.get("SLURM_PROCID", 0)),
        )

    config, tasks, num_workers = parse_args()
    num_devices = jax.device_count()
    local_devices = jax.local_device_count()
    process_id = jax.process_index()
    num_processes = jax.process_count()

    logger.info("Process %d/%d | %d local devices | %d total devices",
                process_id, num_processes, local_devices, num_devices)

    if config.batch_size % num_devices != 0:
        raise ValueError(f"batch_size={config.batch_size} not divisible by {num_devices} devices")

    rng = jax.random.key(42)

    # ---- Device mesh (batch-parallel, no FSDP — QKFS is tiny) ----
    mesh = _sharding.make_mesh(num_fsdp_devices=1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(_sharding.BATCH_AXIS),
    )
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(),
    )

    # ---- W&B ----
    if process_id == 0 and config.wandb_enabled:
        wandb.init(name=config.exp_name, project=config.wandb_project,
                   config=dataclasses.asdict(config))
    else:
        wandb.init(mode="disabled")

    # ---- Dataset ----
    per_task_dir = os.path.join(config.dataset_path, "per_task")

    if tasks is None:
        tasks = sorted(os.listdir(per_task_dir))
        logger.info("Auto-detected tasks: %s", tasks)

    datasets = []
    for t in tasks:
        task_path = os.path.join(per_task_dir, t)
        if not os.path.isdir(task_path):
            logger.warning("Skipping non-directory: %s", task_path)
            continue
        ds = QKFSDataset(
            dataset_path=task_path,
            task_name=t,
            config=config,
            topreward_dir=config.topreward_dir,
        )
        logger.info("Task %s: %d samples", t, len(ds))
        datasets.append(ds)

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = MultiTaskQKFSDataset(datasets)
    logger.info("Total dataset size: %d samples", len(dataset))

    # ---- DataLoader ----
    from torch.utils.data import DataLoader as TorchDataLoader
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # avoid fork() deadlock with JAX threads

    loader = TorchDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    data_iter = iter(loader)

    # ---- Model ----
    model_rng, rng = jax.random.split(rng)
    model = QKFS(config, rngs=nnx.Rngs(model_rng))

    param_count = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    logger.info("QKFS params: %d (%.2f M)", param_count, param_count / 1e6)

    # Replicate model across devices
    graphdef, state = nnx.split(model)
    state = jax.device_put(state, replicated_sharding)
    model = nnx.merge(graphdef, state)
    logger.info("Model replicated across %d devices", num_devices)

    # ---- Optimizer ----
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.num_train_steps,
        end_value=config.lr * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(schedule, weight_decay=config.weight_decay),
    )
    opt_state = tx.init(nnx.state(model, nnx.Param))

    # ---- JIT-compiled training step ----
    train_step = make_train_step(
        tx, num_segments=config.max_segments,
        epsilon=config.epsilon, lambda_frame=config.lambda_frame,
    )

    # ---- Training loop ----
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    pbar = tqdm.tqdm(range(config.num_train_steps), dynamic_ncols=True)

    samples_per_epoch = len(dataset)
    samples_seen = 0
    epoch = 0

    logger.info("Training config: lr=%.2e (scaled), batch=%d, steps=%d, workers=%d",
                config.lr, config.batch_size, config.num_train_steps, num_workers)
    logger.info("Samples/epoch=%d, total_samples=%.1fM, epochs=%.1f",
                samples_per_epoch,
                config.num_train_steps * config.batch_size / 1e6,
                config.num_train_steps * config.batch_size / max(samples_per_epoch, 1))

    for step in pbar:
        t0 = time.time()

        # Load batch
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(loader)
            batch = next(data_iter)

        samples_seen += config.batch_size

        # Shard batch across devices
        jax_batch = shard_batch(batch, data_sharding)

        t_data = time.time()

        opt_state, loss, info = train_step(model, opt_state, jax_batch)

        t_step = time.time()

        # ---- Logging ----
        if step % config.log_interval == 0:
            # Fraction of samples with valid targets
            has_target = batch["has_target"] if isinstance(batch["has_target"], np.ndarray) else np.array(batch["has_target"])
            valid_frac = float(has_target.mean())

            log_dict = {
                "loss": float(loss),
                "L_src": float(info["L_src"]),
                "L_frame": float(info["L_frame"]),
                "grad_norm": float(info["grad_norm"]),
                "lr": float(schedule(step)),
                "epoch": epoch + samples_seen / max(samples_per_epoch, 1),
                "samples_seen": samples_seen,
                "valid_target_frac": valid_frac,
                "data_time": t_data - t0,
                "step_time": t_step - t_data,
            }
            wandb.log(log_dict, step=step)

            info_str = (f"loss={log_dict['loss']:.4f}, L_src={log_dict['L_src']:.4f}, "
                        f"L_frame={log_dict['L_frame']:.4f}, grad={log_dict['grad_norm']:.3f}, "
                        f"lr={log_dict['lr']:.1e}, ep={log_dict['epoch']:.1f}, "
                        f"valid={valid_frac:.0%}")
            pbar.write(f"Step {step}: {info_str}")

        # ---- Checkpoint ----
        if step > 0 and step % config.save_interval == 0 and process_id == 0:
            _save_checkpoint(model, config, step)

    # ---- Final checkpoint ----
    if process_id == 0:
        _save_checkpoint(model, config, "final")

    logger.info("Training complete.")


def _save_checkpoint(model: QKFS, config: QKFSConfig, step):
    ckpt_path = os.path.join(config.checkpoint_dir, f"step_{step}")
    os.makedirs(ckpt_path, exist_ok=True)
    params = jax.device_get(nnx.state(model))
    with open(os.path.join(ckpt_path, "qkfs_params.pkl"), "wb") as f:
        pickle.dump(params, f)
    with open(os.path.join(ckpt_path, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    logger.info("Saved checkpoint step %s -> %s", step, ckpt_path)


if __name__ == "__main__":
    main()
