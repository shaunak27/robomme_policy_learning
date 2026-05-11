"""Evaluate QKFS selector on exec-only tasks (no video demo).

For each episode, simulates oracle execution: at sampled eval timesteps,
runs QKFS to select 32 frames from history and compares against what the
sampling rules specify as relevant source subtasks.

Metrics:
- Source precision: fraction of selected frames in a rule-specified source subtask
- Source recall: fraction of source subtasks with >= 1 selected frame
- Source weight correlation: Spearman rank correlation between P_target weights
  and the empirical fraction of selections per subtask
- Temporal spread: std dev of selected frame indices (higher = more diversity)

Baselines:
- Uniform random: select 32 frames uniformly at random from history
- Recency: select the 32 most recent frames

Visualization:
- Per-episode timeline: segment bands + QKFS selections + rule sources

Usage:
    python scripts/eval_qkfs.py \
        --checkpoint runs/ckpts/qkfs/step_5000 \
        --dataset_path data/robomme_preprocessed_data \
        --episodes_per_task 10 \
        --eval_points_per_episode 5 \
        --output_dir runs/eval_qkfs
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from mme_vla_suite.qkfs.inference import load_qkfs, select_frames_qkfs
from mme_vla_suite.shared.sampling_rules import get_sampling_sources, _is_completed

logger = logging.getLogger(__name__)

NO_DEMO_TASKS = [
    "BinFill", "ButtonUnmask", "ButtonUnmaskSwap",
    "PickHighlight", "PickXtimes", "StopCube", "SwingXtimes",
]

ALL_TASKS = NO_DEMO_TASKS + [
    "InsertPeg", "MoveCube", "PatternLock", "RouteStick",
    "VideoPlaceButton", "VideoPlaceOrder", "VideoRepick",
    "VideoUnmask", "VideoUnmaskSwap",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Tasks to evaluate. Default: all no-demo tasks.")
    parser.add_argument("--all_tasks", action="store_true",
                        help="Evaluate on all 16 tasks (including video-demo tasks)")
    parser.add_argument("--episodes_per_task", type=int, default=10)
    parser.add_argument("--eval_points_per_episode", type=int, default=5,
                        help="Number of eval timesteps per episode")
    parser.add_argument("--output_dir", type=str, default="runs/eval_qkfs")
    parser.add_argument("--episode_range", type=str, default=None,
                        help="Episode range to evaluate, e.g. '80-99'. Default: all episodes.")
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def frame_to_segment(frame_idx, segments):
    """Return segment dict for the segment containing frame_idx, or None."""
    for seg in segments:
        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
            return seg
    return None


def get_source_seg_indices(task_name, segments, current_seg, step_idx, task_goal=""):
    """Get rule-specified source segment indices for current_seg at step_idx."""
    source_map = get_sampling_sources(task_name, segments, task_goal=task_goal)
    sources = source_map.get(current_seg["idx"], [])
    # Filter to only past segments
    sources = [
        idx for idx in sources
        if any(s["idx"] == idx and s["start_frame"] < step_idx for s in segments)
    ]
    return sources


def compute_metrics_at_timestep(
    selected_indices, segments, source_seg_indices, num_past,
):
    """Compute metrics for a single eval timestep."""
    if len(selected_indices) == 0 or len(source_seg_indices) == 0:
        return None

    # Which segment does each selected frame belong to?
    selected_segs = []
    for fi in selected_indices:
        seg = frame_to_segment(fi, segments)
        if seg is not None:
            selected_segs.append(seg["idx"])
        else:
            selected_segs.append(-1)

    selected_segs = np.array(selected_segs)
    source_set = set(source_seg_indices)

    # Source precision: fraction of selected frames in a source subtask
    in_source = sum(1 for s in selected_segs if s in source_set)
    precision = in_source / len(selected_segs)

    # Source recall: fraction of source subtasks with >= 1 selected frame
    covered = set(s for s in selected_segs if s in source_set)
    recall = len(covered) / len(source_set)

    # Per-subtask selection fractions vs equal-weight P_target
    all_exec_segs = [s for s in segments if s["phase"] == "exec" and not _is_completed(s["label"])]
    n_segs = len(all_exec_segs)
    empirical_weights = np.zeros(n_segs)
    target_weights = np.zeros(n_segs)

    seg_idx_to_pos = {s["idx"]: i for i, s in enumerate(all_exec_segs)}
    for si in selected_segs:
        if si in seg_idx_to_pos:
            empirical_weights[seg_idx_to_pos[si]] += 1
    if empirical_weights.sum() > 0:
        empirical_weights /= empirical_weights.sum()

    for src_idx in source_seg_indices:
        if src_idx in seg_idx_to_pos:
            target_weights[seg_idx_to_pos[src_idx]] = 1.0 / len(source_seg_indices)

    # Spearman correlation (only meaningful with >= 3 segments)
    if n_segs >= 3:
        corr, _ = scipy_stats.spearmanr(empirical_weights, target_weights)
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = float(precision > 0.5)  # fallback

    # Temporal spread: normalized std of selected indices
    temporal_std = np.std(selected_indices) / max(num_past, 1)

    return {
        "precision": precision,
        "recall": recall,
        "correlation": corr,
        "temporal_std": temporal_std,
        "n_selected": len(selected_indices),
        "n_sources": len(source_seg_indices),
        "n_past": num_past,
    }


def baseline_uniform(num_past, k, rng):
    """Select k frames uniformly at random from 0..num_past-1."""
    if num_past <= k:
        return np.arange(num_past, dtype=np.int32)
    return np.sort(rng.choice(num_past, k, replace=False).astype(np.int32))


def baseline_recency(num_past, k):
    """Select the k most recent frames."""
    if num_past <= k:
        return np.arange(num_past, dtype=np.int32)
    return np.arange(num_past - k, num_past, dtype=np.int32)


def visualize_episode(
    task_name, epis_idx, segments, eval_results, output_dir,
):
    """Create a timeline visualization for one episode."""
    fig, axes = plt.subplots(
        len(eval_results), 1,
        figsize=(14, 2.5 * len(eval_results)),
        squeeze=False,
    )

    # Color palette for segments
    exec_segs = [s for s in segments if s["phase"] == "exec" and not _is_completed(s["label"])]
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(exec_segs), 1)))
    seg_colors = {s["idx"]: colors[i] for i, s in enumerate(exec_segs)}

    # "completed" segment in gray
    for s in segments:
        if _is_completed(s.get("label", "")):
            seg_colors[s["idx"]] = (0.85, 0.85, 0.85, 1.0)

    total_frames = segments[-1]["end_frame"] + 1

    for row, er in enumerate(eval_results):
        ax = axes[row, 0]
        step_idx = er["step_idx"]
        source_segs = er["source_seg_indices"]
        source_set = set(source_segs)

        # Draw segment bands
        for seg in segments:
            color = seg_colors.get(seg["idx"], (0.9, 0.9, 0.9, 1.0))
            is_source = seg["idx"] in source_set
            ax.axvspan(seg["start_frame"], seg["end_frame"],
                       alpha=0.5 if is_source else 0.15, color=color)
            # Label segment
            mid = (seg["start_frame"] + seg["end_frame"]) / 2
            label_text = seg["label"][:20]
            if len(seg["label"]) > 20:
                label_text += "..."
            ax.text(mid, 0.92, label_text, ha="center", va="top",
                    fontsize=6, transform=ax.get_xaxis_transform(),
                    fontweight="bold" if is_source else "normal")

        # Draw current timestep
        ax.axvline(step_idx, color="red", linewidth=2, linestyle="--", label="current t")

        # Draw QKFS selections
        qkfs_sel = er["qkfs_selected"]
        ax.scatter(qkfs_sel, np.full(len(qkfs_sel), 0.5), marker="|", s=100,
                   color="blue", linewidths=1.5, zorder=5, label="QKFS")

        # Draw uniform baseline
        uni_sel = er["uniform_selected"]
        ax.scatter(uni_sel, np.full(len(uni_sel), 0.3), marker="|", s=60,
                   color="gray", linewidths=0.8, zorder=4, label="uniform")

        # Draw recency baseline
        rec_sel = er["recency_selected"]
        ax.scatter(rec_sel, np.full(len(rec_sel), 0.1), marker="|", s=60,
                   color="orange", linewidths=0.8, zorder=4, label="recency")

        ax.set_xlim(0, total_frames)
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        # Metrics text
        m = er["qkfs_metrics"]
        if m is not None:
            ax.set_ylabel(
                f"P={m['precision']:.0%} R={m['recall']:.0%}",
                fontsize=8, rotation=0, labelpad=50, va="center",
            )

        if row == 0:
            ax.set_title(f"{task_name} ep{epis_idx}", fontsize=10, fontweight="bold")
        if row == len(eval_results) - 1:
            ax.set_xlabel("Frame index")
            ax.legend(loc="lower right", fontsize=7, ncol=3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{task_name}_ep{epis_idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.no_visualize:
        args.visualize = False

    if args.all_tasks:
        tasks = ALL_TASKS
    else:
        tasks = args.tasks or NO_DEMO_TASKS
    logger.info("Evaluating on tasks: %s", tasks)

    # Parse episode range filter
    episode_ids = None
    if args.episode_range:
        start, end = args.episode_range.split("-")
        episode_ids = set(range(int(start), int(end) + 1))
        logger.info("Episode filter: %d episodes (%s)", len(episode_ids), args.episode_range)

    # Load QKFS model
    logger.info("Loading checkpoint from %s", args.checkpoint)
    model, config = load_qkfs(args.checkpoint)
    k = config.num_frames_to_select  # 32

    per_task_dir = os.path.join(args.dataset_path, "per_task")
    vis_dir = os.path.join(args.output_dir, "visualizations")

    # Aggregate metrics
    all_metrics = {
        "qkfs": {"precision": [], "recall": [], "correlation": [], "temporal_std": []},
        "uniform": {"precision": [], "recall": [], "correlation": [], "temporal_std": []},
        "recency": {"precision": [], "recall": [], "correlation": [], "temporal_std": []},
    }
    per_task_metrics = {}

    for task_name in tasks:
        task_path = os.path.join(per_task_dir, task_name)
        feature_dir = Path(task_path) / "features"

        ep_dirs = sorted(feature_dir.glob("episode_*"),
                         key=lambda p: int(p.name.split("_")[1]))

        # Filter to specified episode range
        if episode_ids is not None:
            ep_dirs = [d for d in ep_dirs if int(d.name.split("_")[1]) in episode_ids]

        if not ep_dirs:
            logger.info("Task %s: no episodes in range, skipping", task_name)
            continue

        n_eps = min(args.episodes_per_task, len(ep_dirs))
        ep_sample = rng.choice(len(ep_dirs), n_eps, replace=False)
        ep_sample.sort()

        task_metrics = {
            "qkfs": {"precision": [], "recall": [], "correlation": [], "temporal_std": []},
            "uniform": {"precision": [], "recall": [], "correlation": [], "temporal_std": []},
            "recency": {"precision": [], "recall": [], "correlation": [], "temporal_std": []},
        }

        logger.info("Task: %s (%d episodes)", task_name, n_eps)

        for ep_i in ep_sample:
            ep_dir = ep_dirs[ep_i]
            epis_idx = int(ep_dir.name.split("_")[1])

            # Load episode data
            instr_path = ep_dir / "instruction_emb.npy"
            if instr_path.exists():
                ep_instruction_emb = np.load(instr_path).astype(np.float32)
            else:
                ep_instruction_emb = np.zeros(config.instruction_emb_dim, dtype=np.float32)

            global_embs = np.load(ep_dir / "global_emb.npy")  # (T, 2048)
            state_path = ep_dir / "detail_state.npy"
            if state_path.exists():
                states = np.load(state_path)  # (T, 8)
            else:
                states = np.zeros((global_embs.shape[0], config.proprio_dim), dtype=np.float32)

            segments = json.load(open(ep_dir / "segments.json"))
            T = global_embs.shape[0]

            # Find valid eval points: timesteps in exec segments that have source subtasks
            eval_candidates = []
            for seg in segments:
                if seg["phase"] != "exec" or _is_completed(seg["label"]):
                    continue
                sources = get_source_seg_indices(task_name, segments, seg, seg["end_frame"])
                if not sources:
                    continue
                # Sample from the middle-to-end of the segment (more history)
                start = max(seg["start_frame"] + 5, seg["start_frame"])
                end = seg["end_frame"]
                if start < end:
                    eval_candidates.extend(range(start, end + 1))

            if not eval_candidates:
                continue

            n_points = min(args.eval_points_per_episode, len(eval_candidates))
            eval_timesteps = sorted(rng.choice(eval_candidates, n_points, replace=False))

            episode_eval_results = []

            for step_idx in eval_timesteps:
                current_seg = frame_to_segment(step_idx, segments)
                if current_seg is None:
                    continue

                source_seg_indices = get_source_seg_indices(
                    task_name, segments, current_seg, step_idx
                )
                if not source_seg_indices:
                    continue

                num_past = step_idx  # frames 0..step_idx-1

                # --- QKFS selection ---
                instruction_emb = ep_instruction_emb
                current_obs_emb = global_embs[min(step_idx, T - 1)]
                current_proprio = states[min(step_idx, T - 1)]

                qkfs_selected = select_frames_qkfs(
                    model, config,
                    instruction_emb=instruction_emb,
                    current_obs_emb=current_obs_emb,
                    all_past_embs=global_embs[:num_past],
                    all_past_proprios=states[:num_past],
                    current_proprio=current_proprio,
                )

                # --- Baselines ---
                uniform_selected = baseline_uniform(num_past, k, rng)
                recency_selected = baseline_recency(num_past, k)

                # --- Metrics ---
                qkfs_m = compute_metrics_at_timestep(
                    qkfs_selected, segments, source_seg_indices, num_past)
                uniform_m = compute_metrics_at_timestep(
                    uniform_selected, segments, source_seg_indices, num_past)
                recency_m = compute_metrics_at_timestep(
                    recency_selected, segments, source_seg_indices, num_past)

                for method, m in [("qkfs", qkfs_m), ("uniform", uniform_m), ("recency", recency_m)]:
                    if m is not None:
                        for key in ["precision", "recall", "correlation", "temporal_std"]:
                            task_metrics[method][key].append(m[key])
                            all_metrics[method][key].append(m[key])

                episode_eval_results.append({
                    "step_idx": step_idx,
                    "source_seg_indices": source_seg_indices,
                    "current_seg": current_seg,
                    "qkfs_selected": qkfs_selected,
                    "uniform_selected": uniform_selected,
                    "recency_selected": recency_selected,
                    "qkfs_metrics": qkfs_m,
                    "uniform_metrics": uniform_m,
                    "recency_metrics": recency_m,
                })

            # Visualize
            if args.visualize and episode_eval_results:
                vis_path = visualize_episode(
                    task_name, epis_idx, segments, episode_eval_results,
                    os.path.join(vis_dir, task_name),
                )
                logger.info("  Saved visualization: %s", vis_path)

        # Per-task summary
        per_task_metrics[task_name] = task_metrics
        if task_metrics["qkfs"]["precision"]:
            logger.info("  %s summary (n=%d):", task_name, len(task_metrics["qkfs"]["precision"]))
            for method in ["qkfs", "uniform", "recency"]:
                m = task_metrics[method]
                logger.info("    %-8s  prec=%.1f%%  recall=%.1f%%  corr=%.2f  t_spread=%.3f",
                            method,
                            100 * np.mean(m["precision"]),
                            100 * np.mean(m["recall"]),
                            np.mean(m["correlation"]),
                            np.mean(m["temporal_std"]))

    # --- Overall summary ---
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    header = f"{'Method':<10} {'Precision':>10} {'Recall':>10} {'Correlation':>12} {'T-Spread':>10}  {'N':>5}"
    print(header)
    print("-" * len(header))
    for method in ["qkfs", "uniform", "recency"]:
        m = all_metrics[method]
        if m["precision"]:
            print(f"{method:<10} {np.mean(m['precision']):>9.1%} {np.mean(m['recall']):>9.1%} "
                  f"{np.mean(m['correlation']):>11.3f} {np.mean(m['temporal_std']):>9.3f}  {len(m['precision']):>5}")

    print("\n" + "=" * 80)
    print("PER-TASK BREAKDOWN")
    print("=" * 80)

    for task_name in tasks:
        m = per_task_metrics.get(task_name, {}).get("qkfs", {})
        if not m or not m.get("precision"):
            continue
        um = per_task_metrics[task_name]["uniform"]
        print(f"\n{task_name}:")
        print(f"  QKFS:    prec={np.mean(m['precision']):.1%}  recall={np.mean(m['recall']):.1%}  "
              f"corr={np.mean(m['correlation']):.3f}  spread={np.mean(m['temporal_std']):.3f}")
        print(f"  Uniform: prec={np.mean(um['precision']):.1%}  recall={np.mean(um['recall']):.1%}  "
              f"corr={np.mean(um['correlation']):.3f}  spread={np.mean(um['temporal_std']):.3f}")
        rm = per_task_metrics[task_name]["recency"]
        print(f"  Recency: prec={np.mean(rm['precision']):.1%}  recall={np.mean(rm['recall']):.1%}  "
              f"corr={np.mean(rm['correlation']):.3f}  spread={np.mean(rm['temporal_std']):.3f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "checkpoint": args.checkpoint,
        "tasks": tasks,
        "overall": {
            method: {k: float(np.mean(v)) for k, v in metrics.items() if v}
            for method, metrics in all_metrics.items()
        },
        "per_task": {
            task: {
                method: {k: float(np.mean(v)) for k, v in metrics.items() if v}
                for method, metrics in task_data.items()
            }
            for task, task_data in per_task_metrics.items()
        },
    }
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    if args.visualize:
        logger.info("Visualizations saved to %s", vis_dir)


if __name__ == "__main__":
    main()
