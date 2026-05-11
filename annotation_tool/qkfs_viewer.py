"""Flask viewer for QKFS eval results.

Displays interactive timelines showing QKFS frame selections vs baselines
overlaid on segment structure, plus summary metrics.

Usage:
    python annotation_tool/qkfs_viewer.py runs/eval_qkfs/step_20000
    python annotation_tool/qkfs_viewer.py runs/eval_qkfs/step_20000 --port 5050
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mme_vla_suite.qkfs.inference import load_qkfs, select_frames_qkfs
from mme_vla_suite.qkfs.targets import (
    build_target_distributions,
    load_topreward_intervals,
)
from mme_vla_suite.shared.sampling_rules import get_sampling_sources, _is_completed

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state set by main()
EVAL_DIR: Path = None
DATASET_DIR: Path = None
H5_DIR: Path = None
EVAL_RESULTS: dict = None
EVAL_EPISODE_IDS: set = None  # episodes marked as eval split
MODEL = None
CONFIG = None
THUMB_SIZE = 128  # thumbnail px

_h5_handles: dict[str, h5py.File] = {}


def _get_h5(task: str) -> h5py.File:
    if task not in _h5_handles or not _h5_handles[task].id.valid:
        path = H5_DIR / f"record_dataset_{task}.h5"
        _h5_handles[task] = h5py.File(str(path), "r")
    return _h5_handles[task]

NO_DEMO_TASKS = [
    "BinFill", "ButtonUnmask", "ButtonUnmaskSwap",
    "PickHighlight", "PickXtimes", "StopCube", "SwingXtimes",
]


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("qkfs_viewer.html")


# ---------------------------------------------------------------------------
# API — data
# ---------------------------------------------------------------------------

@app.route("/api/summary")
def api_summary():
    """Return overall + per-task metrics."""
    return jsonify(EVAL_RESULTS)


@app.route("/api/tasks")
def api_tasks():
    """List all tasks from the dataset directory."""
    per_task_dir = DATASET_DIR / "per_task"
    if not per_task_dir.exists():
        # Fallback to eval results
        return jsonify(list(EVAL_RESULTS.get("per_task", {}).keys()))
    tasks = sorted(
        d.name for d in per_task_dir.iterdir()
        if d.is_dir() and (d / "features").exists()
    )
    return jsonify(tasks)


@app.route("/api/<task>/episodes")
def api_episodes(task: str):
    """List episodes from the features directory, tagged with train/eval split."""
    feature_dir = DATASET_DIR / "per_task" / task / "features"
    if not feature_dir.exists():
        return jsonify([])
    eps = []
    for ep_dir in feature_dir.iterdir():
        if ep_dir.is_dir() and ep_dir.name.startswith("episode_"):
            try:
                ep_id = int(ep_dir.name.split("_")[1])
                split = "eval" if (EVAL_EPISODE_IDS and ep_id in EVAL_EPISODE_IDS) else "train"
                eps.append({"id": ep_id, "split": split})
            except ValueError:
                continue
    eps.sort(key=lambda x: x["id"])
    return jsonify(eps)


@app.route("/api/<task>/<int:ep>/segments")
def api_segments(task: str, ep: int):
    """Return segment annotations for an episode."""
    seg_file = DATASET_DIR / "per_task" / task / "features" / f"episode_{ep}" / "segments.json"
    if not seg_file.exists():
        return jsonify([])
    segments = json.load(open(seg_file))
    return jsonify(segments)


@app.route("/api/<task>/<int:ep>/episode_info")
def api_episode_info(task: str, ep: int):
    """Return episode metadata: total frames, segments, segment colors."""
    ep_dir = DATASET_DIR / "per_task" / task / "features" / f"episode_{ep}"
    seg_file = ep_dir / "segments.json"
    if not seg_file.exists():
        return jsonify({"error": "no segments"})

    segments = json.load(open(seg_file))
    emb_path = ep_dir / "global_emb.npy"
    total_frames = int(np.load(emb_path, mmap_mode="r").shape[0]) if emb_path.exists() else 0

    return jsonify({
        "total_frames": total_frames,
        "segments": segments,
        "task": task,
        "episode": ep,
    })


@app.route("/api/<task>/<int:ep>/run_qkfs")
def api_run_qkfs(task: str, ep: int):
    """Run QKFS at a specific timestep and return selections + metrics."""
    if MODEL is None:
        return jsonify({"error": "No model loaded — run on a GPU node with --checkpoint"}), 503

    step_idx = request.args.get("step", type=int)
    if step_idx is None:
        return jsonify({"error": "step parameter required"}), 400

    ep_dir = DATASET_DIR / "per_task" / task / "features" / f"episode_{ep}"
    global_embs = np.load(ep_dir / "global_emb.npy")
    state_path = ep_dir / "detail_state.npy"
    states = np.load(state_path) if state_path.exists() else np.zeros(
        (global_embs.shape[0], CONFIG.proprio_dim), dtype=np.float32)
    segments = json.load(open(ep_dir / "segments.json"))

    T = global_embs.shape[0]
    num_past = min(step_idx, T)

    if num_past == 0:
        return jsonify({"qkfs": [], "uniform": [], "recency": [], "sources": [], "metrics": {}})

    # Load per-episode instruction embedding
    instr_path = ep_dir / "instruction_emb.npy"
    if instr_path.exists():
        instruction_emb = np.load(instr_path).astype(np.float32)
    else:
        instruction_emb = np.zeros(CONFIG.instruction_emb_dim, dtype=np.float32)

    # QKFS selection
    qkfs_selected = select_frames_qkfs(
        MODEL, CONFIG,
        instruction_emb=instruction_emb,
        current_obs_emb=global_embs[min(step_idx, T - 1)],
        all_past_embs=global_embs[:num_past],
        all_past_proprios=states[:num_past],
        current_proprio=states[min(step_idx, T - 1)],
    ).tolist()

    # Baselines
    k = CONFIG.num_frames_to_select
    if num_past <= k:
        uniform_selected = list(range(num_past))
        recency_selected = list(range(num_past))
    else:
        rng = np.random.default_rng(step_idx)
        uniform_selected = sorted(rng.choice(num_past, k, replace=False).tolist())
        recency_selected = list(range(num_past - k, num_past))

    # Source subtasks from rules
    current_seg = None
    for seg in segments:
        if seg["start_frame"] <= step_idx <= seg["end_frame"]:
            current_seg = seg
            break

    source_seg_indices = []
    if current_seg and not _is_completed(current_seg.get("label", "")):
        source_map = get_sampling_sources(task, segments)
        sources = source_map.get(current_seg["idx"], [])
        source_seg_indices = [
            idx for idx in sources
            if any(s["idx"] == idx and s["start_frame"] < step_idx for s in segments)
        ]

    # Metrics
    source_set = set(source_seg_indices)

    def compute_precision(selected):
        if not selected or not source_set:
            return 0
        hits = 0
        for fi in selected:
            for seg in segments:
                if seg["start_frame"] <= fi <= seg["end_frame"] and seg["idx"] in source_set:
                    hits += 1
                    break
        return hits / len(selected)

    def compute_recall(selected):
        if not selected or not source_set:
            return 0
        covered = set()
        for fi in selected:
            for seg in segments:
                if seg["start_frame"] <= fi <= seg["end_frame"] and seg["idx"] in source_set:
                    covered.add(seg["idx"])
                    break
        return len(covered) / len(source_set)

    metrics = {
        "qkfs": {"precision": compute_precision(qkfs_selected), "recall": compute_recall(qkfs_selected)},
        "uniform": {"precision": compute_precision(uniform_selected), "recall": compute_recall(uniform_selected)},
        "recency": {"precision": compute_precision(recency_selected), "recall": compute_recall(recency_selected)},
    }

    return jsonify({
        "step_idx": step_idx,
        "qkfs": qkfs_selected,
        "uniform": uniform_selected,
        "recency": recency_selected,
        "source_seg_indices": source_seg_indices,
        "current_seg_idx": current_seg["idx"] if current_seg else None,
        "metrics": metrics,
        "num_past": num_past,
    })


# ---------------------------------------------------------------------------
# API — training targets
# ---------------------------------------------------------------------------

@app.route("/api/<task>/<int:ep>/targets")
def api_targets(task: str, ep: int):
    """Build training target distribution at a timestep and return:
    - target_frames: top-k frames from the target distribution (what QKFS should pick)
    - density_keyframes: per-segment density keyframe absolute indices
    - topreward_keyframes: per-segment TOPReward keyframes + success windows
    - T_j distributions per source segment
    - source_seg_indices, P_target, seg_budgets
    """
    step_idx = request.args.get("step", type=int)
    if step_idx is None:
        return jsonify({"error": "step parameter required"}), 400

    ep_dir = DATASET_DIR / "per_task" / task / "features" / f"episode_{ep}"
    segments = json.load(open(ep_dir / "segments.json"))

    # Load density keyframes
    dk_path = ep_dir / "density_keyframes.json"
    density_kfs = json.load(open(dk_path)) if dk_path.exists() else {}

    # Load topreward intervals
    topreward_dir = EVAL_RESULTS.get("topreward_dir", "data/topreward_full")
    tr_intervals = load_topreward_intervals(topreward_dir, task, ep, segments)

    # Get task goal from a sample file if available
    task_goal = ""
    import pickle as _pkl
    data_dir = ep_dir.parent.parent / "data"
    sample_0 = data_dir / "0.pkl"
    if sample_0.exists():
        try:
            d = _pkl.load(open(sample_0, "rb"))
            task_goal = d.get("prompt", "")
        except Exception:
            pass

    # Build target distributions
    targets = build_target_distributions(
        step_idx=step_idx,
        segments=segments,
        task_name=task,
        task_goal=task_goal,
        density_keyframes=density_kfs,
        topreward_intervals=tr_intervals,
        sigma=2.0,
        max_frames=32,
    )

    if targets is None:
        return jsonify({
            "step_idx": step_idx,
            "has_target": False,
            "target_frames": [],
            "density_keyframes_abs": {},
            "topreward_info": {},
            "source_seg_indices": [],
        })

    seg_lookup = {s["idx"]: s for s in segments}

    # Build the full target distribution over all past frames
    # and pick the top-k as the "ideal" selection
    num_past = step_idx
    full_dist = np.zeros(num_past, dtype=np.float64)

    for seg_idx in targets["source_seg_indices"]:
        seg = seg_lookup[seg_idx]
        T_j = targets["T_j"].get(seg_idx)
        if T_j is None or len(T_j) == 0:
            continue
        w_j = targets["P_target"][seg_idx]
        seg_start = seg["start_frame"]
        for i, val in enumerate(T_j):
            abs_frame = seg_start + i
            if 0 <= abs_frame < num_past:
                full_dist[abs_frame] += w_j * val

    # Normalize
    total = full_dist.sum()
    if total > 0:
        full_dist /= total

    # Top-k frames from target distribution
    k = 32
    if num_past <= k:
        target_frames = list(range(num_past))
    else:
        target_frames = sorted(np.argsort(full_dist)[-k:].tolist())

    # Absolute density keyframes per segment
    density_kfs_abs = {}
    for seg_idx in targets["source_seg_indices"]:
        seg = seg_lookup.get(seg_idx)
        if seg is None:
            continue
        raw = density_kfs.get(str(seg_idx), density_kfs.get(seg_idx, []))
        abs_kfs = [seg["start_frame"] + kf for kf in raw if seg["start_frame"] + kf < step_idx]
        if abs_kfs:
            density_kfs_abs[seg_idx] = abs_kfs

    # TOPReward info per segment
    topreward_info = {}
    for seg_idx in targets["source_seg_indices"]:
        seg = seg_lookup.get(seg_idx)
        tr = tr_intervals.get(seg_idx)
        if tr is None or not tr.get("success", False):
            continue
        topreward_info[seg_idx] = {
            "window_start": seg["start_frame"] + tr["start"],
            "window_end": seg["start_frame"] + tr["end"],
            "keyframes": [seg["start_frame"] + kf for kf in tr.get("keyframes", [])
                          if seg["start_frame"] + kf < step_idx],
        }

    # T_j as lists for JSON
    tj_json = {}
    for seg_idx, tj in targets["T_j"].items():
        if len(tj) > 0:
            seg = seg_lookup[seg_idx]
            tj_json[int(seg_idx)] = {
                "start_frame": seg["start_frame"],
                "values": tj.tolist(),
            }

    return jsonify({
        "step_idx": step_idx,
        "has_target": True,
        "target_frames": target_frames,
        "source_seg_indices": targets["source_seg_indices"],
        "P_target": {int(k): v for k, v in targets["P_target"].items()},
        "seg_budgets": {int(k): v for k, v in targets["seg_budgets"].items()},
        "density_keyframes_abs": {int(k): v for k, v in density_kfs_abs.items()},
        "topreward_info": {int(k): v for k, v in topreward_info.items()},
        "T_j": tj_json,
    })


# ---------------------------------------------------------------------------
# API — frame images
# ---------------------------------------------------------------------------

@app.route("/api/<task>/<int:ep>/<int:ts>/frame")
def api_frame(task: str, ep: int, ts: int):
    """Serve a single frame image as JPEG."""
    h5 = _get_h5(task)
    key = f"episode_{ep}/timestep_{ts}/obs/front_rgb"
    if key not in h5:
        return "Frame not found", 404
    rgb = h5[key][()]
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    resp.cache_control.max_age = 86400
    return resp


@app.route("/api/<task>/<int:ep>/frame_strip")
def api_frame_strip(task: str, ep: int):
    """Serve a horizontal strip of frame thumbnails for given indices.

    Query param: indices=0,5,10,20,...
    Returns a single JPEG with thumbnails side by side.
    """
    indices_str = request.args.get("indices", "")
    if not indices_str:
        return "indices parameter required", 400

    indices = [int(x) for x in indices_str.split(",") if x.strip()]
    if not indices:
        return "no valid indices", 400

    h5 = _get_h5(task)
    ep_key = f"episode_{ep}"

    thumbs = []
    for ts in indices:
        key = f"{ep_key}/timestep_{ts}/obs/front_rgb"
        if key in h5:
            rgb = h5[key][()]
            img = Image.fromarray(rgb).resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR)
        else:
            img = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (40, 40, 40))
        thumbs.append(img)

    strip = Image.new("RGB", (THUMB_SIZE * len(thumbs), THUMB_SIZE))
    for i, img in enumerate(thumbs):
        strip.paste(img, (i * THUMB_SIZE, 0))

    buf = io.BytesIO()
    strip.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    resp.cache_control.max_age = 300
    return resp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global EVAL_DIR, DATASET_DIR, H5_DIR, EVAL_RESULTS, EVAL_EPISODE_IDS, MODEL, CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=str, help="Path to eval output dir (e.g. runs/eval_qkfs/step_20000)")
    parser.add_argument("--dataset_path", type=str, default="data/robomme_preprocessed_data")
    parser.add_argument("--h5_dir", type=str,
                        default="/coc/testnvme/shalbe3/robomme_data/robomme_data_h5",
                        help="Path to H5 files with raw frame images")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Override checkpoint path. Default: read from eval_results.json")
    parser.add_argument("--topreward_dir", type=str, default="data/topreward_full",
                        help="Path to TOPReward annotations")
    parser.add_argument("--eval_episodes", type=str, default=None,
                        help="Episode range for eval split, e.g. '80-99'. Shown as tag in UI.")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    EVAL_DIR = Path(args.eval_dir)
    DATASET_DIR = Path(args.dataset_path)
    H5_DIR = Path(args.h5_dir)

    if args.eval_episodes:
        start, end = args.eval_episodes.split("-")
        EVAL_EPISODE_IDS = set(range(int(start), int(end) + 1))
    else:
        EVAL_EPISODE_IDS = None

    # Load eval results
    results_path = EVAL_DIR / "eval_results.json"
    if results_path.exists():
        EVAL_RESULTS = json.load(open(results_path))
    else:
        EVAL_RESULTS = {"overall": {}, "per_task": {}}
        logger.warning("No eval_results.json found at %s", results_path)

    EVAL_RESULTS["topreward_dir"] = args.topreward_dir

    # Load model for interactive queries
    ckpt = args.checkpoint or EVAL_RESULTS.get("checkpoint", "")
    if ckpt:
        try:
            logger.info("Loading QKFS model from %s", ckpt)
            MODEL, CONFIG = load_qkfs(ckpt)
            logger.info("Model loaded.")
        except Exception as e:
            logger.warning("Failed to load model (no GPU?): %s", e)
            logger.warning("Interactive QKFS queries disabled — browse pre-computed results only.")
    else:
        logger.warning("No checkpoint specified — interactive QKFS queries disabled.")

    logger.info("Starting QKFS viewer on %s:%d", args.host, args.port)
    logger.info("Eval dir: %s", EVAL_DIR)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
