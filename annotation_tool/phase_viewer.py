"""Flask viewer for pseudo-labeling results.

Interactive browser for inspecting CLIP/MLLM phase labels alongside the
original video.  Video player, color-coded timeline, live score curves,
and representative frame thumbnails.

Usage:
    python annotation_tool/phase_viewer.py [--port 5051]
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    get_timestep_indices,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
H5_DIR = PROJECT_ROOT / "data" / "robomme_data_h5"
PSEUDO_LABEL_DIR = PROJECT_ROOT / "data" / "pseudo_labels"
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "phase_viewer"
THUMB_H = 64
VIDEO_FPS = 8

CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HDF5 handle cache
# ---------------------------------------------------------------------------
_h5_handles: dict[str, h5py.File] = {}


def _get_h5(task: str) -> h5py.File:
    if task not in _h5_handles or not _h5_handles[task].id.valid:
        path = H5_DIR / f"record_dataset_{task}.h5"
        _h5_handles[task] = h5py.File(str(path), "r")
    return _h5_handles[task]


# ---------------------------------------------------------------------------
# Segment discovery
# ---------------------------------------------------------------------------
def _discover_segments() -> list[dict]:
    """Scan pseudo_labels dir and return segment metadata."""
    segments = []
    if not PSEUDO_LABEL_DIR.exists():
        return segments
    for seg_dir in sorted(PSEUDO_LABEL_DIR.iterdir()):
        if not seg_dir.is_dir() or seg_dir.name.startswith("_"):
            continue
        summary_path = seg_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        if summary.get("skipped"):
            continue
        summary["_dir"] = str(seg_dir)
        segments.append(summary)
    return segments


_segments_cache: list[dict] | None = None


def _get_segments() -> list[dict]:
    global _segments_cache
    if _segments_cache is None:
        _segments_cache = _discover_segments()
    return _segments_cache


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("phase_viewer.html")


# ---------------------------------------------------------------------------
# API — segment list
# ---------------------------------------------------------------------------
@app.route("/api/segments")
def api_segments():
    segs = _get_segments()
    # Return light metadata (no _dir)
    result = []
    for s in segs:
        result.append({
            "seg_id": s["seg_id"],
            "env_id": s["env_id"],
            "episode_idx": s["episode_idx"],
            "subtask_label": s["subtask_label"],
            "task_goal": s["task_goal"],
            "start_frame": s["start_frame"],
            "end_frame": s["end_frame"],
            "num_frames": s["num_frames"],
            "label_source": s.get("label_source", "clip_heuristic_only"),
            "transition_start": s.get("transition_start"),
            "transition_frame": s.get("transition_frame"),
            "completion_start": s.get("completion_start"),
            "num_failed_attempts": s.get("num_failed_attempts", 0),
            "failed_attempt_frames": s.get("failed_attempt_frames", []),
            "label_counts": s.get("label_counts", {}),
            "mean_confidence": s.get("mean_confidence", 0),
        })
    return jsonify(result)


# ---------------------------------------------------------------------------
# API — per-segment data
# ---------------------------------------------------------------------------
@app.route("/api/segment/<seg_id>/labels")
def api_labels(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    labels_path = Path(seg["_dir"]) / "frame_labels.json"
    with open(labels_path) as f:
        return jsonify(json.load(f))


@app.route("/api/segment/<seg_id>/scores")
def api_scores(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    scores_path = Path(seg["_dir"]) / "clip_scores_smooth.npz"
    data = np.load(str(scores_path))
    return jsonify({
        "frame_indices": data["frame_indices"].tolist(),
        "precondition": data["precondition"].tolist(),
        "contact_or_transition": data["contact_or_transition"].tolist(),
        "postcondition_success": data["postcondition_success"].tolist(),
        "failed_attempt": data["failed_attempt"].tolist(),
    })


@app.route("/api/segment/<seg_id>/attempt_scores")
def api_attempt_scores(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    path = Path(seg["_dir"]) / "attempt_score.npz"
    if not path.exists():
        return jsonify({})
    data = np.load(str(path))
    return jsonify({
        "frame_indices": data["frame_indices"].tolist(),
        "attempt_score": data["attempt_score"].tolist(),
        "visual_change": data["visual_change"].tolist(),
        "gripper_change": data["gripper_change"].tolist(),
        "state_change": data["state_change"].tolist(),
        "clip_contact": data["clip_contact"].tolist(),
    })


@app.route("/api/segment/<seg_id>/candidates")
def api_candidates(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    path = Path(seg["_dir"]) / "candidate_windows.json"
    if not path.exists():
        return jsonify([])
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/segment/<seg_id>/mllm")
def api_mllm(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    mllm_path = Path(seg["_dir"]) / "mllm_results.json"
    if not mllm_path.exists():
        return jsonify([])
    with open(mllm_path) as f:
        return jsonify(json.load(f))


# ---------------------------------------------------------------------------
# API — video (segment clip only)
# ---------------------------------------------------------------------------
@app.route("/api/segment/<seg_id>/video")
def api_segment_video(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404

    cache_path = CACHE_DIR / f"{seg_id}.mp4"
    if cache_path.exists():
        return send_file(str(cache_path), mimetype="video/mp4")

    h5 = _get_h5(seg["env_id"])
    ep = h5[f"episode_{seg['episode_idx']}"]

    frames = []
    for t in range(seg["start_frame"], seg["end_frame"] + 1):
        ts_key = f"timestep_{t}"
        if ts_key not in ep:
            continue
        rgb = ep[f"{ts_key}/obs/front_rgb"][()]
        frames.append(rgb)

    if not frames:
        return jsonify({"error": "no frames"}), 404

    imageio.mimsave(
        str(cache_path), frames, fps=VIDEO_FPS,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    return send_file(str(cache_path), mimetype="video/mp4")


# ---------------------------------------------------------------------------
# API — frame image
# ---------------------------------------------------------------------------
@app.route("/api/segment/<seg_id>/frame/<int:global_idx>")
def api_frame(seg_id: str, global_idx: int):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404

    h5 = _get_h5(seg["env_id"])
    ep = h5[f"episode_{seg['episode_idx']}"]
    ts_key = f"timestep_{global_idx}"
    if ts_key not in ep:
        return jsonify({"error": "frame not found"}), 404

    rgb = ep[f"{ts_key}/obs/front_rgb"][()]
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    resp.cache_control.max_age = 86400
    return resp


@app.route("/api/segment/<seg_id>/wrist_frame/<int:global_idx>")
def api_wrist_frame(seg_id: str, global_idx: int):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404

    h5 = _get_h5(seg["env_id"])
    ep = h5[f"episode_{seg['episode_idx']}"]
    ts_key = f"timestep_{global_idx}"
    if ts_key not in ep:
        return jsonify({"error": "frame not found"}), 404

    rgb = ep[f"{ts_key}/obs/wrist_rgb"][()]
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    resp.cache_control.max_age = 86400
    return resp


# ---------------------------------------------------------------------------
# API — thumbnail strip (segment frames only)
# ---------------------------------------------------------------------------
@app.route("/api/segment/<seg_id>/thumbstrip")
def api_thumbstrip(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404

    cache_path = CACHE_DIR / f"{seg_id}_strip.jpg"
    if cache_path.exists():
        resp = send_file(str(cache_path), mimetype="image/jpeg")
        resp.cache_control.max_age = 86400
        return resp

    h5 = _get_h5(seg["env_id"])
    ep = h5[f"episode_{seg['episode_idx']}"]

    n_frames = seg["end_frame"] - seg["start_frame"] + 1
    strip = Image.new("RGB", (THUMB_H * n_frames, THUMB_H))
    i = 0
    for t in range(seg["start_frame"], seg["end_frame"] + 1):
        ts_key = f"timestep_{t}"
        if ts_key not in ep:
            i += 1
            continue
        rgb = ep[f"{ts_key}/obs/front_rgb"][()]
        thumb = Image.fromarray(rgb).resize((THUMB_H, THUMB_H), Image.BILINEAR)
        strip.paste(thumb, (i * THUMB_H, 0))
        i += 1

    strip.save(str(cache_path), format="JPEG", quality=80)
    resp = send_file(str(cache_path), mimetype="image/jpeg")
    resp.cache_control.max_age = 86400
    return resp


# ---------------------------------------------------------------------------
# API — static plot images
# ---------------------------------------------------------------------------
@app.route("/api/segment/<seg_id>/plot/<filename>")
def api_plot(seg_id: str, filename: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    plot_path = Path(seg["_dir"]) / filename
    if not plot_path.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(str(plot_path), mimetype="image/png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_segment(seg_id: str) -> dict | None:
    for s in _get_segments():
        if s["seg_id"] == seg_id:
            return s
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True)
