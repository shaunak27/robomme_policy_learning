"""Flask viewer for TOPReward pseudo-labeling results.

Interactive browser for inspecting prefix-based progress curves, success
interval detection, failed region filtering, and keyframe selection.

Usage:
    python annotation_tool/topreward_viewer.py [--port 5052]
    python annotation_tool/topreward_viewer.py --labels_dir data/topreward_labels
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np
from flask import Flask, jsonify, send_file, render_template
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    get_timestep_indices,
)

# ---------------------------------------------------------------------------
# Config (overridable via CLI)
# ---------------------------------------------------------------------------
H5_DIR = PROJECT_ROOT / "data" / "robomme_data_h5"
LABELS_DIR = PROJECT_ROOT / "data" / "topreward_lastpeak"
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "topreward_viewer"
THUMB_H = 64
VIDEO_FPS = 8

CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HDF5 handle cache
# ---------------------------------------------------------------------------
_h5_handles: dict[str, h5py.File] = {}


def _get_h5(env_id: str) -> h5py.File:
    if env_id not in _h5_handles or not _h5_handles[env_id].id.valid:
        path = H5_DIR / f"record_dataset_{env_id}.h5"
        _h5_handles[env_id] = h5py.File(str(path), "r")
    return _h5_handles[env_id]


# ---------------------------------------------------------------------------
# Segment discovery
# ---------------------------------------------------------------------------
def _discover_segments() -> list[dict]:
    segments = []
    if not LABELS_DIR.exists():
        return segments
    for seg_dir in sorted(LABELS_DIR.iterdir()):
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


def _find_segment(seg_id: str) -> dict | None:
    for s in _get_segments():
        if s["seg_id"] == seg_id:
            return s
    return None


# ---------------------------------------------------------------------------
# Routes — page
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("topreward_viewer.html")


# ---------------------------------------------------------------------------
# API — segment list
# ---------------------------------------------------------------------------
@app.route("/api/segments")
def api_segments():
    segs = _get_segments()
    result = []
    for s in segs:
        result.append({
            "seg_id": s["seg_id"],
            "env_id": s.get("env_id", ""),
            "episode_idx": s.get("episode_idx", 0),
            "subtask_label": s.get("subtask_label", ""),
            "task_goal": s.get("task_goal", ""),
            "num_frames": s.get("num_frames", 0),
            "num_prefixes": s.get("num_prefixes", 0),
            "success_detected": s.get("success_detected", False),
            "successful_interval_start_frame": s.get("successful_interval_start_frame"),
            "completion_start_frame": s.get("completion_start_frame"),
            "successful_interval_end_frame": s.get("successful_interval_end_frame"),
            "failed_regions": s.get("failed_regions", []),
            "selected_keyframes": s.get("selected_keyframes", []),
            "confidence": s.get("confidence", 0),
            "method": s.get("method", ""),
        })
    return jsonify(result)


# ---------------------------------------------------------------------------
# API — per-segment data
# ---------------------------------------------------------------------------
@app.route("/api/segment/<seg_id>/progress_scores")
def api_progress_scores(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    path = Path(seg["_dir"]) / "progress_scores.json"
    if not path.exists():
        return jsonify({"error": "no progress scores"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/segment/<seg_id>/interval")
def api_interval(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    path = Path(seg["_dir"]) / "successful_interval.json"
    if not path.exists():
        return jsonify({"error": "no interval"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/segment/<seg_id>/keyframes_data")
def api_keyframes_data(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    path = Path(seg["_dir"]) / "selected_keyframes.json"
    if not path.exists():
        return jsonify({"error": "no keyframes"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/segment/<seg_id>/window_verification")
def api_window_verification(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    path = Path(seg["_dir"]) / "window_verification.json"
    if not path.exists():
        return jsonify({})
    with open(path) as f:
        return jsonify(json.load(f))


# ---------------------------------------------------------------------------
# API — video (segment clip)
# ---------------------------------------------------------------------------
@app.route("/api/segment/<seg_id>/video")
def api_segment_video(seg_id: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404

    cache_path = CACHE_DIR / f"{seg_id}.mp4"
    if cache_path.exists():
        return send_file(str(cache_path), mimetype="video/mp4")

    # Parse seg_id to get env_id, episode_idx, frame range
    env_id = seg["env_id"]
    ep_idx = seg["episode_idx"]
    # Derive frame range from seg_id: env_ep{N}_f{start}-{end}
    seg_id_str = seg["seg_id"]
    parts = seg_id_str.rsplit("_f", 1)
    frame_range = parts[1] if len(parts) == 2 else ""
    start_frame, end_frame = 0, 0
    if "-" in frame_range:
        start_frame, end_frame = map(int, frame_range.split("-"))

    h5 = _get_h5(env_id)
    ep = h5[f"episode_{ep_idx}"]

    frames = []
    for t in range(start_frame, end_frame + 1):
        ts_key = f"timestep_{t}"
        if ts_key not in ep:
            continue
        rgb = ep[f"{ts_key}/obs/front_rgb"][()]
        frames.append(rgb)

    if not frames:
        return jsonify({"error": "no frames"}), 404

    imageio.mimsave(
        str(cache_path), frames, fps=VIDEO_FPS,
        codec="libx264", output_params=["-pix_fmt", "yuv420p"],
    )
    return send_file(str(cache_path), mimetype="video/mp4")


# ---------------------------------------------------------------------------
# API — single frame image
# ---------------------------------------------------------------------------
@app.route("/api/segment/<seg_id>/frame/<int:local_idx>")
def api_frame(seg_id: str, local_idx: int):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404

    env_id = seg["env_id"]
    ep_idx = seg["episode_idx"]
    seg_id_str = seg["seg_id"]
    parts = seg_id_str.rsplit("_f", 1)
    frame_range = parts[1] if len(parts) == 2 else ""
    start_frame = 0
    if "-" in frame_range:
        start_frame = int(frame_range.split("-")[0])

    global_idx = start_frame + local_idx

    h5 = _get_h5(env_id)
    ep = h5[f"episode_{ep_idx}"]
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


# ---------------------------------------------------------------------------
# API — thumbnail strip
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

    env_id = seg["env_id"]
    ep_idx = seg["episode_idx"]
    seg_id_str = seg["seg_id"]
    parts = seg_id_str.rsplit("_f", 1)
    frame_range = parts[1] if len(parts) == 2 else ""
    start_frame, end_frame = 0, 0
    if "-" in frame_range:
        start_frame, end_frame = map(int, frame_range.split("-"))

    h5 = _get_h5(env_id)
    ep = h5[f"episode_{ep_idx}"]

    n_frames = end_frame - start_frame + 1
    strip = Image.new("RGB", (THUMB_H * n_frames, THUMB_H))
    i = 0
    for t in range(start_frame, end_frame + 1):
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
# API — static plot images (progress_curve.png etc.)
# ---------------------------------------------------------------------------
@app.route("/api/segment/<seg_id>/plot/<filename>")
def api_plot(seg_id: str, filename: str):
    seg = _find_segment(seg_id)
    if seg is None:
        return jsonify({"error": "not found"}), 404
    plot_path = Path(seg["_dir"]) / filename
    if not plot_path.exists():
        return jsonify({"error": "not found"}), 404
    mime = "image/png" if filename.endswith(".png") else "image/jpeg"
    return send_file(str(plot_path), mimetype=mime)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5052)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Override topreward labels directory")
    parser.add_argument("--h5_dir", type=str, default=None,
                        help="Override H5 data directory")
    args = parser.parse_args()

    if args.labels_dir:
        LABELS_DIR = Path(args.labels_dir)
    if args.h5_dir:
        H5_DIR = Path(args.h5_dir)

    print(f"Labels dir: {LABELS_DIR}")
    print(f"H5 dir: {H5_DIR}")
    app.run(host=args.host, port=args.port, debug=True)
