"""Flask backend for the frame-importance annotation tool.

Redesigned for per-episode annotation: serves episode videos and
a timeline strip. Annotations are saved per-episode (not per-timestep).
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import imageio
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template
from PIL import Image

# Add project root so we can import h5 utilities
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    get_episode_indices,
    get_task_goal,
    get_timestep_indices,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
H5_DIR = Path("/coc/testnvme/shalbe3/robomme_data/robomme_data_h5")
ANNOTATION_DIR = Path(__file__).resolve().parent / "annotations"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
THUMB_H = 64       # thumbnail height in timeline strip
VIDEO_FPS = 8      # playback fps (slowed down from real-time)

ANNOTATION_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

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


def _available_tasks() -> list[str]:
    return sorted(
        f.stem.replace("record_dataset_", "")
        for f in H5_DIR.glob("record_dataset_*.h5")
    )


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API — metadata
# ---------------------------------------------------------------------------
@app.route("/api/tasks")
def api_tasks():
    return jsonify(_available_tasks())


@app.route("/api/<task>/episodes")
def api_episodes(task: str):
    h5 = _get_h5(task)
    ep_indices = get_episode_indices(h5)
    episodes = []
    for ep_idx in ep_indices:
        ep = h5[f"episode_{ep_idx}"]
        ts_indices = get_timestep_indices(ep)
        exec_start = first_execution_step(ep)
        goal = get_task_goal(ep)
        episodes.append({
            "idx": ep_idx,
            "total_timesteps": len(ts_indices),
            "exec_start_idx": exec_start,
            "task_goal": goal,
        })
    return jsonify(episodes)


@app.route("/api/<task>/<int:ep>/timeline_meta")
def api_timeline_meta(task: str, ep: int):
    h5 = _get_h5(task)
    ep_data = h5[f"episode_{ep}"]
    ts_indices = get_timestep_indices(ep_data)
    meta = []
    last_subgoal = None
    for ts in ts_indices:
        info = ep_data[f"timestep_{ts}"]["info"]
        is_demo = bool(info["is_video_demo"][()])
        sg_val = info["simple_subgoal"][()] if "simple_subgoal" in info else b""
        raw_subgoal = sg_val.decode() if isinstance(sg_val, bytes) else str(sg_val)
        is_boundary = bool(info["is_subgoal_boundary"][()]) if "is_subgoal_boundary" in info else False
        if "complete" in raw_subgoal.lower() and last_subgoal is not None:
            subgoal = last_subgoal
        else:
            subgoal = raw_subgoal
            last_subgoal = raw_subgoal
        meta.append({
            "idx": ts,
            "is_demo": is_demo,
            "subgoal": subgoal,
            "is_boundary": is_boundary,
        })
    return jsonify(meta)


# ---------------------------------------------------------------------------
# API — video
# ---------------------------------------------------------------------------
@app.route("/api/<task>/<int:ep>/video")
def api_video(task: str, ep: int):
    """Serve an MP4 video of the full episode. Cached to disk."""
    cache_path = CACHE_DIR / f"{task}_ep{ep}.mp4"
    if cache_path.exists():
        return send_file(str(cache_path), mimetype="video/mp4")

    h5 = _get_h5(task)
    ep_data = h5[f"episode_{ep}"]
    ts_indices = get_timestep_indices(ep_data)

    frames = []
    for ts in ts_indices:
        rgb = ep_data[f"timestep_{ts}/obs/front_rgb"][()]  # (256,256,3)
        frames.append(rgb)

    imageio.mimsave(
        str(cache_path),
        frames,
        fps=VIDEO_FPS,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],  # browser-compatible
    )
    return send_file(str(cache_path), mimetype="video/mp4")


# ---------------------------------------------------------------------------
# API — timeline thumb strip
# ---------------------------------------------------------------------------
@app.route("/api/<task>/<int:ep>/thumbstrip")
def api_thumbstrip(task: str, ep: int):
    """Sprite sheet: all front_rgb frames at THUMB_H px, concatenated horizontally."""
    cache_path = CACHE_DIR / f"{task}_ep{ep}_strip.jpg"
    if cache_path.exists():
        resp = send_file(str(cache_path), mimetype="image/jpeg")
        resp.cache_control.max_age = 86400
        return resp

    h5 = _get_h5(task)
    ep_data = h5[f"episode_{ep}"]
    ts_indices = get_timestep_indices(ep_data)

    strip = Image.new("RGB", (THUMB_H * len(ts_indices), THUMB_H))
    for i, ts in enumerate(ts_indices):
        rgb = ep_data[f"timestep_{ts}/obs/front_rgb"][()]
        thumb = Image.fromarray(rgb).resize((THUMB_H, THUMB_H), Image.BILINEAR)
        strip.paste(thumb, (i * THUMB_H, 0))

    strip.save(str(cache_path), format="JPEG", quality=80)
    resp = send_file(str(cache_path), mimetype="image/jpeg")
    resp.cache_control.max_age = 86400
    return resp


@app.route("/api/<task>/<int:ep>/<int:ts>/frame/<camera>")
def api_frame(task: str, ep: int, ts: int, camera: str):
    h5 = _get_h5(task)
    rgb = h5[f"episode_{ep}/timestep_{ts}/obs/{camera}"][()]
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    resp.cache_control.max_age = 86400
    return resp


# ---------------------------------------------------------------------------
# API — Annotations (per-episode)
# ---------------------------------------------------------------------------
def _annotation_path(task: str) -> Path:
    return ANNOTATION_DIR / f"{task}.json"


def _load_annotations(task: str) -> dict:
    path = _annotation_path(task)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"task": task, "updated": None, "episodes": {}}


def _save_annotations(task: str, data: dict):
    data["updated"] = datetime.now(timezone.utc).isoformat()
    with open(_annotation_path(task), "w") as f:
        json.dump(data, f, indent=2)


def _segments_to_flat(segments: list[list[int]]) -> list[int]:
    frames = set()
    for start, end in segments:
        frames.update(range(start, end + 1))
    return sorted(frames)


@app.route("/api/annotations/<task>")
def api_get_annotations(task: str):
    return jsonify(_load_annotations(task))


@app.route("/api/annotations/<task>/<int:ep>", methods=["PUT"])
def api_save_episode_annotation(task: str, ep: int):
    """Save annotation for an entire episode."""
    body = request.get_json()
    segments = body.get("selected_segments", [])

    data = _load_annotations(task)
    ep_key = str(ep)

    if ep_key not in data["episodes"]:
        h5 = _get_h5(task)
        ep_data = h5[f"episode_{ep}"]
        ts_indices = get_timestep_indices(ep_data)
        exec_start = first_execution_step(ep_data)
        goal = get_task_goal(ep_data)
        data["episodes"][ep_key] = {
            "total_timesteps": len(ts_indices),
            "exec_start_idx": exec_start,
            "task_goal": goal,
        }

    data["episodes"][ep_key]["selected_segments"] = segments
    data["episodes"][ep_key]["selected_frames"] = _segments_to_flat(segments)
    data["episodes"][ep_key]["timestamp"] = datetime.now(timezone.utc).isoformat()
    _save_annotations(task, data)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
