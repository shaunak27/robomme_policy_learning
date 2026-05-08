"""Flask backend for visualizing MeMER-computed keyframes per episode.

Read-only viewer: no annotation, just inspection of keyframes overlaid
on the full episode timeline with subgoal and task metadata.
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

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    get_episode_indices,
    get_task_goal,
    get_timestep_indices,
)
from mme_vla_suite.dataset_builder.build_vlm_subgoal_dataset_memer import (
    find_local_minima,
    get_middle_point,
    merge,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
H5_DIR = Path("/coc/testnvme/shalbe3/robomme_data/robomme_data_h5")
CACHE_DIR = Path(__file__).resolve().parent / "cache"
THUMB_H = 64
VIDEO_FPS = 8

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
# MeMER keyframe computation (mirrors build_vlm_subgoal_dataset_memer.py)
# ---------------------------------------------------------------------------

def compute_memer_keyframes(
    episode_data: h5py.Group, env_id: str
) -> dict:
    """Compute MeMER keyframes for an episode. Returns structured result."""
    timestep_indices = get_timestep_indices(episode_data)
    exec_start_idx = first_execution_step(episode_data)

    # 1. Subgoal transition indices
    transition_idxs = []
    last_simple_subgoal = None
    for idx in range(exec_start_idx, len(timestep_indices)):
        sg = episode_data[f"timestep_{idx}"]["info"]["simple_subgoal"][()].decode().lower()
        if "complete" in sg:
            sg = last_simple_subgoal
        if sg != last_simple_subgoal:
            transition_idxs.append(idx)
        last_simple_subgoal = sg
    transition_idxs.append(len(timestep_indices) - 1)

    # 2. Local minima of delta-action norm
    local_minima_idx = find_local_minima(episode_data, timestep_indices)

    # 3. Per-task merging logic
    midpoint_idxs = []
    if env_id in ["PatternLock", "RouteStick"]:
        key_frame_idx = merge(transition_idxs, local_minima_idx, exec_start_idx)
    else:
        key_frame_idx = transition_idxs.copy()

    if env_id == "PickHighlight" and len(key_frame_idx) >= 2:
        frm_0, frm_1 = key_frame_idx[0], key_frame_idx[1]
        mids = get_middle_point(frm_0, frm_1, 1)
        midpoint_idxs.extend(mids)
        key_frame_idx.extend(mids)
        key_frame_idx = sorted(key_frame_idx)

    if env_id == "ButtonUnmaskSwap" and len(key_frame_idx) >= 3:
        frm_0, frm_1, frm_2 = key_frame_idx[0], key_frame_idx[1], key_frame_idx[2]
        mids1 = get_middle_point(frm_0, frm_1, 2)
        mids2 = get_middle_point(frm_1, frm_2, 2)
        midpoint_idxs.extend(mids1 + mids2)
        key_frame_idx.extend(mids1 + mids2)
        key_frame_idx = sorted(key_frame_idx)

    # Remove last (end-of-episode sentinel)
    if key_frame_idx:
        key_frame_idx.pop()

    # Classify provenance of each keyframe
    transition_set = set(transition_idxs)
    minima_set = set(local_minima_idx)
    midpoint_set = set(midpoint_idxs)

    keyframes = []
    for idx in key_frame_idx:
        provenance = []
        if idx in transition_set:
            provenance.append("transition")
        if idx in minima_set:
            provenance.append("local_minimum")
        if idx in midpoint_set:
            provenance.append("midpoint")
        if not provenance:
            # Came from merge() averaging
            provenance.append("merged")

        # Get subgoal at this frame
        sg = episode_data[f"timestep_{idx}"]["info"]["simple_subgoal"][()].decode().lower()
        grounded_sg = episode_data[f"timestep_{idx}"]["info"]["grounded_subgoal"][()].decode().lower()
        if "complete" in sg:
            sg = ""
        if "complete" in grounded_sg:
            grounded_sg = ""

        keyframes.append({
            "idx": idx,
            "provenance": provenance,
            "simple_subgoal": sg,
            "grounded_subgoal": grounded_sg,
        })

    return {
        "keyframes": keyframes,
        "transition_idxs": transition_idxs,
        "local_minima_idxs": local_minima_idx,
        "midpoint_idxs": midpoint_idxs,
        "exec_start_idx": exec_start_idx,
    }


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("memer_viewer.html")


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


# ---------------------------------------------------------------------------
# API — timeline metadata (per-timestep)
# ---------------------------------------------------------------------------
@app.route("/api/<task>/<int:ep>/timeline_meta")
def api_timeline_meta(task: str, ep: int):
    h5 = _get_h5(task)
    ep_data = h5[f"episode_{ep}"]
    ts_indices = get_timestep_indices(ep_data)
    meta = []
    last_subgoal = None
    last_grounded = None
    for ts in ts_indices:
        info = ep_data[f"timestep_{ts}"]["info"]
        is_demo = bool(info["is_video_demo"][()])
        sg_val = info["simple_subgoal"][()].decode() if "simple_subgoal" in info else ""
        grounded_val = info["grounded_subgoal"][()].decode() if "grounded_subgoal" in info else ""
        is_boundary = bool(info["is_subgoal_boundary"][()]) if "is_subgoal_boundary" in info else False

        if "complete" in sg_val.lower() and last_subgoal is not None:
            subgoal = last_subgoal
        else:
            subgoal = sg_val
            last_subgoal = sg_val

        if "complete" in grounded_val.lower() and last_grounded is not None:
            grounded = last_grounded
        else:
            grounded = grounded_val
            last_grounded = grounded_val

        meta.append({
            "idx": ts,
            "is_demo": is_demo,
            "subgoal": subgoal,
            "grounded_subgoal": grounded,
            "is_boundary": is_boundary,
        })
    return jsonify(meta)


# ---------------------------------------------------------------------------
# API — MeMER keyframes
# ---------------------------------------------------------------------------
@app.route("/api/<task>/<int:ep>/memer_keyframes")
def api_memer_keyframes(task: str, ep: int):
    h5 = _get_h5(task)
    ep_data = h5[f"episode_{ep}"]
    result = compute_memer_keyframes(ep_data, task)
    return jsonify(result)


# ---------------------------------------------------------------------------
# API — video
# ---------------------------------------------------------------------------
@app.route("/api/<task>/<int:ep>/video")
def api_video(task: str, ep: int):
    cache_path = CACHE_DIR / f"{task}_ep{ep}.mp4"
    if cache_path.exists():
        return send_file(str(cache_path), mimetype="video/mp4")

    h5 = _get_h5(task)
    ep_data = h5[f"episode_{ep}"]
    ts_indices = get_timestep_indices(ep_data)

    frames = [ep_data[f"timestep_{ts}/obs/front_rgb"][()] for ts in ts_indices]
    imageio.mimsave(
        str(cache_path), frames, fps=VIDEO_FPS,
        codec="libx264", output_params=["-pix_fmt", "yuv420p"],
    )
    return send_file(str(cache_path), mimetype="video/mp4")


# ---------------------------------------------------------------------------
# API — timeline thumb strip
# ---------------------------------------------------------------------------
@app.route("/api/<task>/<int:ep>/thumbstrip")
def api_thumbstrip(task: str, ep: int):
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


# ---------------------------------------------------------------------------
# API — single frame
# ---------------------------------------------------------------------------
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
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=True)