"""Flask app for interactive subtask taxonomy visualization.

Reads the precomputed taxonomy.json and serves an interactive dashboard
showing per-task subtask statistics, ordering, transitions, and coverage.

Usage:
    python annotation_tool/taxonomy_viewer.py
    # Then open http://localhost:5052
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy.json"
RULES_PATH = Path(__file__).resolve().parent / "taxonomy_rules.json"
SAMPLING_RESULTS_PATH = Path(__file__).resolve().parent / "sampling_results.json"
EDIT_INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "edit_instructions.json"
SAMPLING_DENSITY_PATH = Path(__file__).resolve().parent / "sampling_density.json"
DENSITY_RESULTS_PATH = Path(__file__).resolve().parent / "density_results.json"

app = Flask(__name__)

_taxonomy: dict | None = None


def _load_taxonomy() -> dict:
    global _taxonomy
    if _taxonomy is None:
        with open(TAXONOMY_PATH) as f:
            _taxonomy = json.load(f)
    return _taxonomy


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("taxonomy_viewer.html")


@app.route("/api/tasks")
def api_tasks():
    tax = _load_taxonomy()
    edit_data = _load_edit_instructions()
    edited_tasks = {k for k, v in edit_data.items() if k != "_updated" and v.get("text", "").strip()}
    summary = []
    for name, data in sorted(tax.items()):
        summary.append({
            "name": name,
            "task_goal": data["task_goal"],
            "num_episodes": data["num_episodes"],
            "has_video_demo": data["has_video_demo"],
            "num_exec_subtasks": len(data["exec"]["subtasks"]),
            "num_demo_subtasks": len(data["demo"]["subtasks"]) if data.get("demo") else 0,
            "needs_review": name in edited_tasks,
        })
    return jsonify(summary)


@app.route("/api/task/<task_name>")
def api_task_detail(task_name: str):
    tax = _load_taxonomy()
    if task_name not in tax:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(tax[task_name])


@app.route("/api/overview")
def api_overview():
    """Cross-task summary for the overview panel."""
    tax = _load_taxonomy()
    overview = {
        "total_tasks": len(tax),
        "total_episodes": sum(d["num_episodes"] for d in tax.values()),
        "tasks_with_demo": sum(1 for d in tax.values() if d["has_video_demo"]),
        "per_task": [],
    }
    for name, data in sorted(tax.items()):
        exec_subtasks = data["exec"]["subtasks"]
        overview["per_task"].append({
            "name": name,
            "num_episodes": data["num_episodes"],
            "num_subtasks": len(exec_subtasks),
            "has_demo": data["has_video_demo"],
            "top_canonical": data["exec"]["canonical_sequences"][0]["sequence"] if data["exec"]["canonical_sequences"] else [],
        })
    return jsonify(overview)


# ---------------------------------------------------------------------------
# API — Subtask dependency rules
# ---------------------------------------------------------------------------
def _load_rules() -> dict:
    if RULES_PATH.exists():
        with open(RULES_PATH) as f:
            return json.load(f)
    return {}


def _save_rules(data: dict):
    data["_updated"] = datetime.now(timezone.utc).isoformat()
    with open(RULES_PATH, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/api/rules/<task_name>")
def api_get_rules(task_name: str):
    rules = _load_rules()
    return jsonify(rules.get(task_name, {"rules": []}))


@app.route("/api/rules/<task_name>", methods=["PUT"])
def api_save_rules(task_name: str):
    body = request.get_json()
    all_rules = _load_rules()
    all_rules[task_name] = {
        "rules": body.get("rules", []),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    _save_rules(all_rules)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# API — Sampling results (applied rules)
# ---------------------------------------------------------------------------
def _load_sampling_results() -> dict:
    if SAMPLING_RESULTS_PATH.exists():
        with open(SAMPLING_RESULTS_PATH) as f:
            return json.load(f)
    return {}


@app.route("/api/sampling/<task_name>")
def api_sampling_results(task_name: str):
    """Return applied sampling results for all episodes of a task."""
    all_results = _load_sampling_results()
    task_episodes = []
    for key, data in all_results.items():
        if data["task"] == task_name:
            task_episodes.append(data)
    task_episodes.sort(key=lambda x: x["episode_idx"])
    return jsonify(task_episodes)


# ---------------------------------------------------------------------------
# API — Edit instructions (post-review feedback)
# ---------------------------------------------------------------------------
def _load_edit_instructions() -> dict:
    if EDIT_INSTRUCTIONS_PATH.exists():
        with open(EDIT_INSTRUCTIONS_PATH) as f:
            return json.load(f)
    return {}


@app.route("/api/edit_instructions/<task_name>")
def api_get_edit_instructions(task_name: str):
    data = _load_edit_instructions()
    return jsonify(data.get(task_name, {"text": ""}))


@app.route("/api/edit_instructions/<task_name>", methods=["PUT"])
def api_save_edit_instructions(task_name: str):
    body = request.get_json()
    data = _load_edit_instructions()
    data[task_name] = {
        "text": body.get("text", ""),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    data["_updated"] = datetime.now(timezone.utc).isoformat()
    with open(EDIT_INSTRUCTIONS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# API — Density results (frame-level selections)
# ---------------------------------------------------------------------------

def _load_density_results() -> dict:
    if DENSITY_RESULTS_PATH.exists():
        with open(DENSITY_RESULTS_PATH) as f:
            return json.load(f)
    return {}


@app.route("/api/density/<task_name>")
def api_density_results(task_name: str):
    """Return frame-level density results for all episodes of a task."""
    all_results = _load_density_results()
    task_episodes = []
    for key, data in all_results.items():
        if data["task"] == task_name:
            task_episodes.append(data)
    task_episodes.sort(key=lambda x: x["episode_idx"])
    return jsonify(task_episodes)


# ---------------------------------------------------------------------------
# API — Subtask categories & sampling density
# ---------------------------------------------------------------------------

# Colors and ordinals to normalize when building generic categories
_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "cyan",
    "white", "black", "brown", "gray", "grey",
]
_ORDINALS = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
]


def _normalize_label(label: str) -> str:
    """Replace concrete colors and ordinals with placeholders to form a generic category."""
    result = label
    for color in _COLORS:
        result = re.sub(rf'\b{color}\b', '{color}', result)
    for ordinal in _ORDINALS:
        result = re.sub(rf'\b{ordinal}\b', '{N}', result)
    # Collapse multiple {color}/{N} tokens if adjacent
    result = re.sub(r'\{N\}\s+\{color\}', '{N} {color}', result)
    # Also normalize numbered patterns like "1st", "2nd" etc.
    result = re.sub(r'\b\d+(st|nd|rd|th)\b', '{N}', result)
    # Normalize "for 1 time" -> "for {N} time"
    result = re.sub(r'for \d+ time', 'for {N} time', result)
    return result


def _get_categories(task_name: str) -> list[dict]:
    """Compute generic subtask categories for both exec and demo phases."""
    tax = _load_taxonomy()
    if task_name not in tax:
        return []

    data = tax[task_name]
    categories: dict[str, dict] = {}  # generic_label -> {phase, members}

    for phase_key in ["exec", "demo"]:
        phase_data = data.get(phase_key)
        if not phase_data:
            continue
        for label, stats in phase_data["subtasks"].items():
            generic = _normalize_label(label)
            cat_key = f"{phase_key}::{generic}"
            if cat_key not in categories:
                categories[cat_key] = {
                    "generic_label": generic,
                    "phase": phase_key,
                    "members": [],
                    "total_episode_count": 0,
                }
            categories[cat_key]["members"].append(label)
            categories[cat_key]["total_episode_count"] += stats.get("episode_count", 0)

    # Sort: exec first, then by total episode count descending
    result = sorted(
        categories.values(),
        key=lambda c: (0 if c["phase"] == "exec" else 1, -c["total_episode_count"]),
    )
    return result


@app.route("/api/categories/<task_name>")
def api_categories(task_name: str):
    return jsonify(_get_categories(task_name))


def _load_sampling_density() -> dict:
    if SAMPLING_DENSITY_PATH.exists():
        with open(SAMPLING_DENSITY_PATH) as f:
            return json.load(f)
    return {}


@app.route("/api/sampling_density/<task_name>")
def api_get_sampling_density(task_name: str):
    data = _load_sampling_density()
    return jsonify(data.get(task_name, {}))


@app.route("/api/sampling_density/<task_name>", methods=["PUT"])
def api_save_sampling_density(task_name: str):
    body = request.get_json()
    data = _load_sampling_density()
    data[task_name] = {
        "densities": body.get("densities", {}),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    data["_updated"] = datetime.now(timezone.utc).isoformat()
    with open(SAMPLING_DENSITY_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"ok": True})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Subtask taxonomy viewer")
    parser.add_argument("--port", "-p", type=int, default=5058)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
