"""Apply natural-language sampling rules to episodes via Gemini.

For each episode, extracts the subtask segment sequence (demo + exec),
sends it along with the task's human-written rules to Gemini, and gets
back a structured mapping: for each exec segment, which past segments
should be sampled from.

Usage:
    # Sample run (2 episodes per task):
    python scripts/apply_sampling_rules.py --episodes-per-task 2

    # Full run:
    python scripts/apply_sampling_rules.py --episodes-per-task 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    get_episode_indices,
    get_task_goal,
    get_timestep_indices,
)

H5_DIR = Path("/coc/testnvme/shalbe3/robomme_data/robomme_data_h5")
RULES_PATH = PROJECT_ROOT / "annotation_tool" / "taxonomy_rules.json"

# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def extract_segments(ep_data, ts_indices: list[int], exec_start: int) -> list[dict]:
    """Extract ordered list of subtask segments for an episode."""
    segments = []
    current_label = None
    seg_start = None

    for t in ts_indices:
        is_demo = t < exec_start
        raw = ep_data[f"timestep_{t}"]["info"]["simple_subgoal"][()].decode()
        label = raw.strip().lower()
        phase = "demo" if is_demo else "exec"
        key = (phase, label)

        if key != current_label:
            if current_label is not None:
                segments.append({
                    "idx": len(segments),
                    "phase": current_label[0],
                    "label": current_label[1],
                    "start_frame": seg_start,
                    "end_frame": t - 1,
                    "num_frames": t - seg_start,
                })
            current_label = key
            seg_start = t

    if current_label is not None:
        segments.append({
            "idx": len(segments),
            "phase": current_label[0],
            "label": current_label[1],
            "start_frame": seg_start,
            "end_frame": ts_indices[-1],
            "num_frames": ts_indices[-1] - seg_start + 1,
        })

    return segments


# ---------------------------------------------------------------------------
# Gemini prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert at interpreting natural-language rules about robot task \
structure. Given the subtask segment sequence of a robot episode and a set \
of human-written sampling rules, you determine which past segments each \
execution subtask should sample observations from.

You must output valid JSON only — no markdown, no explanation."""

def build_prompt(task_name: str, task_goal: str, segments: list[dict], rules: list[str]) -> str:
    seg_table = []
    for s in segments:
        seg_table.append(
            f"  [{s['idx']:2d}] {s['phase']:4s} | \"{s['label']}\" "
            f"| frames {s['start_frame']}-{s['end_frame']} ({s['num_frames']}f)"
        )

    exec_indices = [s["idx"] for s in segments if s["phase"] == "exec" and "all tasks completed" not in s["label"]]

    rules_text = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(rules))

    return f"""\
Task: {task_name}
Task goal: {task_goal}

Episode segment sequence:
{chr(10).join(seg_table)}

Human-written sampling rules for this task:
{rules_text}

For each execution subtask listed below, decide which past segments \
(by index) it should sample observations from, according to the rules above.

Execution subtask indices to annotate: {exec_indices}

Important guidelines:
- "past segments" means segments with a LOWER index than the current one.
- Include both demo and exec past segments as appropriate.
- If a rule says "no memory" or "needs no memory", return an empty list.
- If a rule says "uniform" or "all of the past", include ALL past segment indices.
- If a rule references "corresponding video segment", find the demo segment(s) \
  whose label best matches the current exec subtask label.
- For rules about "Nth" subtasks needing previous tuples, identify the \
  concrete prior segments by their labels.
- For "all tasks completed" segments, skip them entirely.

Return a JSON object mapping each exec segment index (as a string key) to a \
list of past segment indices to sample from. Example:
{{"4": [0, 1, 2, 3], "6": [0, 5]}}

JSON output:"""


def call_gemini(prompt: str, model, max_retries: int = 3) -> dict:
    """Call Gemini and parse JSON response."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Strip markdown code fences if present
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"    Attempt {attempt+1}: JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"    Attempt {attempt+1}: API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_result(result: dict, segments: list[dict]) -> list[str]:
    """Check that the Gemini output is structurally valid."""
    issues = []
    exec_indices = {
        s["idx"] for s in segments
        if s["phase"] == "exec" and "all tasks completed" not in s["label"]
    }
    all_indices = {s["idx"] for s in segments}

    for key, sources in result.items():
        seg_idx = int(key)
        if seg_idx not in exec_indices:
            issues.append(f"Segment {seg_idx} is not an exec subtask")

        if not isinstance(sources, list):
            issues.append(f"Segment {seg_idx}: sources is not a list")
            continue

        for src in sources:
            if src not in all_indices:
                issues.append(f"Segment {seg_idx}: source {src} does not exist")
            elif src >= seg_idx:
                issues.append(f"Segment {seg_idx}: source {src} is not in the past")

    missing = exec_indices - {int(k) for k in result.keys()}
    if missing:
        issues.append(f"Missing exec segments: {sorted(missing)}")

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply sampling rules via Gemini")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--output", "-o", default=str(PROJECT_ROOT / "data" / "sampling_rules_applied.json"))
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--rate-limit-delay", type=float, default=0.3)
    parser.add_argument("--tasks", nargs="*", help="Subset of tasks to process (default: all)")
    args = parser.parse_args()

    # Load rules
    with open(RULES_PATH) as f:
        all_rules = json.load(f)

    # Init Gemini
    import google.generativeai as genai
    model = genai.GenerativeModel(
        model_name=args.model,
        system_instruction=SYSTEM_PROMPT,
    )
    print(f"Using model: {args.model}")

    h5_files = sorted(H5_DIR.glob("record_dataset_*.h5"))
    results = {}
    stats = {"total": 0, "success": 0, "validation_issues": 0}

    for h5_path in h5_files:
        task_name = h5_path.stem.replace("record_dataset_", "")

        if args.tasks and task_name not in args.tasks:
            continue

        task_rules = all_rules.get(task_name, {}).get("rules", [])
        if not task_rules:
            print(f"\n[SKIP] {task_name}: no rules defined")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Rules: {len(task_rules)}")

        with h5py.File(str(h5_path), "r") as f:
            episode_indices = get_episode_indices(f)
            n_sample = min(args.episodes_per_task, len(episode_indices))
            sampled = episode_indices[:n_sample]

            for ep_idx in sampled:
                ep = f[f"episode_{ep_idx}"]
                task_goal = get_task_goal(ep)
                ts_indices = get_timestep_indices(ep)
                exec_start = first_execution_step(ep)
                segments = extract_segments(ep, ts_indices, exec_start)

                # Skip episodes with no exec subtasks (beyond "completed")
                exec_segs = [s for s in segments if s["phase"] == "exec" and "all tasks completed" not in s["label"]]
                if not exec_segs:
                    continue

                print(f"\n  Episode {ep_idx}: {len(segments)} segments ({len(exec_segs)} exec)")
                for s in segments:
                    print(f"    [{s['idx']:2d}] {s['phase']:4s} | {s['label']}")

                prompt = build_prompt(task_name, task_goal, segments, task_rules)
                stats["total"] += 1

                result = call_gemini(prompt, model)
                time.sleep(args.rate_limit_delay)

                if result is None:
                    print(f"    FAILED: could not get valid JSON")
                    continue

                # Validate
                issues = validate_result(result, segments)
                if issues:
                    print(f"    VALIDATION ISSUES:")
                    for issue in issues:
                        print(f"      - {issue}")
                    stats["validation_issues"] += 1

                stats["success"] += 1

                # Pretty print result
                print(f"    RESULT:")
                for seg_idx_str, sources in sorted(result.items(), key=lambda x: int(x[0])):
                    seg = segments[int(seg_idx_str)]
                    source_labels = []
                    for src_idx in sources:
                        src_seg = segments[src_idx]
                        source_labels.append(f"[{src_idx}]{src_seg['phase']}:{src_seg['label']}")
                    print(f"      [{seg_idx_str}] \"{seg['label']}\" <- {source_labels if source_labels else '(no memory)'}")

                ep_key = f"{task_name}_ep{ep_idx}"
                results[ep_key] = {
                    "task": task_name,
                    "episode_idx": ep_idx,
                    "task_goal": task_goal,
                    "segments": segments,
                    "sampling_map": result,
                    "validation_issues": issues,
                }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f_out:
        json.dump(results, f_out, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. {stats['success']}/{stats['total']} episodes processed successfully.")
    print(f"Validation issues: {stats['validation_issues']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
