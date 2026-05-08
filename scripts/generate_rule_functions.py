"""Generate programmatic sampling-rule functions via Gemini.

For each task, sends the human-written rules + example episode segment
sequences to Gemini, and asks it to produce a Python function that
determines which past segments each exec subtask should sample from.

The generated functions are saved to src/mme_vla_suite/shared/sampling_rules.py
and then applied to a sample of episodes for validation.

Usage:
    python scripts/generate_rule_functions.py
    python scripts/generate_rule_functions.py --model gemini-2.0-flash-lite
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
OUTPUT_MODULE = PROJECT_ROOT / "src" / "mme_vla_suite" / "shared" / "sampling_rules.py"


def extract_segments(ep_data, ts_indices, exec_start):
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
        })
    return segments


def get_example_episodes(task_name: str, n: int = 3) -> list[dict]:
    """Get a few example episodes with their segment sequences."""
    h5_path = H5_DIR / f"record_dataset_{task_name}.h5"
    examples = []
    with h5py.File(str(h5_path), "r") as f:
        ep_indices = get_episode_indices(f)
        for ep_idx in ep_indices[:n]:
            ep = f[f"episode_{ep_idx}"]
            goal = get_task_goal(ep)
            ts_indices = get_timestep_indices(ep)
            exec_start = first_execution_step(ep)
            segments = extract_segments(ep, ts_indices, exec_start)
            examples.append({
                "episode_idx": ep_idx,
                "task_goal": goal,
                "exec_start": exec_start,
                "segments": segments,
            })
    return examples


SYSTEM_PROMPT = """\
You are an expert Python programmer who writes clean, correct functions for \
robot learning data pipelines. You produce only Python code, no markdown."""


def build_codegen_prompt(task_name: str, rules: list[str], examples: list[dict]) -> str:
    # Format example episodes
    ex_strs = []
    for ex in examples:
        seg_lines = []
        for s in ex["segments"]:
            seg_lines.append(f'    {{"idx": {s["idx"]}, "phase": "{s["phase"]}", "label": "{s["label"]}"}}')
        ex_strs.append(
            f'  # Episode {ex["episode_idx"]}: goal="{ex["task_goal"]}"\n'
            f'  # segments:\n' + "\n".join(seg_lines)
        )

    rules_text = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(rules))

    return f"""\
Write a Python function for the task "{task_name}" that determines which past \
segments each execution subtask should sample observations from.

Human-written sampling rules:
{rules_text}

Example episode segment sequences for this task:
{chr(10).join(ex_strs)}

Write a function with this exact signature:

def rule_{task_name.lower()}(segments: list[dict]) -> dict[int, list[int]]:
    \"\"\"Return sampling sources for each exec subtask in {task_name}.

    Args:
        segments: List of dicts with keys "idx", "phase", "label".
            phase is "demo" or "exec". label is the subtask name (lowercase).

    Returns:
        Dict mapping each exec segment index to a list of past segment
        indices to sample from. Omit "all tasks completed" segments.
        An empty list means "no memory needed".
    \"\"\"

Requirements:
- Handle varying numbers of subtasks (episodes may have 2, 3, 4+ cubes etc.)
- Use string matching on labels (e.g. "first", "second", "third" to detect ordinals)
- "all tasks completed" segments should be excluded from output
- Only reference past segments (lower index than current)
- The function must work for ALL episodes of this task, not just the examples
- Keep the code simple and readable — no external imports needed
- Return an empty list [] for subtasks needing no memory

Output ONLY the Python function, nothing else. No markdown fences, no explanation."""


def call_gemini_with_backoff(prompt: str, model, max_retries: int = 6) -> str | None:
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            # Extract retry delay from error
            retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
            if "429" in err_str and retry_match:
                wait = float(retry_match.group(1)) + 2
                print(f"    Rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
            elif "429" in err_str:
                wait = min(60 * (2 ** attempt), 300)
                print(f"    Rate limited (no delay hint), waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"    API error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
    return None


def clean_code(text: str) -> str:
    """Remove markdown fences and any non-code text."""
    text = re.sub(r"^```(?:python)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    # Find the function def and take everything from there
    match = re.search(r"^(def rule_\w+\(.*)", text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1)
    return text


def validate_function(func_code: str, task_name: str, examples: list[dict]) -> list[str]:
    """Try to execute the function on example episodes and check output."""
    issues = []
    namespace = {}
    try:
        exec(func_code, namespace)
    except Exception as e:
        return [f"Syntax/exec error: {e}"]

    func_name = f"rule_{task_name.lower()}"
    if func_name not in namespace:
        return [f"Function {func_name} not found in generated code"]

    func = namespace[func_name]

    for ex in examples:
        try:
            result = func(ex["segments"])
        except Exception as e:
            issues.append(f"Episode {ex['episode_idx']}: runtime error: {e}")
            continue

        if not isinstance(result, dict):
            issues.append(f"Episode {ex['episode_idx']}: returned {type(result)}, expected dict")
            continue

        for key, sources in result.items():
            seg_idx = int(key)
            seg = next((s for s in ex["segments"] if s["idx"] == seg_idx), None)
            if seg is None:
                issues.append(f"Episode {ex['episode_idx']}: segment {seg_idx} not found")
            elif seg["phase"] != "exec":
                issues.append(f"Episode {ex['episode_idx']}: segment {seg_idx} is not exec")
            elif "all tasks completed" in seg["label"]:
                issues.append(f"Episode {ex['episode_idx']}: includes 'all tasks completed'")

            for src in sources:
                if src >= seg_idx:
                    issues.append(f"Episode {ex['episode_idx']}: seg {seg_idx} references future segment {src}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.0-flash-lite")
    parser.add_argument("--examples-per-task", type=int, default=3)
    args = parser.parse_args()

    import google.generativeai as genai
    model = genai.GenerativeModel(
        model_name=args.model,
        system_instruction=SYSTEM_PROMPT,
    )
    print(f"Using model: {args.model}")

    with open(RULES_PATH) as f:
        all_rules = json.load(f)

    generated_functions = {}
    task_names = sorted(
        k for k in all_rules.keys()
        if k != "_updated" and all_rules[k].get("rules")
    )

    for task_name in task_names:
        rules = all_rules[task_name]["rules"]
        print(f"\n{'='*50}")
        print(f"Task: {task_name} ({len(rules)} rules)")

        examples = get_example_episodes(task_name, n=args.examples_per_task)
        prompt = build_codegen_prompt(task_name, rules, examples)

        code = call_gemini_with_backoff(prompt, model)
        if code is None:
            print(f"  FAILED: could not get response")
            continue

        code = clean_code(code)
        print(f"  Generated {len(code)} chars of code")

        # Validate
        issues = validate_function(code, task_name, examples)
        if issues:
            print(f"  VALIDATION ISSUES:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  Validated OK on {len(examples)} example episodes")

        generated_functions[task_name] = code

        # Brief pause between calls
        time.sleep(2)

    # Write the module
    print(f"\n{'='*50}")
    print(f"Writing {len(generated_functions)} functions to {OUTPUT_MODULE}")

    module_parts = [
        '"""Auto-generated sampling rule functions per task.',
        '',
        'Each function takes a list of segment dicts and returns a mapping',
        'from exec segment index to list of past segment indices to sample from.',
        '',
        'Generated by scripts/generate_rule_functions.py using Gemini.',
        '"""',
        '',
        'from __future__ import annotations',
        '',
        '',
    ]

    for task_name in sorted(generated_functions.keys()):
        module_parts.append(generated_functions[task_name])
        module_parts.append("")
        module_parts.append("")

    # Add a dispatch function
    dispatch_entries = []
    for task_name in sorted(generated_functions.keys()):
        dispatch_entries.append(f'    "{task_name}": rule_{task_name.lower()},')

    module_parts.extend([
        "TASK_RULES = {",
        *dispatch_entries,
        "}",
        "",
        "",
        "def get_sampling_sources(task_name: str, segments: list[dict]) -> dict[int, list[int]]:",
        '    """Look up and apply the rule function for the given task."""',
        "    func = TASK_RULES.get(task_name)",
        "    if func is None:",
        f'        raise ValueError(f"No sampling rule for task {{task_name}}")',
        "    return func(segments)",
        "",
    ])

    OUTPUT_MODULE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MODULE, "w") as f:
        f.write("\n".join(module_parts))

    print(f"Module written: {OUTPUT_MODULE}")
    print("Done.")


if __name__ == "__main__":
    main()
