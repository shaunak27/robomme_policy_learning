"""Select a stratified subset of episodes for pseudo-labeling.

Samples ~10% of training episodes, stratified across task files (each H5
file corresponds to a different task environment).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import h5py
import numpy as np

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    get_episode_indices,
    get_task_goal,
    get_timestep_indices,
)


@dataclass
class SubtaskSegment:
    """A contiguous segment of frames sharing the same subtask label."""

    episode_h5_path: str
    episode_idx: int
    task_goal: str
    subtask_label: str
    start_frame: int  # inclusive
    end_frame: int  # inclusive
    exec_start_idx: int
    env_id: str


def extract_subtask_segments(
    h5_path: str, episode_idx: int, env_id: str
) -> list[SubtaskSegment]:
    """Extract contiguous subtask segments from one episode.

    Only considers execution frames (after the video demo portion).
    Each segment spans from the first frame of a subtask label to the last
    frame before the next subtask label changes.
    """
    segments: list[SubtaskSegment] = []
    with h5py.File(h5_path, "r") as f:
        ep = f[f"episode_{episode_idx}"]
        task_goal = get_task_goal(ep, lower=True)
        exec_start = first_execution_step(ep)
        timestep_idxs = get_timestep_indices(ep)

        current_label: str | None = None
        seg_start: int | None = None

        for t in timestep_idxs:
            if t < exec_start:
                continue
            ts = ep[f"timestep_{t}"]
            if ts["info"]["is_completed"][()]:
                # End of episode; close current segment
                if current_label is not None and seg_start is not None:
                    segments.append(
                        SubtaskSegment(
                            episode_h5_path=h5_path,
                            episode_idx=episode_idx,
                            task_goal=task_goal,
                            subtask_label=current_label,
                            start_frame=seg_start,
                            end_frame=t - 1,
                            exec_start_idx=exec_start,
                            env_id=env_id,
                        )
                    )
                break

            label = ts["info"]["simple_subgoal"][()].decode().lower()
            if label != current_label:
                # Close previous segment
                if current_label is not None and seg_start is not None:
                    segments.append(
                        SubtaskSegment(
                            episode_h5_path=h5_path,
                            episode_idx=episode_idx,
                            task_goal=task_goal,
                            subtask_label=current_label,
                            start_frame=seg_start,
                            end_frame=t - 1,
                            exec_start_idx=exec_start,
                            env_id=env_id,
                        )
                    )
                current_label = label
                seg_start = t

        # Close final segment if episode ended without is_completed
        if current_label is not None and seg_start is not None:
            last_t = max(t for t in timestep_idxs if t >= exec_start)
            if not segments or segments[-1].end_frame < last_t:
                segments.append(
                    SubtaskSegment(
                        episode_h5_path=h5_path,
                        episode_idx=episode_idx,
                        task_goal=task_goal,
                        subtask_label=current_label,
                        start_frame=seg_start,
                        end_frame=last_t,
                        exec_start_idx=exec_start,
                        env_id=env_id,
                    )
                )

    # Filter out very short segments (< 5 frames)
    segments = [s for s in segments if s.end_frame - s.start_frame >= 4]
    return segments


def sample_episodes_stratified(
    raw_data_path: str,
    fraction: float = 0.10,
    episodes_per_task: int | None = None,
    seed: int = 42,
) -> list[SubtaskSegment]:
    """Sample episodes per H5 file, return all their subtask segments.

    Stratification is by H5 file (each file = one task environment).
    If episodes_per_task is set, it overrides fraction.
    """
    rng = random.Random(seed)
    all_segments: list[SubtaskSegment] = []

    h5_files = sorted(
        f for f in os.listdir(raw_data_path) if f.endswith(".h5")
    )

    for fname in h5_files:
        h5_path = os.path.join(raw_data_path, fname)
        env_id = fname.split(".")[0].split("_")[-1]

        with h5py.File(h5_path, "r") as f:
            episode_indices = get_episode_indices(f)

        if episodes_per_task is not None:
            n_sample = min(episodes_per_task, len(episode_indices))
        else:
            n_sample = max(1, int(len(episode_indices) * fraction))
        sampled = sorted(rng.sample(episode_indices, n_sample))

        print(f"  {env_id}: {len(sampled)}/{len(episode_indices)} episodes sampled")

        for ep_idx in sampled:
            segs = extract_subtask_segments(h5_path, ep_idx, env_id)
            all_segments.extend(segs)

    print(f"\nTotal segments to pseudo-label: {len(all_segments)}")
    return all_segments
