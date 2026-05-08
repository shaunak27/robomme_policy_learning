"""Normalize, smooth, and compute derivatives of a TOPReward progress curve."""

from __future__ import annotations

import numpy as np


def _median_filter(x: np.ndarray, size: int = 3) -> np.ndarray:
    """Simple 1-D median filter (no scipy dependency)."""
    if size < 2:
        return x.copy()
    out = np.empty_like(x)
    half = size // 2
    padded = np.pad(x, half, mode="edge")
    for i in range(len(x)):
        out[i] = np.median(padded[i : i + size])
    return out


def smooth_and_derive(
    reward_norm: np.ndarray,
    median_size: int = 3,
) -> dict:
    """Smooth a normalized reward curve and compute its discrete slope.

    Parameters
    ----------
    reward_norm : 1-D array, values in [0, 1] (min-max normalized).
    median_size : kernel size for the median filter.

    Returns
    -------
    dict with keys:
        reward_smooth : np.ndarray
        delta : np.ndarray  (same length; delta[0] = 0)
    """
    reward_smooth = _median_filter(reward_norm, size=median_size)
    delta = np.zeros_like(reward_smooth)
    delta[1:] = reward_smooth[1:] - reward_smooth[:-1]
    return {
        "reward_smooth": reward_smooth,
        "delta": delta,
    }


def normalize_rewards(raw_reward: np.ndarray) -> np.ndarray:
    """Min-max normalize raw reward values to [0, 1]."""
    rmin, rmax = raw_reward.min(), raw_reward.max()
    return (raw_reward - rmin) / (rmax - rmin + 1e-6)
