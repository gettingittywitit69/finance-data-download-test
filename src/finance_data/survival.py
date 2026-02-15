"""Helpers for Sharpe-ratio survival maps."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional
    plt = None

__all__ = [
    "compute_survival_map",
    "plot_survival_map",
    "rolling_windows",
]


def rolling_windows(n: int, window: int = 120, step: int = 12) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for rolling windows with a given step."""
    out: list[tuple[int, int]] = []
    i = 0
    while i + window <= n:
        out.append((i, i + window))
        i += step
    return out


def compute_survival_map(
    excess_returns: pd.DataFrame,
    window: int = 120,
    step: int = 12,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute rolling Sharpe ratios and a binary survival map (SR>0) over time.
    """
    ex = excess_returns.copy().sort_index()
    ex = ex.dropna(how="all")

    windows = rolling_windows(len(ex), window=window, step=step)
    cols = list(ex.columns)
    sr_mat = np.full((len(cols), len(windows)), np.nan)
    survival = np.full_like(sr_mat, np.nan)

    for w_idx, (a, b) in enumerate(windows):
        window_df = ex.iloc[a:b]
        mu = window_df.mean(axis=0)
        sd = window_df.std(axis=0, ddof=1)
        sr = mu / sd
        sr = sr.replace([np.inf, -np.inf], np.nan)
        sr_mat[:, w_idx] = sr.values
        survival[:, w_idx] = (sr > 0).astype(float).values

    end_dates = [ex.index[end - 1] for _, end in windows] if windows else []
    meta = {"columns": cols, "window_end_dates": end_dates, "windows": windows}
    return sr_mat, survival, meta


def plot_survival_map(
    survival: np.ndarray,
    columns: Sequence[str],
    window_end_dates: Sequence[pd.Timestamp],
    title: str | None = None,
):
    """Visualize the survival map (1 if SR>0 else 0) over rolling windows."""
    if plt is None or survival.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(survival, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)

    tick_pos = list(range(0, len(window_end_dates), max(1, len(window_end_dates) // 12))) if window_end_dates else []
    tick_labels = [pd.to_datetime(window_end_dates[t]).strftime("%Y") for t in tick_pos] if tick_pos else []
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_title(title or "Sharpe Survival Map (SR>0)")
    ax.set_xlabel("Window end year")
    ax.set_ylabel("Portfolio")
    fig.tight_layout()
    return fig
