"""Visualization helpers for the NUTS calibration skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def build_trace_plot(samples: np.ndarray, parameter_names: list[str], path: Path, dpi: int = 140) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    chains, draws, dim = samples.shape
    fig, axes = plt.subplots(dim, 1, figsize=(8, 2.4 * dim), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    xs = np.arange(draws)
    for idx, name in enumerate(parameter_names):
        for chain in range(chains):
            axes[idx].plot(xs, samples[chain, :, idx], linewidth=0.8, alpha=0.8)
        axes[idx].set_ylabel(name)
    axes[-1].set_xlabel("draw")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def build_marginal_plot(samples: np.ndarray, parameter_names: list[str], path: Path, dpi: int = 140) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    merged = samples.reshape(-1, samples.shape[-1])
    cols = min(3, len(parameter_names))
    rows = int(np.ceil(len(parameter_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, name in enumerate(parameter_names):
        axes[idx].hist(merged[:, idx], bins=30, color="#2f6f8f", alpha=0.85)
        axes[idx].set_title(name)
    for idx in range(len(parameter_names), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def build_pairwise_plot(samples: np.ndarray, parameter_names: list[str], path: Path, dpi: int = 140) -> bool:
    plt = _load_pyplot()
    if plt is None or len(parameter_names) < 2:
        return False
    merged = samples.reshape(-1, samples.shape[-1])
    names = parameter_names[: min(4, len(parameter_names))]
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for row, y_name in enumerate(names):
        for col, x_name in enumerate(names):
            ax = axes[row, col]
            x = merged[:, parameter_names.index(x_name)]
            y = merged[:, parameter_names.index(y_name)]
            if row == col:
                ax.hist(x, bins=25, color="#7eaed2", alpha=0.9)
            else:
                ax.scatter(x, y, s=6, alpha=0.35, color="#1b4d6b")
            if row == n - 1:
                ax.set_xlabel(x_name)
            if col == 0:
                ax.set_ylabel(y_name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def build_autocorrelation_plot(samples: np.ndarray, parameter_names: list[str], path: Path, dpi: int = 140, max_lag: int = 40) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    merged = samples.reshape(-1, samples.shape[-1])
    cols = min(3, len(parameter_names))
    rows = int(np.ceil(len(parameter_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, name in enumerate(parameter_names):
        series = merged[:, idx] - np.mean(merged[:, idx])
        variance = np.var(series)
        if variance == 0:
            acf = np.ones(max_lag)
        else:
            values = [1.0]
            for lag in range(1, max_lag):
                left = series[:-lag]
                right = series[lag:]
                values.append(float(np.dot(left, right) / (variance * left.size)))
            acf = np.asarray(values, dtype=float)
        axes[idx].stem(np.arange(max_lag), acf, basefmt=" ")
        axes[idx].set_title(name)
        axes[idx].set_ylim(-0.1, 1.05)
    for idx in range(len(parameter_names), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def build_energy_plot(energy: np.ndarray, path: Path, dpi: int = 140) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs = np.arange(energy.shape[1])
    for chain in range(energy.shape[0]):
        axes[0].plot(xs, energy[chain], linewidth=0.8, alpha=0.8)
    axes[0].set_title("Energy Trace")
    axes[1].hist(energy.reshape(-1), bins=30, color="#8b5a2b", alpha=0.85)
    axes[1].set_title("Energy Histogram")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def build_posterior_predictive_plot(
    predictive: dict[str, Any],
    observed: np.ndarray,
    path: Path,
    dpi: int = 140,
) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    mean = np.asarray(predictive.get("mean", []), dtype=float).reshape(-1)
    lower = np.asarray(predictive.get("ci_2_5", []), dtype=float).reshape(-1)
    upper = np.asarray(predictive.get("ci_97_5", []), dtype=float).reshape(-1)
    obs = np.asarray(observed, dtype=float).reshape(-1)
    if mean.size == 0 or obs.size == 0:
        return False
    xs = np.arange(obs.size)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(xs, lower, upper, color="#c9d9e8", alpha=0.8, label="95% predictive interval")
    ax.plot(xs, mean, color="#1f4e79", linewidth=1.5, label="predictive mean")
    ax.scatter(xs, obs, color="#b03a2e", s=18, label="observed", zorder=3)
    ax.legend(loc="best")
    ax.set_xlabel("output index")
    ax.set_ylabel("value")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True

