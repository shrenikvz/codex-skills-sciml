"""Plotting helpers for posterior diagnostics and predictive checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        return plt
    except Exception:
        return None


def plot_posterior_marginals(samples: np.ndarray, parameter_names: list[str], path: Path, dpi: int = 140) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    flat = samples.reshape(-1, samples.shape[-1])
    cols = min(3, len(parameter_names))
    rows = int(math.ceil(len(parameter_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, name in enumerate(parameter_names):
        axes[idx].hist(flat[:, idx], bins=30, color="#2f6f8f", alpha=0.85)
        axes[idx].set_title(name)
    for idx in range(len(parameter_names), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def plot_pairwise(samples: np.ndarray, parameter_names: list[str], path: Path, dpi: int = 140) -> bool:
    plt = _load_pyplot()
    if plt is None or len(parameter_names) < 2:
        return False
    flat = samples.reshape(-1, samples.shape[-1])
    names = parameter_names[: min(5, len(parameter_names))]
    idxs = [parameter_names.index(name) for name in names]
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for row, y_idx in enumerate(idxs):
        for col, x_idx in enumerate(idxs):
            ax = axes[row, col]
            if row == col:
                ax.hist(flat[:, x_idx], bins=25, color="#8ab5d6", alpha=0.85)
            else:
                ax.scatter(flat[:, x_idx], flat[:, y_idx], s=5, alpha=0.35, color="#0f3d56")
            if row == n - 1:
                ax.set_xlabel(names[col])
            if col == 0:
                ax.set_ylabel(names[row])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def plot_trace(samples: np.ndarray, parameter_names: list[str], path: Path, dpi: int = 140) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    chains, draws, dims = samples.shape
    fig, axes = plt.subplots(dims, 1, figsize=(9, 2.4 * dims), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    xs = np.arange(draws)
    for dim, name in enumerate(parameter_names):
        for chain in range(chains):
            axes[dim].plot(xs, samples[chain, :, dim], linewidth=0.8, alpha=0.8)
        axes[dim].set_ylabel(name)
    axes[-1].set_xlabel("draw")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def plot_autocorrelation(samples: np.ndarray, parameter_names: list[str], path: Path, dpi: int = 140, max_lag: int = 40) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    chains, _, dims = samples.shape
    fig, axes = plt.subplots(dims, 1, figsize=(8, 2.4 * dims), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for dim, name in enumerate(parameter_names):
        mean_ac = np.zeros(max_lag + 1, dtype=float)
        for chain in range(chains):
            values = samples[chain, :, dim]
            centered = values - np.mean(values)
            var = np.var(centered)
            ac = [1.0]
            for lag in range(1, max_lag + 1):
                if lag >= values.size or var <= 0:
                    ac.append(0.0)
                else:
                    ac.append(float(np.dot(centered[:-lag], centered[lag:]) / ((values.size - lag) * var)))
            mean_ac += np.asarray(ac)
        mean_ac /= max(chains, 1)
        axes[dim].bar(np.arange(max_lag + 1), mean_ac, color="#6997b8")
        axes[dim].set_ylabel(name)
    axes[-1].set_xlabel("lag")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def plot_energy(energies: np.ndarray, path: Path, dpi: int = 140) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    flat = energies.reshape(-1)
    axes[0].hist(flat, bins=30, color="#aa7744", alpha=0.85)
    axes[0].set_title("Energy distribution")
    for chain in range(energies.shape[0]):
        axes[1].plot(energies[chain], linewidth=0.8)
    axes[1].set_title("Energy trace")
    axes[1].set_xlabel("draw")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def plot_posterior_predictive(observed: np.ndarray, predictive: np.ndarray, path: Path, dpi: int = 140) -> bool:
    plt = _load_pyplot()
    if plt is None or predictive.size == 0:
        return False
    obs = np.asarray(observed, dtype=float).reshape(-1)
    pred = np.asarray(predictive, dtype=float)
    mean = np.mean(pred, axis=0)
    lower = np.quantile(pred, 0.05, axis=0)
    upper = np.quantile(pred, 0.95, axis=0)
    xs = np.arange(obs.size)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(xs, lower, upper, color="#c5d9ea", alpha=0.8, label="90% predictive interval")
    ax.plot(xs, mean, color="#27617e", linewidth=1.5, label="predictive mean")
    ax.plot(xs, obs, color="#111111", linewidth=1.2, label="observed")
    ax.legend(loc="best")
    ax.set_title("Posterior predictive check")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def build_requested_plots(
    samples: np.ndarray,
    parameter_names: list[str],
    energies: np.ndarray,
    observed: np.ndarray,
    predictive: np.ndarray | None,
    plots: list[str],
    figures_dir: Path,
    dpi: int = 140,
) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for plot in plots:
        if plot == "posterior_marginals":
            out[plot] = plot_posterior_marginals(samples, parameter_names, figures_dir / "posterior_marginals.png", dpi=dpi)
        elif plot == "pairwise":
            out[plot] = plot_pairwise(samples, parameter_names, figures_dir / "pairwise.png", dpi=dpi)
        elif plot == "trace":
            out[plot] = plot_trace(samples, parameter_names, figures_dir / "trace.png", dpi=dpi)
        elif plot == "autocorrelation":
            out[plot] = plot_autocorrelation(samples, parameter_names, figures_dir / "autocorrelation.png", dpi=dpi)
        elif plot == "energy":
            out[plot] = plot_energy(energies, figures_dir / "energy.png", dpi=dpi)
        elif plot == "posterior_predictive":
            out[plot] = plot_posterior_predictive(observed, predictive if predictive is not None else np.asarray([]), figures_dir / "posterior_predictive.png", dpi=dpi)
        else:
            out[plot] = False
    return out
