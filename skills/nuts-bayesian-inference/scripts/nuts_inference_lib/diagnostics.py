"""Diagnostics for multi-chain MCMC output."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _split_chains(samples: np.ndarray) -> np.ndarray:
    chains, draws, dims = samples.shape
    if draws < 4:
        return samples
    half = draws // 2
    return np.concatenate([samples[:, :half, :], samples[:, -half:, :]], axis=0)


def _rhat(samples: np.ndarray) -> np.ndarray:
    chains, draws, _ = samples.shape
    if chains < 2 or draws < 2:
        return np.full(samples.shape[-1], np.nan)
    chain_means = np.mean(samples, axis=1)
    chain_vars = np.var(samples, axis=1, ddof=1)
    between = draws * np.var(chain_means, axis=0, ddof=1)
    within = np.mean(chain_vars, axis=0)
    var_hat = ((draws - 1) / draws) * within + between / draws
    return np.sqrt(np.maximum(var_hat / np.maximum(within, 1e-12), 0.0))


def _autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    centered = series - np.mean(series)
    var = np.var(centered)
    if var <= 0:
        return np.ones(max_lag + 1)
    corr = [1.0]
    for lag in range(1, max_lag + 1):
        if lag >= series.size:
            corr.append(0.0)
            continue
        value = np.dot(centered[:-lag], centered[lag:]) / ((series.size - lag) * var)
        corr.append(float(value))
    return np.asarray(corr, dtype=float)


def _ess(samples: np.ndarray) -> np.ndarray:
    chains, draws, dims = samples.shape
    if chains == 0 or draws == 0:
        return np.zeros(dims)
    max_lag = min(draws - 1, 250)
    out = np.zeros(dims, dtype=float)
    for dim in range(dims):
        acov = np.mean([_autocorrelation(samples[chain, :, dim], max_lag) for chain in range(chains)], axis=0)
        positive_sum = 0.0
        for lag in range(1, max_lag, 2):
            pair = acov[lag] + (acov[lag + 1] if lag + 1 < acov.size else 0.0)
            if pair <= 0:
                break
            positive_sum += pair
        tau = max(1.0, -1.0 + 2.0 * positive_sum)
        out[dim] = chains * draws / tau
    return out


def summarize_posterior(samples: np.ndarray, parameter_names: list[str]) -> dict[str, Any]:
    flat = samples.reshape(-1, samples.shape[-1]) if samples.size else np.zeros((0, len(parameter_names)))
    summary = {"draws": int(flat.shape[0]), "parameters": {}}
    for idx, name in enumerate(parameter_names):
        values = flat[:, idx] if flat.size else np.asarray([], dtype=float)
        if values.size == 0:
            summary["parameters"][name] = {}
            continue
        summary["parameters"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "ci_80": [float(np.quantile(values, 0.1)), float(np.quantile(values, 0.9))],
            "ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def credible_intervals(samples: np.ndarray, parameter_names: list[str]) -> dict[str, Any]:
    flat = samples.reshape(-1, samples.shape[-1]) if samples.size else np.zeros((0, len(parameter_names)))
    intervals = {}
    for idx, name in enumerate(parameter_names):
        values = flat[:, idx]
        if values.size == 0:
            intervals[name] = {}
            continue
        intervals[name] = {
            "50": [float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))],
            "80": [float(np.quantile(values, 0.1)), float(np.quantile(values, 0.9))],
            "95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
    return intervals


def energy_diagnostics(energies: np.ndarray) -> dict[str, Any]:
    energies = np.asarray(energies, dtype=float)
    if energies.size == 0:
        return {}
    bfmi = []
    for chain_energy in energies:
        denom = np.var(chain_energy)
        if denom <= 0:
            bfmi.append(float("nan"))
            continue
        bfmi.append(float(np.mean(np.diff(chain_energy) ** 2) / denom))
    return {
        "bfmi_per_chain": bfmi,
        "min_bfmi": float(np.nanmin(bfmi)),
        "mean_energy": float(np.mean(energies)),
        "std_energy": float(np.std(energies)),
    }


def summarize_acceptance(acceptance_rate: np.ndarray, divergences: np.ndarray, tree_depth: np.ndarray, max_tree_depth: int) -> dict[str, Any]:
    acceptance_rate = np.asarray(acceptance_rate, dtype=float)
    divergences = np.asarray(divergences, dtype=bool)
    tree_depth = np.asarray(tree_depth, dtype=float)
    return {
        "mean_acceptance_rate": float(np.mean(acceptance_rate)),
        "min_acceptance_rate": float(np.min(acceptance_rate)),
        "max_acceptance_rate": float(np.max(acceptance_rate)),
        "divergent_transitions": int(np.sum(divergences)),
        "divergence_fraction": float(np.mean(divergences)),
        "tree_depth_saturation_count": int(np.sum(tree_depth >= max_tree_depth)),
        "tree_depth_saturation_fraction": float(np.mean(tree_depth >= max_tree_depth)),
    }


def build_diagnostics(
    samples: np.ndarray,
    parameter_names: list[str],
    energies: np.ndarray,
    divergences: np.ndarray,
    acceptance_rate: np.ndarray,
    tree_depth: np.ndarray,
    max_tree_depth: int,
) -> dict[str, Any]:
    split = _split_chains(np.asarray(samples, dtype=float))
    rhat = _rhat(split)
    ess = _ess(split)
    energy = energy_diagnostics(energies)
    acceptance = summarize_acceptance(acceptance_rate, divergences, tree_depth, max_tree_depth)
    warnings = []
    for idx, name in enumerate(parameter_names):
        if np.isfinite(rhat[idx]) and rhat[idx] > 1.01:
            warnings.append(f"R-hat for {name} is {rhat[idx]:.3f}, which suggests incomplete mixing.")
        if ess[idx] < 100:
            warnings.append(f"Effective sample size for {name} is {ess[idx]:.1f}, which is low.")
    if acceptance["divergent_transitions"] > 0:
        warnings.append("Divergent transitions were detected. Increase target_acceptance_rate or reparameterize the model.")
    if acceptance["tree_depth_saturation_count"] > 0:
        warnings.append("Some transitions saturated the maximum tree depth. Increase max_tree_depth or improve geometry.")
    if energy and math.isfinite(energy.get("min_bfmi", float("nan"))) and energy["min_bfmi"] < 0.3:
        warnings.append("Low BFMI suggests poor energy exploration.")
    return {
        "rhat": {name: float(rhat[idx]) for idx, name in enumerate(parameter_names)},
        "effective_sample_size": {name: float(ess[idx]) for idx, name in enumerate(parameter_names)},
        "energy": energy,
        "acceptance": acceptance,
        "warnings": warnings,
    }
