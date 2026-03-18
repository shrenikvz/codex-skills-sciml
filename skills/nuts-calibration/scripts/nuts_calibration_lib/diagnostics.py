"""Sampling diagnostics for NUTS outputs."""

from __future__ import annotations

from typing import Any

import numpy as np


class DiagnosticError(RuntimeError):
    """Diagnostics failure."""


def _autocorrelation_1d(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x - np.mean(x)
    if x.size == 0:
        return np.asarray([], dtype=float)
    variance = np.var(x)
    if variance == 0:
        return np.ones(x.size, dtype=float)
    padded = np.concatenate([x, np.zeros_like(x)])
    spectrum = np.fft.rfft(padded)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum))[: x.size]
    return np.asarray(acf / (variance * x.size), dtype=float)


def split_rhat(samples: np.ndarray) -> float:
    chains, draws = samples.shape
    if chains < 2 or draws < 4:
        return float("nan")
    half = draws // 2
    split = np.concatenate([samples[:, :half], samples[:, -half:]], axis=0)
    m, n = split.shape
    chain_means = np.mean(split, axis=1)
    chain_vars = np.var(split, axis=1, ddof=1)
    within = np.mean(chain_vars)
    between = n * np.var(chain_means, ddof=1)
    var_hat = ((n - 1.0) / n) * within + between / n
    return float(np.sqrt(var_hat / within)) if within > 0 else 1.0


def effective_sample_size(samples: np.ndarray) -> float:
    chains, draws = samples.shape
    if chains < 1 or draws < 4:
        return float(chains * draws)
    acov = np.mean(np.stack([_autocorrelation_1d(samples[idx]) for idx in range(chains)], axis=0), axis=0)
    positive_sum = 0.0
    for lag in range(1, draws - 1, 2):
        pair = acov[lag] + acov[lag + 1]
        if pair < 0:
            break
        positive_sum += pair
    tau = -1.0 + 2.0 * positive_sum
    tau = max(tau, 1.0)
    return float((chains * draws) / tau)


def energy_diagnostics(energy: np.ndarray) -> dict[str, Any]:
    array = np.asarray(energy, dtype=float)
    per_chain = []
    for idx in range(array.shape[0]):
        chain = array[idx]
        variance = float(np.var(chain)) if chain.size > 1 else 0.0
        delta = np.diff(chain)
        numerator = float(np.mean(delta**2)) if delta.size else 0.0
        ebfmi = numerator / variance if variance > 0 else float("nan")
        per_chain.append({"chain": idx, "ebfmi": ebfmi})
    valid = [item["ebfmi"] for item in per_chain if np.isfinite(item["ebfmi"])]
    return {
        "per_chain": per_chain,
        "min_ebfmi": float(min(valid)) if valid else None,
        "mean_energy": float(np.mean(array)) if array.size else None,
        "std_energy": float(np.std(array)) if array.size else None,
    }


def summarize_diagnostics(
    samples: np.ndarray,
    parameter_names: list[str],
    info: dict[str, np.ndarray],
    max_tree_depth: int,
) -> dict[str, Any]:
    if samples.ndim != 3:
        raise DiagnosticError("Samples must have shape (chains, draws, parameters).")
    chains, draws, dim = samples.shape
    parameters: dict[str, Any] = {}
    for idx, name in enumerate(parameter_names):
        series = samples[:, :, idx]
        parameters[name] = {
            "rhat": split_rhat(series),
            "ess": effective_sample_size(series),
            "mean": float(np.mean(series)),
            "std": float(np.std(series, ddof=1)) if series.size > 1 else 0.0,
        }
    acceptance = np.asarray(info.get("acceptance_rate", np.zeros((chains, draws))), dtype=float)
    energy = np.asarray(info.get("energy", np.zeros((chains, draws))), dtype=float)
    divergences = np.asarray(info.get("is_divergent", np.zeros((chains, draws), dtype=bool)), dtype=bool)
    tree_depth = np.asarray(info.get("tree_depth", np.zeros((chains, draws))), dtype=int)
    warnings = []
    bad_rhat = [name for name, payload in parameters.items() if np.isfinite(payload["rhat"]) and payload["rhat"] > 1.01]
    if bad_rhat:
        warnings.append(f"R-hat exceeds 1.01 for: {', '.join(bad_rhat)}.")
    low_ess = [name for name, payload in parameters.items() if payload["ess"] < 0.1 * chains * draws]
    if low_ess:
        warnings.append(f"Effective sample size is low for: {', '.join(low_ess)}.")
    divergence_count = int(np.sum(divergences))
    if divergence_count:
        warnings.append("Divergent transitions detected. Increase target acceptance, re-scale parameters, or revisit transforms.")
    saturation_fraction = float(np.mean(tree_depth >= max_tree_depth)) if tree_depth.size else 0.0
    if saturation_fraction > 0.01:
        warnings.append("The sampler is saturating max tree depth. Increase max_tree_depth or improve posterior geometry.")
    accept_mean = float(np.mean(acceptance)) if acceptance.size else 0.0
    if accept_mean < 0.6 or accept_mean > 0.98:
        warnings.append("Average acceptance rate is outside the usual operating range.")
    energy_report = energy_diagnostics(energy)
    if energy_report.get("min_ebfmi") is not None and energy_report["min_ebfmi"] < 0.3:
        warnings.append("Energy transitions are poor (low E-BFMI). Reparameterization may help.")
    return {
        "parameters": parameters,
        "acceptance_rate": {
            "mean": accept_mean,
            "min": float(np.min(acceptance)) if acceptance.size else None,
            "max": float(np.max(acceptance)) if acceptance.size else None,
        },
        "divergences": {
            "count": divergence_count,
            "fraction": float(np.mean(divergences)) if divergences.size else 0.0,
        },
        "tree_depth": {
            "mean": float(np.mean(tree_depth)) if tree_depth.size else None,
            "max": int(np.max(tree_depth)) if tree_depth.size else None,
            "saturation_fraction": saturation_fraction,
        },
        "energy": energy_report,
        "warnings": warnings,
    }

