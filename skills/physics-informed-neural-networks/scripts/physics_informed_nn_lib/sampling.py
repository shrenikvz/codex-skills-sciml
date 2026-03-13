"""Sampling utilities for collocation, boundary, and evaluation points."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class SamplingError(RuntimeError):
    """Sampling failure."""


def _bounds(problem: dict[str, Any], override: dict[str, dict[str, float]] | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    domains = override or problem.get("domains", {})
    names = list(problem.get("independent_variables", []))
    if not names:
        names = list(domains.keys())
    low = np.asarray([float(domains[name]["min"]) for name in names], dtype=float)
    high = np.asarray([float(domains[name]["max"]) for name in names], dtype=float)
    return names, low, high


def _uniform_points(low: np.ndarray, high: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(low=low, high=high, size=(int(n), low.shape[0]))


def _latin_hypercube(low: np.ndarray, high: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    dim = low.shape[0]
    cut = np.linspace(0.0, 1.0, int(n) + 1)
    u = rng.uniform(size=(int(n), dim))
    a = cut[:n]
    b = cut[1 : n + 1]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    samples = np.zeros_like(rdpoints)
    for j in range(dim):
        order = rng.permutation(int(n))
        samples[:, j] = rdpoints[order, j]
    return low + samples * (high - low)


def _sobol_points(low: np.ndarray, high: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    try:
        from scipy.stats import qmc  # type: ignore

        engine = qmc.Sobol(d=low.shape[0], scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
        samples = engine.random(int(n))
        return qmc.scale(samples, low, high)
    except Exception:  # noqa: BLE001
        return _latin_hypercube(low, high, n, rng)


def sample_domain_points(
    problem: dict[str, Any],
    n: int,
    strategy: str,
    seed: int,
    fixed_location: dict[str, Any] | None = None,
    bounds_override: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    names, low, high = _bounds(problem, override=bounds_override)
    rng = np.random.default_rng(int(seed))
    strategy_name = str(strategy or "uniform").lower()
    if strategy_name in {"adaptive", "residual_adaptive"}:
        strategy_name = "uniform"
    if strategy_name == "uniform":
        points = _uniform_points(low, high, n, rng)
    elif strategy_name == "latin_hypercube":
        points = _latin_hypercube(low, high, n, rng)
    elif strategy_name == "sobol":
        points = _sobol_points(low, high, n, rng)
    else:
        raise SamplingError(f"Unsupported sampling strategy: {strategy}")
    fixed = fixed_location or {}
    for index, name in enumerate(names):
        if name not in fixed:
            continue
        value = fixed[name]
        if value == "min":
            points[:, index] = low[index]
        elif value == "max":
            points[:, index] = high[index]
        else:
            points[:, index] = float(value)
    return points


def iter_subdomain_bounds(problem: dict[str, Any], count: int) -> list[dict[str, dict[str, float]]]:
    total = max(1, int(count))
    if total == 1:
        return [dict(problem.get("domains", {}))]
    names, low, high = _bounds(problem)
    first = names[0]
    edges = np.linspace(low[0], high[0], total + 1)
    bounds: list[dict[str, dict[str, float]]] = []
    for idx in range(total):
        local = {name: dict(problem["domains"][name]) for name in names}
        local[first] = {"min": float(edges[idx]), "max": float(edges[idx + 1])}
        bounds.append(local)
    return bounds


def build_prediction_points(problem: dict[str, Any], n: int) -> tuple[np.ndarray, list[int] | None]:
    names, low, high = _bounds(problem)
    dim = len(names)
    if dim == 1:
        grid = np.linspace(low[0], high[0], int(n), dtype=float).reshape(-1, 1)
        return grid, [int(n)]
    if dim == 2:
        side = max(8, int(math.sqrt(int(n))))
        x0 = np.linspace(low[0], high[0], side, dtype=float)
        x1 = np.linspace(low[1], high[1], side, dtype=float)
        mesh = np.meshgrid(x0, x1, indexing="ij")
        points = np.column_stack([mesh[0].reshape(-1), mesh[1].reshape(-1)])
        return points, [side, side]
    rng = np.random.default_rng(12345)
    return _uniform_points(low, high, int(n), rng), None

