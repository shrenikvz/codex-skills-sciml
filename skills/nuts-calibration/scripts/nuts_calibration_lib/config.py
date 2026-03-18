"""Configuration helpers for the NUTS calibration skill."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

CONFIG_VERSION = 1

DEFAULT_CONFIG: dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "objective": {
        "text": "",
        "observed_path": "",
        "observed_format": "auto",
        "observed_output_names": [],
        "observed_output_indices": [],
        "noise_structure": "auto",
    },
    "model": {
        "adapter": "auto",
        "path": "",
        "callable": "simulate",
        "call_style": "auto",
        "command_template": None,
        "working_directory": None,
        "runtime_hint": None,
        "parameter_names": [],
        "parameter_defaults": {},
        "parameter_constraints": {},
        "output_names": [],
        "observed_output_names": [],
        "observed_output_indices": [],
        "supports_batch": False,
        "stochastic": None,
        "differentiable": "auto",
        "gradient_strategy": "auto",
        "timeout_seconds": 300,
    },
    "priors": {},
    "likelihood": {
        "name": "auto",
        "params": {},
        "custom_python_path": None,
        "custom_callable": None,
        "custom_command_template": None,
    },
    "transformations": {
        "mode": "auto",
        "parameters": {},
    },
    "scaling": {
        "enabled": None,
        "mode": "auto",
        "center": [],
        "scale": [],
    },
    "algorithm": {
        "name": "nuts",
        "backend": "blackjax",
        "random_seed": 7,
        "warmup": {
            "num_steps": None,
            "target_acceptance": None,
            "mass_matrix": "auto",
            "initial_step_size": None,
        },
        "sampling": {
            "num_samples": None,
            "num_chains": None,
            "max_tree_depth": None,
            "thin": 1,
            "store_warmup": False,
        },
    },
    "compute": {
        "device_preference": "auto",
        "chain_method": "auto",
        "jit": "auto",
        "enable_x64": False,
    },
    "posterior_predictive": {
        "enabled": True,
        "draws": 200,
    },
    "visualization": {
        "enabled": False,
        "plots": [],
        "dpi": 140,
    },
    "backend_options": {
        "blackjax": {
            "progress_bar": False,
        },
        "numpyro": {},
        "pymc": {},
        "stan": {},
        "tensorflow_probability": {},
    },
    "output": {
        "results_dir": "results",
        "artifacts_dir": "artifacts",
    },
}


class ConfigError(RuntimeError):
    """Configuration failure."""


def clone_default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def migrate_config(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = deep_merge(DEFAULT_CONFIG, payload)
    migrated["config_version"] = CONFIG_VERSION
    return migrated


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a JSON object.")
    cfg = migrate_config(raw)
    validate_config(cfg)
    return cfg


def save_config(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def validate_config(cfg: dict[str, Any]) -> None:
    objective = cfg.get("objective", {})
    if not isinstance(objective.get("observed_path"), str):
        raise ConfigError("config.objective.observed_path must be a string.")
    model = cfg.get("model")
    if not isinstance(model, dict):
        raise ConfigError("config.model must be an object.")
    parameter_names = model.get("parameter_names", [])
    if not isinstance(parameter_names, list) or not all(isinstance(item, str) for item in parameter_names):
        raise ConfigError("config.model.parameter_names must be a list of strings.")
    if not isinstance(cfg.get("priors"), dict):
        raise ConfigError("config.priors must be an object keyed by parameter name.")
    algorithm = cfg.get("algorithm", {})
    if algorithm.get("name") not in {"nuts", "hmc"}:
        raise ConfigError("config.algorithm.name must be 'nuts' or 'hmc'.")
    if algorithm.get("backend") not in {"blackjax", "numpyro", "pymc", "stan", "tensorflow_probability"}:
        raise ConfigError("Unsupported config.algorithm.backend.")
    warmup = algorithm.get("warmup", {})
    num_steps = warmup.get("num_steps")
    if num_steps is not None and (not isinstance(num_steps, int) or num_steps <= 0):
        raise ConfigError("config.algorithm.warmup.num_steps must be a positive integer or null.")
    target_acceptance = warmup.get("target_acceptance")
    if target_acceptance is not None and (
        not isinstance(target_acceptance, (int, float)) or not (0 < float(target_acceptance) < 1)
    ):
        raise ConfigError("config.algorithm.warmup.target_acceptance must be in (0, 1).")
    sampling = algorithm.get("sampling", {})
    for key in ["num_samples", "num_chains", "max_tree_depth", "thin"]:
        value = sampling.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ConfigError(f"config.algorithm.sampling.{key} must be a positive integer or null.")
    visualization = cfg.get("visualization", {})
    if not isinstance(visualization.get("plots", []), list):
        raise ConfigError("config.visualization.plots must be a list.")
    timeout = model.get("timeout_seconds")
    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        raise ConfigError("config.model.timeout_seconds must be a positive integer or null.")


def infer_default_hyperparameters(parameter_count: int, observed_size: int, model_complexity: str = "standard") -> dict[str, Any]:
    dim = max(1, int(parameter_count or 1))
    obs = max(1, int(observed_size or 1))
    heavy = model_complexity in {"external", "stochastic", "timeseries", "high_dimensional"}
    warmup = max(800 if heavy else 600, 100 * dim)
    samples = max(1000 if not heavy else 1200, 200 * dim)
    chains = 4 if dim <= 20 else 6
    target = 0.9 if heavy or dim > 8 else 0.8
    tree_depth = 12 if heavy or obs > 1000 else 10
    mass_matrix = "dense" if dim <= 8 else "diagonal"
    posterior_predictive_draws = min(500, max(100, 20 * dim))
    return {
        "warmup": {
            "num_steps": int(warmup),
            "target_acceptance": float(target),
            "mass_matrix": mass_matrix,
            "initial_step_size": None,
        },
        "sampling": {
            "num_samples": int(samples),
            "num_chains": int(chains),
            "max_tree_depth": int(tree_depth),
            "thin": 1,
        },
        "posterior_predictive": {
            "draws": int(min(posterior_predictive_draws, max(50, obs))),
        },
    }


def resolve_runtime_hyperparameters(
    cfg: dict[str, Any],
    observed_size: int | None = None,
    model_complexity: str = "standard",
) -> dict[str, Any]:
    parameter_names = cfg.get("model", {}).get("parameter_names", [])
    defaults = infer_default_hyperparameters(len(parameter_names), observed_size or 1, model_complexity=model_complexity)
    warmup = cfg.setdefault("algorithm", {}).setdefault("warmup", {})
    sampling = cfg.setdefault("algorithm", {}).setdefault("sampling", {})
    for key, value in defaults["warmup"].items():
        if warmup.get(key) in {None, "", 0, "auto"}:
            warmup[key] = value
    for key, value in defaults["sampling"].items():
        if sampling.get(key) in {None, "", 0, "auto"}:
            sampling[key] = value
    posterior_predictive = cfg.setdefault("posterior_predictive", {})
    if posterior_predictive.get("draws") in {None, "", 0}:
        posterior_predictive["draws"] = defaults["posterior_predictive"]["draws"]
    return {
        "warmup": warmup,
        "sampling": sampling,
        "posterior_predictive": posterior_predictive,
    }

