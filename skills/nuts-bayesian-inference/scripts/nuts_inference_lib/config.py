"""Configuration helpers for the NUTS Bayesian inference skill."""

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
        "output_names": [],
        "observed_output_names": [],
        "observed_output_indices": [],
        "stochastic": None,
        "timeout_seconds": 300,
        "gradient_strategy": "auto",
        "gradient_step_size": 1e-4,
    },
    "priors": {},
    "likelihood": {
        "family": "auto",
        "params": {
            "sigma": None,
            "sigma_parameter": None,
            "df": 5.0,
            "total_count": None,
            "dispersion": None,
            "dispersion_parameter": None,
        },
        "custom_python_path": None,
        "custom_callable": None,
        "custom_command_template": None,
    },
    "scaling": {
        "enabled": None,
        "mode": "auto",
    },
    "sampler": {
        "algorithm": "nuts",
        "backend": "blackjax",
        "random_seed": 7,
        "num_warmup": None,
        "num_samples": None,
        "num_chains": None,
        "target_acceptance_rate": None,
        "max_tree_depth": None,
        "mass_matrix": "diagonal",
        "step_size": None,
    },
    "compute": {
        "device_preference": "auto",
        "parallel_mode": "auto",
        "enable_x64": False,
    },
    "posterior_predictive": {
        "enabled": True,
        "draws": 100,
    },
    "visualization": {
        "enabled": False,
        "plots": [],
        "dpi": 140,
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
    model = cfg.get("model")
    if not isinstance(model, dict):
        raise ConfigError("config.model must be an object.")
    priors = cfg.get("priors")
    if not isinstance(priors, dict):
        raise ConfigError("config.priors must be an object keyed by parameter name.")
    parameter_names = model.get("parameter_names", [])
    if not isinstance(parameter_names, list) or not all(isinstance(item, str) for item in parameter_names):
        raise ConfigError("config.model.parameter_names must be a list of strings.")
    observed_path = cfg.get("objective", {}).get("observed_path")
    if not isinstance(observed_path, str) or not observed_path:
        raise ConfigError("config.objective.observed_path must be a non-empty string.")
    sampler = cfg.get("sampler", {})
    for key in ["num_warmup", "num_samples", "num_chains", "max_tree_depth"]:
        value = sampler.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ConfigError(f"config.sampler.{key} must be a positive integer or null.")
    target = sampler.get("target_acceptance_rate")
    if target is not None and (not isinstance(target, (int, float)) or not (0.5 <= float(target) < 1.0)):
        raise ConfigError("config.sampler.target_acceptance_rate must be in [0.5, 1.0).")
    timeout = model.get("timeout_seconds")
    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        raise ConfigError("config.model.timeout_seconds must be a positive integer or null.")


def infer_default_hyperparameters(parameter_count: int, observed_size: int) -> dict[str, int | float | str]:
    dim = max(1, int(parameter_count or 1))
    obs = max(1, int(observed_size or 1))
    warmup = max(800, 100 * dim)
    samples = max(1000, 150 * dim)
    chains = 4 if dim <= 32 else 6
    target_acceptance = 0.8 if dim <= 8 else 0.9
    max_tree_depth = 10 if obs <= 512 else 12
    mass_matrix = "diagonal" if dim > 15 else "dense"
    return {
        "num_warmup": int(warmup),
        "num_samples": int(samples),
        "num_chains": int(chains),
        "target_acceptance_rate": float(target_acceptance),
        "max_tree_depth": int(max_tree_depth),
        "mass_matrix": mass_matrix,
    }


def resolve_runtime_hyperparameters(cfg: dict[str, Any], observed_size: int | None = None) -> dict[str, Any]:
    sampler = cfg.setdefault("sampler", {})
    defaults = infer_default_hyperparameters(
        len(cfg.get("model", {}).get("parameter_names", [])) + len(cfg.get("likelihood", {}).get("parameter_names", [])),
        observed_size or 1,
    )
    for key, value in defaults.items():
        if sampler.get(key) in {None, "", 0}:
            sampler[key] = value
    return sampler
