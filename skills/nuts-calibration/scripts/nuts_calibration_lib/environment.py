"""Environment probing and backend recommendation helpers."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from functools import lru_cache
from typing import Any


def _safe_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_module(module_name: str, probe_code: str) -> dict[str, Any]:
    installed = importlib.util.find_spec(module_name) is not None
    version = _package_version(module_name)
    if not installed:
        return {
            "installed": False,
            "available": False,
            "version": None,
            "devices": [],
            "gpu_count": 0,
            "error": "not_installed",
        }
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
            env=_safe_env(),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "installed": True,
            "available": False,
            "version": version,
            "devices": [],
            "gpu_count": 0,
            "error": str(exc),
        }
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "").strip() or f"{module_name} probe failed"
        return {
            "installed": True,
            "available": False,
            "version": version,
            "devices": [],
            "gpu_count": 0,
            "error": error,
        }
    try:
        payload = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        payload = {}
    return {
        "installed": True,
        "available": True,
        "version": version,
        "devices": list(payload.get("devices", [])),
        "gpu_count": int(payload.get("gpu_count", 0)),
        "error": None,
    }


def _probe_jax() -> dict[str, Any]:
    code = """
import json
import jax
devices = jax.devices()
payload = {
    "gpu_count": sum(1 for device in devices if device.platform == "gpu"),
    "devices": [f"{device.platform}:{device.id}" for device in devices],
}
print(json.dumps(payload))
"""
    return _probe_module("jax", code)


def _probe_blackjax() -> dict[str, Any]:
    code = """
import json
import blackjax
payload = {"devices": [], "gpu_count": 0}
print(json.dumps(payload))
"""
    return _probe_module("blackjax", code)


def _probe_pymc() -> dict[str, Any]:
    code = """
import json
import pymc
payload = {"devices": ["cpu"], "gpu_count": 0}
print(json.dumps(payload))
"""
    return _probe_module("pymc", code)


def _probe_numpyro() -> dict[str, Any]:
    code = """
import json
import numpyro
payload = {"devices": [], "gpu_count": 0}
print(json.dumps(payload))
"""
    return _probe_module("numpyro", code)


def _probe_tfp() -> dict[str, Any]:
    code = """
import json
import tensorflow_probability
payload = {"devices": [], "gpu_count": 0}
print(json.dumps(payload))
"""
    return _probe_module("tensorflow_probability", code)


def _probe_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def peek_environment() -> dict[str, Any]:
    report = {
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax": {
            "installed": importlib.util.find_spec("jax") is not None,
            "available": False,
            "version": _package_version("jax"),
            "devices": [],
            "gpu_count": 0,
            "error": None,
        },
        "blackjax": {
            "installed": importlib.util.find_spec("blackjax") is not None,
            "available": False,
            "version": _package_version("blackjax"),
            "devices": [],
            "gpu_count": 0,
            "error": None,
        },
        "numpyro": {
            "installed": importlib.util.find_spec("numpyro") is not None,
            "available": False,
            "version": _package_version("numpyro"),
            "devices": [],
            "gpu_count": 0,
            "error": None,
        },
        "pymc": {
            "installed": importlib.util.find_spec("pymc") is not None,
            "available": False,
            "version": _package_version("pymc"),
            "devices": [],
            "gpu_count": 0,
            "error": None,
        },
        "tensorflow_probability": {
            "installed": importlib.util.find_spec("tensorflow_probability") is not None,
            "available": False,
            "version": _package_version("tensorflow_probability"),
            "devices": [],
            "gpu_count": 0,
            "error": None,
        },
        "matplotlib": _probe_matplotlib(),
    }
    report["default_backend"] = recommend_backend(report)["backend"]
    return report


@lru_cache(maxsize=1)
def probe_environment() -> dict[str, Any]:
    report = {
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jax": _probe_jax(),
        "blackjax": _probe_blackjax(),
        "numpyro": _probe_numpyro(),
        "pymc": _probe_pymc(),
        "tensorflow_probability": _probe_tfp(),
        "matplotlib": _probe_matplotlib(),
    }
    report["default_backend"] = recommend_backend(report)["backend"]
    return report


def recommend_backend(environment: dict[str, Any], requested: str | None = None) -> dict[str, Any]:
    choice = str(requested or "auto").lower()
    if choice == "blackjax":
        ready = bool(environment.get("jax", {}).get("available")) and bool(environment.get("blackjax", {}).get("available"))
        return {"backend": "blackjax", "reason": "user_preference", "ready": ready}
    if choice in {"numpyro", "pymc", "stan", "tensorflow_probability"}:
        ready = bool(environment.get(choice, {}).get("available"))
        return {"backend": choice, "reason": "user_preference", "ready": ready}
    if bool(environment.get("jax", {}).get("available")) and bool(environment.get("blackjax", {}).get("available")):
        return {"backend": "blackjax", "reason": "default_nuts_backend", "ready": True}
    if bool(environment.get("numpyro", {}).get("available")) and bool(environment.get("jax", {}).get("available")):
        return {"backend": "numpyro", "reason": "jax_available_without_blackjax", "ready": True}
    if bool(environment.get("pymc", {}).get("available")):
        return {"backend": "pymc", "reason": "fallback_backend", "ready": True}
    if bool(environment.get("tensorflow_probability", {}).get("available")):
        return {"backend": "tensorflow_probability", "reason": "fallback_backend", "ready": True}
    return {"backend": None, "reason": "no_available_backend", "ready": False}

