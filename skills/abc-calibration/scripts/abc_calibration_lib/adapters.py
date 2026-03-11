"""Model execution adapters."""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class AdapterError(RuntimeError):
    """Model execution failure."""



def _load_python_callable(path: str, callable_name: str):
    target = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(target.stem, target)
    if spec is None or spec.loader is None:
        raise AdapterError(f"Could not load Python module: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, callable_name, None)
    if not callable(fn):
        raise AdapterError(f"Callable {callable_name!r} not found in {target}")
    return fn



def _run_python_callable(model_cfg: dict[str, Any], params: dict[str, float]):
    fn = _load_python_callable(model_cfg["path"], model_cfg.get("callable") or "simulate")
    style = (model_cfg.get("call_style") or "kwargs").lower()
    if style == "mapping":
        return fn(dict(params))
    if style == "positional":
        order = model_cfg.get("parameter_names") or list(params.keys())
        return fn(*[params[name] for name in order])
    return fn(**params)



def _run_command(model_cfg: dict[str, Any], params: dict[str, float], workdir: Path | None = None):
    command_template = model_cfg.get("command_template")
    if not command_template:
        raise AdapterError("model.command_template is required for command adapters.")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        params_path = root / "params.json"
        output_path = root / "output.json"
        params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
        context = {
            "params_json": shlex.quote(str(params_path)),
            "output_json": shlex.quote(str(output_path)),
            "model_path": shlex.quote(str(model_cfg.get("path", ""))),
        }
        for key, value in params.items():
            context[key] = shlex.quote(str(value))
        command = command_template.format(**context)
        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=model_cfg.get("timeout_seconds") or 300,
            cwd=str((Path(model_cfg["working_directory"]).expanduser().resolve() if model_cfg.get("working_directory") else workdir) or Path.cwd()),
            check=False,
        )
        if completed.returncode != 0:
            raise AdapterError(completed.stderr.strip() or completed.stdout.strip() or "Command adapter failed.")
        if output_path.exists():
            return json.loads(output_path.read_text(encoding="utf-8"))
        stdout = completed.stdout.strip()
        if not stdout:
            raise AdapterError("Command adapter did not produce stdout JSON or output_json.")
        return json.loads(stdout)



def simulate_model(model_cfg: dict[str, Any], params: dict[str, float], workdir: Path | None = None):
    adapter = (model_cfg.get("adapter") or "python_callable").lower()
    if adapter == "python_callable":
        return _run_python_callable(model_cfg, params)
    if adapter == "command":
        return _run_command(model_cfg, params, workdir=workdir)
    raise AdapterError(f"Unsupported model adapter: {adapter}")
