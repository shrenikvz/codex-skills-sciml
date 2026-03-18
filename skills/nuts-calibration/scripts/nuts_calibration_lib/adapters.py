"""Model execution adapters."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class AdapterError(RuntimeError):
    """Model adapter failure."""


def _load_python_callable(path: Path, callable_name: str):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise AdapterError(f"Could not load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, callable_name, None)
    if not callable(fn):
        raise AdapterError(f"Callable {callable_name!r} not found in {path}")
    return fn


def simulate_model(model_cfg: dict[str, Any], params: dict[str, float], workdir: Path) -> Any:
    adapter = model_cfg.get("adapter")
    if adapter == "python_callable":
        path = model_cfg.get("path")
        if not path:
            raise AdapterError("Python model adapter requires model.path.")
        model_path = Path(path)
        if not model_path.is_absolute():
            model_path = (workdir / model_path).resolve()
        fn = _load_python_callable(model_path, model_cfg.get("callable") or "simulate")
        call_style = model_cfg.get("call_style") or "kwargs"
        if call_style == "mapping":
            return fn(params)
        if call_style == "positional":
            ordered = [params[name] for name in model_cfg.get("parameter_names", [])]
            return fn(*ordered)
        return fn(**params)
    if adapter == "command":
        template = model_cfg.get("command_template")
        path = model_cfg.get("path")
        if not template:
            raise AdapterError("Command adapter requires model.command_template.")
        model_path = ""
        if path:
            target = Path(path)
            model_path = str((workdir / target).resolve()) if not target.is_absolute() else str(target)
        working_directory = model_cfg.get("working_directory")
        cwd = (workdir / working_directory).resolve() if working_directory and not Path(working_directory).is_absolute() else Path(working_directory or workdir)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            params_path = root / "params.json"
            output_path = root / "output.json"
            params_path.write_text(json.dumps(params), encoding="utf-8")
            command = template.format(
                model_path=model_path,
                params_json=str(params_path),
                output_json=str(output_path),
                **{name: value for name, value in params.items()},
            )
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                check=False,
                timeout=model_cfg.get("timeout_seconds", 300),
            )
            if completed.returncode != 0:
                raise AdapterError(completed.stderr.strip() or completed.stdout.strip() or "Model command failed.")
            if output_path.exists():
                return json.loads(output_path.read_text(encoding="utf-8"))
            if completed.stdout.strip():
                return json.loads(completed.stdout)
            raise AdapterError("Model command produced no JSON output.")
    raise AdapterError(f"Unsupported model adapter: {adapter}")

