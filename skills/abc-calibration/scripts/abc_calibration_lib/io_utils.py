"""Input/output helpers for the ABC calibration skill."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


class IoError(RuntimeError):
    """Data loading or serialization failure."""



def ensure_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): ensure_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value



def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ensure_jsonable(payload), indent=2) + "\n", encoding="utf-8")



def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(ensure_jsonable(record)) + "\n")



def write_samples_csv(path: Path, records: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        fieldnames = field_order or []
    else:
        fieldnames = field_order or list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: ensure_jsonable(record.get(name)) for name in fieldnames})



def stage_file(src: str | None, dst_dir: Path) -> str | None:
    if not src:
        return None
    source = Path(src).expanduser().resolve()
    if not source.exists():
        raise IoError(f"Path does not exist: {source}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    destination = dst_dir / source.name
    shutil.copy2(source, destination)
    return str(destination)



def _load_json_payload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))



def _load_csv_payload(path: Path) -> tuple[np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(1024)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        delimiter = "," if path.suffix.lower() == ".csv" else "\t"
        if has_header:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = list(reader)
            if not rows:
                raise IoError(f"Observed file is empty: {path}")
            headers = reader.fieldnames or []
            numeric_rows: list[list[float]] = []
            for row in rows:
                numeric_rows.append([float(row[h]) for h in headers])
            return np.asarray(numeric_rows, dtype=float), list(headers)
        reader = csv.reader(handle, delimiter=delimiter)
        rows = [[float(cell) for cell in row] for row in reader if row]
        if not rows:
            raise IoError(f"Observed file is empty: {path}")
        array = np.asarray(rows, dtype=float)
        names = [f"x{i}" for i in range(array.shape[1] if array.ndim > 1 else 1)]
        return array, names



def load_observed_data(path: str) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise IoError(f"Observed data path does not exist: {target}")
    suffix = target.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        array, names = _load_csv_payload(target)
        fmt = suffix.lstrip(".")
    elif suffix == ".json":
        payload = _load_json_payload(target)
        array, names = payload_to_array(payload)
        fmt = "json"
    elif suffix == ".npy":
        array = np.load(target)
        names = [f"x{i}" for i in range(array.shape[-1] if np.asarray(array).ndim > 1 else 1)]
        fmt = "npy"
    elif suffix == ".npz":
        payload = np.load(target)
        keys = sorted(payload.files)
        if len(keys) == 1:
            array = np.asarray(payload[keys[0]], dtype=float)
            names = [keys[0]]
        else:
            arrays = [np.asarray(payload[key], dtype=float).reshape(-1) for key in keys]
            array = np.concatenate(arrays)
            names = keys
        fmt = "npz"
    else:
        raise IoError(f"Unsupported observed data format: {target.suffix}")

    return {
        "path": str(target),
        "format": fmt,
        "array": np.asarray(array, dtype=float),
        "column_names": names,
        "shape": list(np.asarray(array).shape),
        "size": int(np.asarray(array).size),
    }



def payload_to_array(
    payload: Any,
    output_names: list[str] | None = None,
    output_indices: list[int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(payload, dict):
        keys = list(payload.keys())
        if output_names:
            keys = [name for name in output_names if name in payload]
        elif len(keys) == 1:
            keys = [keys[0]]
        arrays = [np.asarray(payload[key], dtype=float).reshape(-1) for key in keys]
        if not arrays:
            raise IoError("Dictionary payload does not contain any selected outputs.")
        return np.concatenate(arrays), keys

    array = np.asarray(payload, dtype=float)
    if output_indices:
        array = np.take(array, output_indices)
    if array.ndim == 0:
        return array.reshape(1), ["y0"]
    if array.ndim == 1:
        return array.reshape(-1), output_names or ["y0"]
    if array.ndim == 2 and array.shape[1] == 1:
        return array.reshape(-1), output_names or ["y0"]
    if array.ndim >= 2:
        flat = array.reshape(-1)
        if output_names:
            return flat, output_names
        return flat, [f"y{i}" for i in range(flat.shape[0])]
    raise IoError("Unsupported payload shape.")



def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
