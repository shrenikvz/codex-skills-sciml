"""Input/output helpers for the PINNs skill."""

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


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _payload_to_numeric_array(payload: Any) -> np.ndarray:
    return np.asarray(payload, dtype=float)


def _table_from_mapping(mapping: dict[str, Any]) -> dict[str, np.ndarray] | None:
    if not mapping:
        return None
    lengths = set()
    out: dict[str, np.ndarray] = {}
    for key, value in mapping.items():
        array = np.asarray(value)
        if array.ndim == 0:
            return None
        array = array.reshape(-1)
        lengths.add(array.shape[0])
        out[str(key)] = array.astype(float)
    if len(lengths) != 1:
        return None
    return out


def _load_csv_payload(path: Path) -> dict[str, Any]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(1024)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        if has_header:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = list(reader)
            if not rows:
                raise IoError(f"Observed file is empty: {path}")
            headers = list(reader.fieldnames or [])
            table = {header: np.asarray([float(row[header]) for row in rows], dtype=float) for header in headers}
            array = np.column_stack([table[header] for header in headers])
            return {"array": array, "column_names": headers, "table": table}
        reader = csv.reader(handle, delimiter=delimiter)
        rows = [[float(cell) for cell in row] for row in reader if row]
        if not rows:
            raise IoError(f"Observed file is empty: {path}")
        array = np.asarray(rows, dtype=float)
        names = [f"x{i}" for i in range(array.shape[1] if array.ndim > 1 else 1)]
        return {"array": array, "column_names": names, "table": None}


def _load_json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload and all(isinstance(row, dict) for row in payload):
        keys = sorted({str(key) for row in payload for key in row})
        table = {key: np.asarray([float(row[key]) for row in payload], dtype=float) for key in keys}
        array = np.column_stack([table[key] for key in keys])
        return {"array": array, "column_names": keys, "table": table}
    if isinstance(payload, dict):
        table = _table_from_mapping(payload)
        if table is not None:
            keys = list(table.keys())
            array = np.column_stack([table[key] for key in keys])
            return {"array": array, "column_names": keys, "table": table}
    array = _payload_to_numeric_array(payload)
    names = [f"x{i}" for i in range(array.shape[-1] if array.ndim > 1 else 1)]
    return {"array": array, "column_names": names, "table": None}


def load_observed_data(path: str) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise IoError(f"Observed data path does not exist: {target}")
    suffix = target.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        loaded = _load_csv_payload(target)
        fmt = suffix.lstrip(".")
    elif suffix == ".json":
        loaded = _load_json_payload(target)
        fmt = "json"
    elif suffix == ".npy":
        array = np.load(target)
        loaded = {
            "array": np.asarray(array, dtype=float),
            "column_names": [f"x{i}" for i in range(np.asarray(array).shape[-1] if np.asarray(array).ndim > 1 else 1)],
            "table": None,
        }
        fmt = "npy"
    elif suffix == ".npz":
        payload = np.load(target)
        keys = sorted(payload.files)
        if not keys:
            raise IoError(f"Observed archive is empty: {target}")
        arrays = [np.asarray(payload[key], dtype=float).reshape(-1) for key in keys]
        loaded = {
            "array": np.column_stack(arrays),
            "column_names": keys,
            "table": {key: array for key, array in zip(keys, arrays, strict=False)},
        }
        fmt = "npz"
    else:
        raise IoError(f"Unsupported observed data format: {target.suffix}")
    array = np.asarray(loaded["array"], dtype=float)
    return {
        "path": str(target),
        "format": fmt,
        "array": array,
        "column_names": list(loaded["column_names"]),
        "table": loaded["table"],
        "shape": list(array.shape),
        "size": int(array.size),
    }


def extract_table_columns(info: dict[str, Any], columns: list[str]) -> np.ndarray:
    table = info.get("table")
    if not table:
        raise IoError("Observed data does not provide named tabular columns.")
    missing = [column for column in columns if column not in table]
    if missing:
        raise IoError(f"Observed data is missing required columns: {missing}")
    return np.column_stack([np.asarray(table[column], dtype=float).reshape(-1) for column in columns])

