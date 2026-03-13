"""Visualization helpers for PINN outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def generate_figures(
    requested_plots: list[str],
    output_dir: Path,
    history: list[dict[str, Any]],
    predictions: dict[str, Any],
    residual_grid: dict[str, Any] | None,
    analytical: dict[str, Any] | None,
    dpi: int = 140,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"generated": [], "skipped": [{"plot": plot, "reason": str(exc)} for plot in requested_plots]}
    generated: list[str] = []
    skipped: list[dict[str, str]] = []
    fields = predictions.get("outputs", {})
    coords = predictions.get("coordinates", {})
    shape = predictions.get("shape")

    for plot in requested_plots:
        try:
            if plot == "loss_curves" and history:
                fig, ax = plt.subplots(figsize=(6, 4))
                epochs = [record["epoch"] for record in history]
                for key in ["total_loss", "pde_loss", "bc_loss", "ic_loss", "data_loss"]:
                    values = [record.get(key) for record in history if record.get(key) is not None]
                    if values:
                        ax.semilogy(epochs[: len(values)], values, label=key)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.legend()
                path = output_dir / "loss_curves.png"
                fig.tight_layout()
                fig.savefig(path, dpi=dpi)
                plt.close(fig)
                generated.append(str(path))
            elif plot == "solution_field" and fields:
                field_name = sorted(fields.keys())[0]
                values = np.asarray(fields[field_name], dtype=float)
                if shape and len(shape) == 2 and len(coords) >= 2:
                    names = list(coords.keys())[:2]
                    x = np.asarray(coords[names[0]], dtype=float).reshape(shape)
                    y = np.asarray(coords[names[1]], dtype=float).reshape(shape)
                    z = values.reshape(shape)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    pcm = ax.pcolormesh(x, y, z, shading="auto")
                    fig.colorbar(pcm, ax=ax, label=field_name)
                    ax.set_xlabel(names[0])
                    ax.set_ylabel(names[1])
                else:
                    name = list(coords.keys())[0]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(np.asarray(coords[name], dtype=float), values)
                    ax.set_xlabel(name)
                    ax.set_ylabel(field_name)
                path = output_dir / "solution_field.png"
                fig.tight_layout()
                fig.savefig(path, dpi=dpi)
                plt.close(fig)
                generated.append(str(path))
            elif plot == "residual_heatmap" and residual_grid:
                if residual_grid.get("shape") and len(residual_grid["shape"]) == 2 and len(residual_grid.get("coordinates", {})) >= 2:
                    names = list(residual_grid["coordinates"].keys())[:2]
                    x = np.asarray(residual_grid["coordinates"][names[0]], dtype=float).reshape(residual_grid["shape"])
                    y = np.asarray(residual_grid["coordinates"][names[1]], dtype=float).reshape(residual_grid["shape"])
                    z = np.asarray(residual_grid["values"], dtype=float).reshape(residual_grid["shape"])
                    fig, ax = plt.subplots(figsize=(6, 4))
                    pcm = ax.pcolormesh(x, y, z, shading="auto")
                    fig.colorbar(pcm, ax=ax, label="Residual magnitude")
                    ax.set_xlabel(names[0])
                    ax.set_ylabel(names[1])
                    path = output_dir / "residual_heatmap.png"
                    fig.tight_layout()
                    fig.savefig(path, dpi=dpi)
                    plt.close(fig)
                    generated.append(str(path))
                else:
                    skipped.append({"plot": plot, "reason": "Residual grid is not 2D."})
            elif plot == "analytical_comparison" and analytical:
                fig, ax = plt.subplots(figsize=(6, 4))
                name = list(coords.keys())[0]
                field_name = sorted(fields.keys())[0]
                ax.plot(np.asarray(coords[name], dtype=float), np.asarray(fields[field_name], dtype=float), label="prediction")
                ax.plot(np.asarray(coords[name], dtype=float), np.asarray(analytical["values"], dtype=float), label="analytical")
                ax.set_xlabel(name)
                ax.set_ylabel(field_name)
                ax.legend()
                path = output_dir / "analytical_comparison.png"
                fig.tight_layout()
                fig.savefig(path, dpi=dpi)
                plt.close(fig)
                generated.append(str(path))
            elif plot == "time_evolution" and fields and "t" in coords and len(coords) >= 2:
                spatial = [name for name in coords if name != "t"]
                if spatial and shape and len(shape) == 2:
                    field_name = sorted(fields.keys())[0]
                    x = np.asarray(coords[spatial[0]], dtype=float).reshape(shape)
                    t = np.asarray(coords["t"], dtype=float).reshape(shape)
                    z = np.asarray(fields[field_name], dtype=float).reshape(shape)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    for idx in np.linspace(0, shape[1] - 1, num=min(4, shape[1]), dtype=int):
                        ax.plot(x[:, idx], z[:, idx], label=f"t={t[0, idx]:.3f}")
                    ax.set_xlabel(spatial[0])
                    ax.set_ylabel(field_name)
                    ax.legend()
                    path = output_dir / "time_evolution.png"
                    fig.tight_layout()
                    fig.savefig(path, dpi=dpi)
                    plt.close(fig)
                    generated.append(str(path))
                else:
                    skipped.append({"plot": plot, "reason": "Time evolution plotting expects a 2D grid with time."})
            else:
                skipped.append({"plot": plot, "reason": "Plot not available for current prediction bundle."})
        except Exception as exc:  # noqa: BLE001
            skipped.append({"plot": plot, "reason": str(exc)})
    return {"generated": generated, "skipped": skipped}

