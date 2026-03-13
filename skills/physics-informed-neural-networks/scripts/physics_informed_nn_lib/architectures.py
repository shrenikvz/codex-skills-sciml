"""Torch architectures for the PINNs skill."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency at import time
    import torch
    from torch import nn
except Exception:  # noqa: BLE001 pragma: no cover
    torch = None
    nn = None


class ArchitectureError(RuntimeError):
    """Network construction failure."""


if nn is not None:

    class SineActivation(nn.Module):
        def forward(self, x):  # type: ignore[override]
            return torch.sin(x)


    class AdaptiveActivation(nn.Module):
        def __init__(self, base: nn.Module):
            super().__init__()
            self.base = base
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, x):  # type: ignore[override]
            return self.base(self.scale * x)


    def _activation(name: str, adaptive: bool = False) -> nn.Module:
        mapping: dict[str, nn.Module] = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "sine": SineActivation(),
        }
        base = mapping.get(str(name).lower(), nn.Tanh())
        return AdaptiveActivation(base) if adaptive else base


    class FourierFeatureEmbedding(nn.Module):
        def __init__(self, input_dim: int, num_features: int, sigma: float):
            super().__init__()
            matrix = torch.randn(input_dim, num_features) * float(sigma)
            self.register_buffer("matrix", matrix)

        def forward(self, x):  # type: ignore[override]
            projected = 2.0 * math.pi * x @ self.matrix
            return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


    class MLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str, adaptive_activation: bool = False):
            super().__init__()
            layers: list[nn.Module] = []
            current = input_dim
            for _ in range(hidden_layers):
                layers.append(nn.Linear(current, hidden_units))
                layers.append(_activation(activation, adaptive=adaptive_activation))
                current = hidden_units
            layers.append(nn.Linear(current, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[override]
            return self.net(x)


    class ResidualMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str, adaptive_activation: bool = False):
            super().__init__()
            self.input = nn.Linear(input_dim, hidden_units)
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_units, hidden_units),
                        _activation(activation, adaptive=adaptive_activation),
                        nn.Linear(hidden_units, hidden_units),
                    )
                    for _ in range(max(2, hidden_layers))
                ]
            )
            self.activation = _activation(activation, adaptive=adaptive_activation)
            self.output = nn.Linear(hidden_units, output_dim)

        def forward(self, x):  # type: ignore[override]
            h = self.activation(self.input(x))
            for block in self.blocks:
                h = self.activation(h + block(h))
            return self.output(h)


    class FourierMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str, num_features: int, sigma: float, adaptive_activation: bool = False):
            super().__init__()
            self.embed = FourierFeatureEmbedding(input_dim, num_features, sigma)
            self.mlp = MLP(2 * num_features, output_dim, hidden_layers, hidden_units, activation, adaptive_activation=adaptive_activation)

        def forward(self, x):  # type: ignore[override]
            return self.mlp(self.embed(x))


    class MultiScaleMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str, scales: list[float], adaptive_activation: bool = False):
            super().__init__()
            self.embeddings = nn.ModuleList(
                [FourierFeatureEmbedding(input_dim, max(8, hidden_units // 4), scale) for scale in scales]
            )
            feature_dim = len(self.embeddings) * 2 * max(8, hidden_units // 4)
            self.mlp = MLP(feature_dim, output_dim, hidden_layers, hidden_units, activation, adaptive_activation=adaptive_activation)

        def forward(self, x):  # type: ignore[override]
            features = [embed(x) for embed in self.embeddings]
            return self.mlp(torch.cat(features, dim=-1))


    class TransformerOperatorNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, width: int, heads: int, layers: int):
            super().__init__()
            self.scalar_projection = nn.Linear(1, width)
            self.feature_embedding = nn.Embedding(input_dim, width)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=max(1, heads),
                dim_feedforward=2 * width,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, layers))
            self.output = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_dim))

        def forward(self, x):  # type: ignore[override]
            batch, features = x.shape
            values = self.scalar_projection(x.unsqueeze(-1))
            ids = torch.arange(features, device=x.device).unsqueeze(0).expand(batch, features)
            encoded = self.encoder(values + self.feature_embedding(ids))
            pooled = encoded.mean(dim=1)
            return self.output(pooled)


def build_torch_model(model_cfg: dict[str, Any], input_dim: int, output_dim: int) -> Any:
    if torch is None or nn is None:  # pragma: no cover
        raise ArchitectureError("PyTorch is not available. Install torch to run PINN training.")
    architecture = str(model_cfg.get("architecture", "mlp")).lower()
    hidden_layers = int(model_cfg.get("hidden_layers", 6))
    hidden_units = int(model_cfg.get("hidden_units", 128))
    activation = str(model_cfg.get("activation", "tanh"))
    adaptive = bool(model_cfg.get("adaptive_activation", False))
    if architecture in {"mlp", "coordinate"}:
        return MLP(input_dim, output_dim, hidden_layers, hidden_units, activation, adaptive_activation=adaptive)
    if architecture == "resnet":
        return ResidualMLP(input_dim, output_dim, hidden_layers, hidden_units, activation, adaptive_activation=adaptive)
    if architecture == "fourier":
        ff = model_cfg.get("fourier_features", {})
        return FourierMLP(
            input_dim,
            output_dim,
            hidden_layers,
            hidden_units,
            activation,
            int(ff.get("num_features", 64)),
            float(ff.get("sigma", 2.0)),
            adaptive_activation=adaptive,
        )
    if architecture == "multiscale":
        ms = model_cfg.get("multiscale", {})
        scales = [float(scale) for scale in ms.get("scales", [1.0, 2.0, 4.0])]
        return MultiScaleMLP(
            input_dim,
            output_dim,
            hidden_layers,
            hidden_units,
            activation,
            scales,
            adaptive_activation=adaptive,
        )
    if architecture == "transformer_operator":
        tfm = model_cfg.get("transformer", {})
        return TransformerOperatorNet(
            input_dim,
            output_dim,
            int(tfm.get("width", hidden_units)),
            int(tfm.get("heads", 4)),
            int(tfm.get("layers", max(2, hidden_layers // 2))),
        )
    raise ArchitectureError(f"Unsupported architecture: {architecture}")


def count_parameters(model: Any) -> int:
    if torch is None:  # pragma: no cover
        return 0
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def model_summary(model_cfg: dict[str, Any], input_dim: int, output_dim: int) -> dict[str, Any]:
    architecture = str(model_cfg.get("architecture", "mlp")).lower()
    summary = {
        "architecture": architecture,
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "hidden_layers": int(model_cfg.get("hidden_layers", 0) or 0),
        "hidden_units": int(model_cfg.get("hidden_units", 0) or 0),
        "activation": str(model_cfg.get("activation", "tanh")),
    }
    if architecture == "fourier":
        summary["fourier_features"] = dict(model_cfg.get("fourier_features", {}))
    if architecture == "multiscale":
        summary["multiscale"] = dict(model_cfg.get("multiscale", {}))
    if architecture == "transformer_operator":
        summary["transformer"] = dict(model_cfg.get("transformer", {}))
    return summary

