from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from .common import ForecastModel


def _num_channels_from_cfg(cfg: Dict) -> int:
    # If seasonality is on, dataset returns (price, sin, cos) = 3 channels
    return 3 if bool(cfg["task"].get("add_seasonality", False)) else 1


def _flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
    """
    x:
      - (B, L)      -> (B, L)
      - (B, L, C)   -> (B, L*C)
    """
    if x.dim() == 3:
        B, L, C = x.shape
        return x.reshape(B, L * C)
    return x


class MLPStudent(ForecastModel):
    def __init__(self, in_features: int, horizon: int, layers: list[int], dropout: float = 0.1):
        super().__init__()
        dims = [in_features] + layers
        blocks = []
        for i in range(len(dims) - 1):
            blocks += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*blocks) if blocks else nn.Identity()
        last_dim = dims[-1] if layers else in_features
        self.head = nn.Linear(last_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_if_needed(x)
        h = self.backbone(x)
        return self.head(h)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_if_needed(x)
        return self.backbone(x)


class GRUStudent(ForecastModel):
    def __init__(self, input_size: int, horizon: int, hidden: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:
        #  - (B,L) -> (B,L,1)
        #  - (B,L,C) stays as is
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        z, _ = self.rnn(x)      # (B,L,H)
        last = z[:, -1, :]      # (B,H)
        return self.head(last)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        z, _ = self.rnn(x)
        return z[:, -1, :]


class SimpleKANLayer(nn.Module):
    """
    A small, readable, KAN-inspired layer:
    - For each input dimension, learn a 1D piecewise-linear function on a fixed grid.
    - Apply learned spline per dim then linear mixing.
    """

    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.register_buffer("knots", torch.linspace(-1.0, 1.0, grid_size))
        self.values = nn.Parameter(torch.zeros(in_dim, grid_size))
        self.mix = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        x = x.clamp(-1.0, 1.0)
        B, D = x.shape
        knots = self.knots  # (G,)
        G = knots.numel()

        idx = torch.searchsorted(knots, x).clamp(1, G - 1)  # (B,D)
        x0 = knots[idx - 1]
        x1 = knots[idx]

        vals = self.values.unsqueeze(0).expand(B, -1, -1)   # (B,D,G)
        v0 = torch.gather(vals, 2, (idx - 1).unsqueeze(-1)).squeeze(-1)  # (B,D)
        v1 = torch.gather(vals, 2, idx.unsqueeze(-1)).squeeze(-1)        # (B,D)

        t = (x - x0) / (x1 - x0 + 1e-8)
        y = v0 + t * (v1 - v0)  # (B,D)

        return self.mix(y)


class KANStudent(ForecastModel):
    def __init__(self, in_features: int, horizon: int, hidden: int = 64, grid_size: int = 16, dropout: float = 0.1):
        super().__init__()
        self.kan1 = SimpleKANLayer(in_features, hidden, grid_size=grid_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.kan2 = SimpleKANLayer(hidden, horizon, grid_size=grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_if_needed(x)
        h = self.drop(self.act(self.kan1(x)))
        return self.kan2(h)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_if_needed(x)
        return self.drop(self.act(self.kan1(x)))


def build_student(cfg: Dict, name: str) -> ForecastModel:
    input_len = int(cfg["task"]["input_len"])
    horizon = int(cfg["task"]["horizon"])
    common = cfg["students"]["common"]
    m = cfg["students"]["models"][name]

    C = _num_channels_from_cfg(cfg)
    in_features = input_len * C  # for MLP/KAN we flatten (L,C)->(L*C)

    if name == "mlp":
        return MLPStudent(
            in_features=in_features,
            horizon=horizon,
            layers=list(m.get("layers", [128, 64])),
            dropout=float(common.get("dropout", 0.1)),
        )

    if name == "gru":
        return GRUStudent(
            input_size=C,  # GRU consumes (B,L,C)
            horizon=horizon,
            hidden=int(common.get("hidden", 64)),
            num_layers=int(m.get("num_layers", 2)),
            dropout=float(common.get("dropout", 0.1)),
        )

    if name == "kan":
        return KANStudent(
            in_features=in_features,
            horizon=horizon,
            hidden=int(common.get("hidden", 64)),
            grid_size=int(m.get("grid_size", 16)),
            dropout=float(common.get("dropout", 0.1)),
        )

    raise ValueError(f"Unknown student: {name}")