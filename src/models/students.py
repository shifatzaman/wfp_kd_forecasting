from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from .common import ForecastModel

class MLPStudent(ForecastModel):
    def __init__(self, input_len: int, horizon: int, layers: list[int], dropout: float = 0.1):
        super().__init__()
        dims = [input_len] + layers
        blocks = []
        for i in range(len(dims)-1):
            blocks += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*blocks) if blocks else nn.Identity()
        last_dim = dims[-1] if layers else input_len
        self.head = nn.Linear(last_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class GRUStudent(ForecastModel):
    def __init__(self, input_len: int, horizon: int, hidden: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L) -> (B,L,1)
        z, h = self.rnn(x.unsqueeze(-1))
        last = z[:, -1, :]
        return self.head(last)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        z, h = self.rnn(x.unsqueeze(-1))
        return z[:, -1, :]

class SimpleKANLayer(nn.Module):
    '''
    A small, readable, KAN-inspired layer:
    - For each input dimension, learn a 1D piecewise-linear function on a fixed grid.
    - Sum transformed dims and apply linear mixing.
    This is NOT a full KAN implementation; it's a practical approximation for experimentation.
    '''
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        # Knot values in [-1, 1] (inputs should be standardized)
        self.register_buffer("knots", torch.linspace(-1.0, 1.0, grid_size))
        # Values of the spline at knots per input dim (learnable)
        self.values = nn.Parameter(torch.zeros(in_dim, grid_size))
        # Linear mixing after nonlinearity
        self.mix = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        # piecewise-linear interpolation per dimension
        x_clamped = x.clamp(-1.0, 1.0)
        # indices of right knot
        idx = torch.bucketize(x_clamped, self.knots)  # (B,in_dim), in [0..grid]
        idx = idx.clamp(1, self.grid_size-1)
        x0 = self.knots[idx-1]
        x1 = self.knots[idx]
        v0 = self.values[:, :].T[idx-1]  # (B,in_dim)
        v1 = self.values[:, :].T[idx]    # (B,in_dim)
        t = (x_clamped - x0) / (x1 - x0 + 1e-8)
        y = v0 + t * (v1 - v0)
        return self.mix(y)

class KANStudent(ForecastModel):
    def __init__(self, input_len: int, horizon: int, hidden: int = 64, grid_size: int = 16, dropout: float = 0.1):
        super().__init__()
        self.kan1 = SimpleKANLayer(input_len, hidden, grid_size=grid_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.kan2 = SimpleKANLayer(hidden, horizon, grid_size=grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.kan1(x)))
        return self.kan2(h)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.kan1(x)))

def build_student(cfg: Dict, name: str) -> ForecastModel:
    input_len = int(cfg["task"]["input_len"])
    horizon = int(cfg["task"]["horizon"])
    common = cfg["students"]["common"]
    m = cfg["students"]["models"][name]

    if name == "mlp":
        return MLPStudent(input_len, horizon, layers=list(m.get("layers", [128, 64])), dropout=float(common.get("dropout", 0.1)))
    if name == "gru":
        return GRUStudent(input_len, horizon, hidden=int(common.get("hidden", 64)), num_layers=int(m.get("num_layers", 2)), dropout=float(common.get("dropout", 0.1)))
    if name == "kan":
        return KANStudent(input_len, horizon, hidden=int(common.get("hidden", 64)), grid_size=int(m.get("grid_size", 16)), dropout=float(common.get("dropout", 0.1)))
    raise ValueError(f"Unknown student: {name}")
