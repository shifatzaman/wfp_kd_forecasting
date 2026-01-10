from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

class ForecastModel(nn.Module):
    '''
    Base class for forecasting models.
    All models take input x: (B, L) and output yhat: (B, H).
    Optionally, they can return an intermediate feature representation via get_features().
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # Default: use last hidden representation if defined
        raise NotImplementedError("Model does not expose features for feature-KD.")

class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def choose_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
