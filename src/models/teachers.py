from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from .common import ForecastModel

class DLinearTeacher(ForecastModel):
    '''
    Minimal DLinear-style model: moving-average trend + seasonal residual, each with linear head.
    '''
    def __init__(self, input_len: int, horizon: int, hidden: int = 64, decomposition_kernel: int = 25, dropout: float = 0.1):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.kernel = decomposition_kernel
        self.avg_pool = nn.AvgPool1d(kernel_size=decomposition_kernel, stride=1, padding=decomposition_kernel//2, count_include_pad=False)
        self.lin_trend = nn.Linear(input_len, horizon)
        self.lin_season = nn.Linear(input_len, horizon)
        self.dropout = nn.Dropout(dropout)

    def _decompose(self, x: torch.Tensor):
        # x: (B, L)
        x1 = x.unsqueeze(1)  # (B,1,L)
        trend = self.avg_pool(x1).squeeze(1)
        season = x - trend
        return trend, season

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend, season = self._decompose(x)
        y = self.lin_trend(self.dropout(trend)) + self.lin_season(self.dropout(season))
        return y

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        trend, season = self._decompose(x)
        feat = torch.cat([trend, season], dim=-1)  # (B, 2L)
        return feat

class PatchTSTTeacher(ForecastModel):
    '''
    Minimal PatchTST-style model: split into patches, embed, TransformerEncoder, predict horizon.
    '''
    def __init__(self, input_len: int, horizon: int, patch_len: int = 8, d_model: int = 64, nhead: int = 4, num_layers: int = 2, ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.patch_len = patch_len
        self.n_patches = (input_len + patch_len - 1) // patch_len
        self.pad_len = self.n_patches * patch_len - input_len

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, horizon),
        )

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_len > 0:
            x = torch.cat([x, x[:, -1:].repeat(1, self.pad_len)], dim=1)
        B, L = x.shape
        patches = x.view(B, self.n_patches, self.patch_len)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self._to_patches(x)
        z = self.patch_proj(p) + self.pos
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        return self.head(pooled)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        p = self._to_patches(x)
        z = self.patch_proj(p) + self.pos
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        return pooled

class NBEATSTeacher(ForecastModel):
    '''
    Minimal N-BEATS-style fully-connected residual stacks.
    '''
    def __init__(self, input_len: int, horizon: int, stacks: int = 2, blocks_per_stack: int = 2, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.stacks = nn.ModuleList([
            nn.ModuleList([_NBEATSBlock(input_len, horizon, hidden, dropout) for _ in range(blocks_per_stack)])
            for _ in range(stacks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backcast = x
        forecast = torch.zeros(x.size(0), self.horizon, device=x.device)
        for stack in self.stacks:
            for block in stack:
                b, f = block(backcast)
                backcast = backcast - b
                forecast = forecast + f
        return forecast

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # Use last block hidden activation as a feature (approx)
        backcast = x
        feat = None
        for stack in self.stacks:
            for block in stack:
                b, f, h = block(backcast, return_hidden=True)
                backcast = backcast - b
                feat = h
        return feat

class _NBEATSBlock(nn.Module):
    def __init__(self, input_len: int, horizon: int, hidden: int, dropout: float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_len, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.backcast = nn.Linear(hidden, input_len)
        self.forecast = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        h = self.fc(x)
        b = self.backcast(h)
        f = self.forecast(h)
        if return_hidden:
            return b, f, h
        return b, f

def build_teachers(cfg: Dict) -> Dict[str, ForecastModel]:
    input_len = int(cfg["task"]["input_len"])
    horizon = int(cfg["task"]["horizon"])
    common = cfg["teachers"]["common"]
    models = cfg["teachers"]["models"]

    out: Dict[str, ForecastModel] = {}
    if models.get("dlinear", {}).get("enabled", False):
        m = models["dlinear"]
        out["dlinear"] = DLinearTeacher(
            input_len=input_len, horizon=horizon,
            hidden=int(common.get("hidden", 64)),
            decomposition_kernel=int(m.get("decomposition_kernel", 25)),
            dropout=float(common.get("dropout", 0.1)),
        )
    if models.get("patchtst", {}).get("enabled", False):
        m = models["patchtst"]
        out["patchtst"] = PatchTSTTeacher(
            input_len=input_len, horizon=horizon,
            patch_len=int(m.get("patch_len", 8)),
            d_model=int(m.get("d_model", 64)),
            nhead=int(m.get("nhead", 4)),
            num_layers=int(m.get("num_layers", 2)),
            ff=int(m.get("ff", 128)),
            dropout=float(common.get("dropout", 0.1)),
        )
    if models.get("nbeats", {}).get("enabled", False):
        m = models["nbeats"]
        out["nbeats"] = NBEATSTeacher(
            input_len=input_len, horizon=horizon,
            stacks=int(m.get("stacks", 2)),
            blocks_per_stack=int(m.get("blocks_per_stack", 2)),
            hidden=int(m.get("hidden", 128)),
            dropout=float(common.get("dropout", 0.1)),
        )
    return out
