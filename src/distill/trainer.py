from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..models.common import ForecastModel, choose_device, MLPProjector
from .losses import mae_loss, mse_loss, contrastive_loss

class WindowDataset(Dataset):
    def __init__(
        self,
        values,
        input_len,
        horizon,
        stride,
        target="price",
        add_seasonality=False,
        months=None,
    ):
        self.values = values.astype(np.float32)
        self.input_len = input_len
        self.horizon = horizon
        self.stride = stride
        self.n = len(values)
        self.target = target
        self.idxs = list(range(0, self.n - input_len - horizon + 1, stride))
        self.add_seasonality = add_seasonality
        self.months = months  # array of month numbers aligned with values

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]

        x = self.values[t : t + self.input_len]           # (L, 1)
        y = self.values[t + self.input_len : t + self.input_len + self.horizon]

        if self.target == "residual":
            y = y - x[-1]

        # --- 🌙 Add seasonality features ---
        if self.add_seasonality:
            m = self.months[t : t + self.input_len]       # month indices [1..12]
            sin_m = np.sin(2 * np.pi * m / 12.0)
            cos_m = np.cos(2 * np.pi * m / 12.0)

            # stack: (L, 3) → price + sin + cos
            x = np.column_stack([x, sin_m, cos_m])

        return torch.from_numpy(x), torch.from_numpy(y)

def make_splits(series: pd.Series, train: float, val: float, test: float):
    n = len(series)
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    train_s = series.iloc[:n_train]
    val_s = series.iloc[n_train:n_train+n_val]
    test_s = series.iloc[n_train+n_val:]
    return train_s, val_s, test_s

def make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    return None

@dataclass
class FitResult:
    best_val_mae: float
    history: pd.DataFrame

def _hard_loss(kind: str, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if kind == "mse":
        return mse_loss(yhat, y)
    return mae_loss(yhat, y)

def train_single_model(
    model: ForecastModel,
    loaders: Dict[str, DataLoader],
    cfg: Dict,
    device: torch.device,
    teacher_ensemble: Optional[callable] = None,
    teacher_feat_fn: Optional[callable] = None,
) -> FitResult:

    if len(loaders["train"]) == 0 or len(loaders["val"]) == 0:
        return FitResult(
            best_val_mae=float("inf"),
            history=pd.DataFrame([{
                "epoch": 0,
                "train_loss": float("nan"),
                "val_mae": float("inf"),
            }])
        )
    train_cfg = cfg["train"]
    dist = cfg["distill"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    best = float("inf")
    patience = int(train_cfg["early_stopping"]["patience"]) if train_cfg["early_stopping"]["enabled"] else 10**9
    bad = 0
    history_rows = []

    # Feature projector for student (optional)
    proj = None
    if dist["enabled"] and dist["losses"]["kd_feat"]["enabled"]:
        proj_dim = int(dist["losses"]["kd_feat"]["proj_dim"])
        # infer student feature dim with one batch
        xb, yb = next(iter(loaders["train"]))
        xb = xb.to(device)
        with torch.no_grad():
            sf = model.get_features(xb)
        in_dim = int(sf.shape[-1])
        if dist["losses"]["kd_feat"]["projector"] == "linear":
            proj = nn.Linear(in_dim, proj_dim).to(device)
        else:
            proj = MLPProjector(in_dim, proj_dim).to(device)
        opt = torch.optim.AdamW(list(model.parameters()) + list(proj.parameters()), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        if proj is not None:
            proj.train()
        tr_losses = []
        pbar = tqdm(loaders["train"], desc=f"train epoch {epoch+1}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            yhat = model(x)

            # Hard loss
            hard = _hard_loss(dist["losses"]["hard"]["type"], yhat, y) * float(dist["losses"]["hard"]["weight"])

            loss = hard

            if dist["enabled"] and teacher_ensemble is not None:
                with torch.no_grad():
                    yhat_t_price = teacher_ensemble(x)

                if cfg["task"].get("target", "price") == "residual":
                    x_last = x[:, -1:].detach()
                    yhat_t = yhat_t_price - x_last
                else:
                    yhat_t = yhat_t_price
                if dist["losses"]["kd_pred"]["enabled"]:
                    kd = mse_loss(yhat, yhat_t) * float(dist["losses"]["kd_pred"]["weight"])
                    loss = loss + kd

                if dist["losses"]["kd_feat"]["enabled"] and teacher_feat_fn is not None:
                    with torch.no_grad():
                        tf = teacher_feat_fn(x)
                    sf = model.get_features(x)
                    sfp = proj(sf) if proj is not None else sf
                    # if dims mismatch, truncate/pad (simple)
                    if sfp.shape[-1] != tf.shape[-1]:
                        d = min(sfp.shape[-1], tf.shape[-1])
                        sfp = sfp[..., :d]
                        tf = tf[..., :d]
                    fd = mse_loss(sfp, tf) * float(dist["losses"]["kd_feat"]["weight"])
                    loss = loss + fd

                if dist["losses"]["kd_contrastive"]["enabled"] and teacher_feat_fn is not None:
                    with torch.no_grad():
                        tf = teacher_feat_fn(x)
                    sf = model.get_features(x)
                    # match dims (truncate)
                    d = min(sf.shape[-1], tf.shape[-1])
                    cl = contrastive_loss(sf[..., :d], tf[..., :d], temperature=float(dist["losses"]["kd_contrastive"]["temperature"])) * float(dist["losses"]["kd_contrastive"]["weight"])
                    loss = loss + cl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(train_cfg["clip_grad"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["clip_grad"]))
            opt.step()

            tr_losses.append(loss.item())
            pbar.set_postfix({"loss": float(np.mean(tr_losses))})

        # Validation
        model.eval()
        if proj is not None:
            proj.eval()
        val_maes = []
        with torch.no_grad():
            for x, y in loaders["val"]:
                x = x.to(device)
                y = y.to(device)
                yhat = model(x)
                val_maes.append(mae_loss(yhat, y).item())
        val_mae = float(np.mean(val_maes))
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")

        history_rows.append({"epoch": epoch+1, "train_loss": tr_loss, "val_mae": val_mae})

        if val_mae < best - 1e-6:
            best = val_mae
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_proj = {k: v.detach().cpu().clone() for k, v in proj.state_dict().items()} if proj is not None else None
        else:
            bad += 1
            if bad >= patience:
                break

    # restore best
    model.load_state_dict(best_state)
    if proj is not None and best_proj is not None:
        proj.load_state_dict(best_proj)

    hist = pd.DataFrame(history_rows)
    return FitResult(best_val_mae=best, history=hist)

def evaluate_model(model, loader, device, scaler=None, target="price"):
    model.eval()
    maes, mses = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)

            if target == "residual":
                x_last = x[:, -1:].detach()   # (B,1)
                y_price = y + x_last
                yhat_price = yhat + x_last
            else:
                y_price = y
                yhat_price = yhat

            y_np = y.cpu().numpy()
            yhat_np = yhat.cpu().numpy()

            # 🔑 Inverse scaling → actual BDT/kg
            if scaler is not None:
                y_np = scaler.inverse_transform(
                    y_np.reshape(-1, 1)
                ).reshape(y_np.shape)

                yhat_np = scaler.inverse_transform(
                    yhat_np.reshape(-1, 1)
                ).reshape(yhat_np.shape)

            maes.append(abs(yhat_np - y_np).mean())
            mses.append(((yhat_np - y_np) ** 2).mean())

    if len(maes) == 0:
        return {"mae": float("nan"), "mse": float("nan")}

    return {
        "mae": float(np.mean(maes)),
        "mse": float(np.mean(mses)),
    }

def build_loaders_for_series(series: pd.Series, cfg: Dict):
    task = cfg["task"]
    split = cfg["split"]
    input_len = int(task["input_len"])
    horizon = int(task["horizon"])
    stride = int(task.get("stride", 1))
    months = series.index.month.values.astype(np.float32)

    train_s, val_s, test_s = make_splits(series, split["train"], split["val"], split["test"])

    scaler = make_scaler(task.get("scale", "standard"))
    if scaler is not None:
        scaler.fit(train_s.values.reshape(-1, 1))
        train_vals = scaler.transform(train_s.values.reshape(-1,1)).reshape(-1)
        val_vals = scaler.transform(val_s.values.reshape(-1,1)).reshape(-1)
        test_vals = scaler.transform(test_s.values.reshape(-1,1)).reshape(-1)
    else:
        train_vals = train_s.values
        val_vals = val_s.values
        test_vals = test_s.values

    target = cfg["task"].get("target", "price")

    add_seasonality = cfg["task"].get("add_seasonality", False)

    ds_train = WindowDataset(
        train_vals,
        input_len,
        horizon,
        stride,
        target=target,
        add_seasonality=add_seasonality,
        months=months[: len(train_vals)],
    )

    ds_val = WindowDataset(
        val_vals,
        input_len,
        horizon,
        stride,
        target=target,
        add_seasonality=add_seasonality,
        months=months[len(train_vals) : len(train_vals) + len(val_vals)],
    )

    ds_test = WindowDataset(
        test_vals,
        input_len,
        horizon,
        stride,
        target=target,
        add_seasonality=add_seasonality,
        months=months[-len(test_vals) :],
    )

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    loaders = {
        "train": DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True),
        "val": DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw),
        "test": DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=nw),
    }
    return loaders, scaler, (train_s, val_s, test_s)
