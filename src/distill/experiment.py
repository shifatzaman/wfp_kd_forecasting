from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
import numpy as np
import pandas as pd
import torch

from ..models.teachers import build_teachers
from ..models.students import build_student
from ..models.common import choose_device, ForecastModel
from ..data.wfp import load_and_prepare
from ..utils.config import save_yaml
from ..utils.logging import RunLogger, append_summary
from ..utils.seed import set_seed
from .trainer import build_loaders_for_series, train_single_model, evaluate_model

def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = -x / max(temp, 1e-8)
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-8)

def _make_teacher_ensemble(teachers: Dict[str, ForecastModel], weights: Dict[str, float]):
    names = list(teachers.keys())
    w = torch.tensor([weights[n] for n in names])
    w = w / (w.sum() + 1e-8)

    def ensemble(x: torch.Tensor) -> torch.Tensor:
        preds = []
        for n in names:
            preds.append(teachers[n](x))
        stacked = torch.stack(preds, dim=0)  # (T,B,H)
        return (w.view(-1,1,1).to(stacked.device) * stacked).sum(dim=0)
    return ensemble

def _make_teacher_feat_fn(teachers: Dict[str, ForecastModel], weights: Dict[str, float]):
    # Weighted average of teacher features (truncate to min dim across teachers)
    names = list(teachers.keys())
    w = torch.tensor([weights[n] for n in names])
    w = w / (w.sum() + 1e-8)

    def feat(x: torch.Tensor) -> torch.Tensor:
        feats = []
        dims = []
        for n in names:
            f = teachers[n].get_features(x)
            feats.append(f)
            dims.append(f.shape[-1])
        d = min(dims)
        feats = [f[..., :d] for f in feats]
        stacked = torch.stack(feats, dim=0)  # (T,B,d)
        return (w.view(-1,1,1).to(stacked.device) * stacked).sum(dim=0)
    return feat

@dataclass
class ComboResult:
    run_id: str
    student: str
    teachers: List[str]
    val_mae: float
    test_mae: float
    test_mse: float

def run_one_combo(cfg: Dict, teacher_names: List[str], student_name: str, run_dir: str) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    set_seed(int(cfg["seed"]))
    device = choose_device(cfg["train"]["device"])
    prep = load_and_prepare(**cfg["data"])
    used_commodities = []

    # Select enabled teachers subset
    all_teachers = build_teachers(cfg)
    teachers = {k: v for k, v in all_teachers.items() if k in teacher_names}
    if len(teachers) == 0:
        raise ValueError("No teachers selected.")

    # Fit teachers per series (independently) then compute teacher weights based on val MAE
    per_series_rows = []
    histories = []

    dist = cfg["distill"]
    weighting = dist["ensemble"]["weighting"]
    temp = float(dist["ensemble"]["temperature"])

    for key, s in prep.series.items():
        loaders, scaler, splits = build_loaders_for_series(s, cfg)

        if len(loaders["train"]) == 0 or len(loaders["val"]) == 0 or len(loaders["test"]) == 0:
            continue

        # Train each teacher on this series
        teacher_val_mae = {}
        for tname, tmodel in teachers.items():
            tmodel = tmodel.to(device)
            fit = train_single_model(tmodel, loaders, cfg, device, teacher_ensemble=None)
            teacher_val_mae[tname] = float(fit.best_val_mae)

        if weighting == "uniform":
            weights = {k: 1.0 for k in teachers.keys()}
        else:
            maes = np.array([teacher_val_mae[k] for k in teachers.keys()], dtype=np.float64)
            sw = _softmax(maes, temp=temp)  # smaller MAE => larger weight
            weights = {k: float(sw[i]) for i, k in enumerate(teachers.keys())}

        # Prepare ensemble fns
        ensemble = _make_teacher_ensemble(teachers, weights)
        feat_fn = _make_teacher_feat_fn(teachers, weights) if dist["losses"]["kd_feat"]["enabled"] or dist["losses"]["kd_contrastive"]["enabled"] else None

        # Train student
        student = build_student(cfg, student_name).to(device)
        fit_s = train_single_model(student, loaders, cfg, device, teacher_ensemble=ensemble, teacher_feat_fn=feat_fn)

        # Evaluate
        target = cfg["task"].get("target", "price")

        val_metrics = evaluate_model(
            student, loaders["val"], device, scaler=scaler, target=target
        )
        test_metrics = evaluate_model(
            student, loaders["test"], device, scaler=scaler, target=target
)
        commodity = prep.meta.loc[
            prep.meta["key"] == key, "commodity"
        ].values[0]

        used_commodities.append(commodity)
        per_series_rows.append({
            "key": key,
            "commodity": prep.meta.loc[prep.meta["key"] == key, "commodity"].values[0],
            "teacher_set": " + ".join(teacher_names),
            "student": student_name,

            # validation (still MAE only)
            "student_val_mae": val_metrics["mae"],

            # test metrics (real price)
            "student_test_mae": test_metrics["mae"],
            "student_test_rmse": test_metrics["rmse"],
            "student_test_mape": test_metrics["mape"],
            "student_test_nmae": test_metrics["nmae"],

            **{f"teacher_{k}_val_mae": v for k, v in teacher_val_mae.items()},
            **{f"teacher_weight_{k}": v for k, v in weights.items()},
        })
        h = fit_s.history.copy()
        h["series_key"] = key
        histories.append(h)

    per_series = pd.DataFrame(per_series_rows)
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    

    def safe_mean(x):
        x = x.dropna()
        return float(x.mean()) if len(x) > 0 else float("nan")

    metrics = {
        # commodity info
        "commodities": "; ".join(used_commodities),
        "n_commodities": len(used_commodities),
        "market": cfg["data"]["market"],
        "teachers": " + ".join(teacher_names),
        "student": student_name,
        "target": cfg["task"]["target"],

        # aggregated test metrics (real price)
        "MAE": safe_mean(per_series["student_test_mae"]),
        "RMSE": safe_mean(per_series["student_test_rmse"]),
        "MAPE": safe_mean(per_series["student_test_mape"]),
        "NMAE": safe_mean(per_series["student_test_nmae"]),
    }
    return metrics, per_series, history

def run_grid(cfg: Dict, out_root: str = "runs") -> None:
    out_root = str(out_root)
    from pathlib import Path
    import datetime

    teacher_sets = cfg["experiments"]["grid"]["teacher_sets"]
    students = cfg["experiments"]["grid"]["students"]

    Path(out_root).mkdir(parents=True, exist_ok=True)
    summary_csv = Path(out_root) / "summary.csv"

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_id = f"grid_{ts}"

    for tset in teacher_sets:
        for sname in students:
            run_id = f"{grid_id}__T={'-'.join(tset)}__S={sname}"
            run_dir = Path(out_root) / run_id
            logger = RunLogger(run_dir)

            # resolve config for this run
            cfg_run = dict(cfg)
            cfg_run["selected"] = {"teachers": tset, "student": sname}
            save_yaml(cfg_run, run_dir / "config_resolved.yaml")

            try:
                metrics, per_series, history = run_one_combo(cfg_run, tset, sname, str(run_dir))
                logger.save_json("metrics.json", metrics)
                logger.save_df("per_series.csv", per_series)
                logger.save_df("history.csv", history)

                append_summary(summary_csv, metrics)
            except Exception as e:
                logger.save_json("error.json", {"error": str(e)})
                append_summary(summary_csv, {
                    "run_id": run_id,
                    "teachers": "+".join(tset),
                    "student": sname,
                    "error": str(e),
                })
