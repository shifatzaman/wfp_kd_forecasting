from __future__ import annotations
import argparse
from pathlib import Path

from src.utils.config import load_yaml
from src.distill.experiment import run_grid, run_one_combo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--single", action="store_true", help="Run a single combo specified in config.selected (teachers + student).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    if args.single:
        sel = cfg.get("selected", {})
        teacher_names = sel.get("teachers", [])
        student = sel.get("student", "mlp")
        run_id = f"single__T={'-'.join(teacher_names)}__S={student}"
        run_dir = Path(args.out_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics, per_series, history = run_one_combo(cfg, teacher_names, student, str(run_dir))
        print("METRICS:", metrics)
        per_series.to_csv(run_dir / "per_series.csv", index=False)
        history.to_csv(run_dir / "history.csv", index=False)
    else:
        run_grid(cfg, out_root=args.out_root)
        print(f"Done. Summary written to {Path(args.out_root) / 'summary.csv'}")

if __name__ == "__main__":
    main()
