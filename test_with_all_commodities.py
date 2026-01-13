#!/usr/bin/env python3
"""
Test with ALL commodities properly configured

Now that we fixed input_len to use all available commodities,
let's see the TRUE baseline performance.
"""
from pathlib import Path
from src.utils.config import load_yaml
from src.distill.experiment import run_one_combo
import json

def main():
    cfg = load_yaml("configs/default.yaml")

    print("="*80)
    print("TEST WITH ALL COMMODITIES (CORRECTED)")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  input_len: {cfg['task']['input_len']} (reduced from 24)")
    print(f"  n_commodities: {cfg['data']['n_commodities']}")
    print(f"  Expected usable: 4-5 commodities (vs 2 before)")

    print("\n" + "="*80)
    print("Training nbeats → mlp with ALL commodities")
    print("="*80 + "\n")

    teacher_names = ["nbeats"]
    student = "mlp"

    run_dir = Path("runs") / f"all_commodities_nbeats_mlp"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        metrics, per_series, history = run_one_combo(
            cfg, teacher_names, student, str(run_dir)
        )

        per_series.to_csv(run_dir / "per_series.csv", index=False)
        history.to_csv(run_dir / "history.csv", index=False)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "="*80)
        print("RESULTS WITH ALL COMMODITIES")
        print("="*80)

        mae = metrics['MAE']
        n_comm = metrics['n_commodities']

        print(f"\n📊 Performance:")
        print(f"   MAE: {mae:.4f} BDT/kg")
        print(f"   Commodities used: {n_comm}")
        print(f"   Previous (2 commodities): 1.53")

        if n_comm > 2:
            print(f"\n   ✓ Using {n_comm - 2} MORE commodities than before!")
        else:
            print(f"\n   ⚠️  Still only using {n_comm} commodities")

        print(f"\n📈 All Metrics:")
        print(f"   MAE:   {metrics['MAE']:.4f}")
        print(f"   RMSE:  {metrics['RMSE']:.4f}")
        print(f"   MAPE:  {metrics['MAPE']:.2f}%")
        print(f"   sMAPE: {metrics['sMAPE']:.2f}%")
        print(f"   MASE:  {metrics['MASE']:.4f}")

        print(f"\n🌾 Per-commodity:")
        for _, row in per_series.iterrows():
            status = "✓" if row['student_test_mae'] < 1.0 else "✗"
            print(f"   {status} {row['commodity']:40s}: {row['student_test_mae']:.4f}")

        print(f"\n🎯 Comparison:")
        if mae < 1.0:
            print(f"   🎉 SUCCESS! MAE < 1.0 with all commodities!")
        elif mae < 1.53:
            improvement = ((1.53 - mae) / 1.53) * 100
            print(f"   ✓ Better than before! {improvement:.1f}% improvement")
            gap = mae - 1.0
            print(f"   Gap to 1.0: {gap:.4f}")
        else:
            regression = ((mae - 1.53) / 1.53) * 100
            print(f"   ⚠️  Worse than 2-commodity result (+{regression:.1f}%)")
            print(f"   This might be because new commodities are harder to predict")

        print("\n" + "="*80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
