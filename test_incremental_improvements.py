#!/usr/bin/env python3
"""
Incremental approach to achieve MAE < 1.0

Strategy:
1. Start with nbeats → mlp (your proven best: 1.53)
2. Try SMALL incremental improvements
3. Test each change independently

Improvements to try:
A. Slightly bigger student: [256, 128] instead of [128, 64]
B. More distillation: 0.4 hard + 0.6 KD
C. Longer training: 50 epochs
D. One more commodity: n_commodities = 5
"""
from pathlib import Path
from src.utils.config import load_yaml
from src.distill.experiment import run_one_combo
import json
import copy

def test_config(cfg, teacher_names, student, run_name, description):
    """Test a single configuration"""
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"{'='*80}")

    try:
        metrics, per_series, history = run_one_combo(
            cfg, teacher_names, student, str(run_dir)
        )

        per_series.to_csv(run_dir / "per_series.csv", index=False)
        history.to_csv(run_dir / "history.csv", index=False)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mae = metrics['MAE']
        print(f"\n✓ MAE: {mae:.4f} BDT/kg")

        if mae < 1.0:
            print(f"  🎉 SUCCESS! Below 1.0!")

        return mae, metrics

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return float('inf'), {}

def main():
    base_cfg = load_yaml("configs/default.yaml")

    print("="*80)
    print("INCREMENTAL IMPROVEMENTS TO ACHIEVE MAE < 1.0")
    print("="*80)
    print("\nBaseline: nbeats → mlp [128, 64]")
    print("Target: 1.53 → < 1.0 (need ~35% improvement)")

    teacher_names = ["nbeats"]
    student = "mlp"

    results = []

    # Test 1: Baseline (reproduce 1.53)
    print("\n" + "="*80)
    print("TEST 1: BASELINE (Original Config)")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    mae, metrics = test_config(cfg, teacher_names, student,
                               "test1_baseline",
                               "Baseline: [128, 64], 0.5/0.5, 30 epochs, 4 commodities")
    results.append(("Baseline", mae))
    baseline_mae = mae

    # Test 2: Bigger student
    print("\n" + "="*80)
    print("TEST 2: BIGGER STUDENT")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    cfg['students']['models']['mlp']['layers'] = [256, 128]
    mae, metrics = test_config(cfg, teacher_names, student,
                               "test2_bigger_student",
                               "Bigger student: [256, 128]")
    results.append(("Bigger student [256, 128]", mae))

    # Test 3: More distillation
    print("\n" + "="*80)
    print("TEST 3: MORE DISTILLATION")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    cfg['distill']['losses']['hard']['weight'] = 0.4
    cfg['distill']['losses']['kd_pred']['weight'] = 0.6
    mae, metrics = test_config(cfg, teacher_names, student,
                               "test3_more_distill",
                               "More distillation: 0.4/0.6")
    results.append(("More distillation 0.4/0.6", mae))

    # Test 4: Longer training
    print("\n" + "="*80)
    print("TEST 4: LONGER TRAINING")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    cfg['train']['epochs'] = 50
    cfg['train']['early_stopping']['patience'] = 10
    mae, metrics = test_config(cfg, teacher_names, student,
                               "test4_longer_training",
                               "Longer training: 50 epochs")
    results.append(("Longer training 50 epochs", mae))

    # Test 5: Combine best improvements
    print("\n" + "="*80)
    print("TEST 5: COMBINE BEST")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    # Apply the improvement that worked best
    cfg['students']['models']['mlp']['layers'] = [256, 128]
    cfg['distill']['losses']['hard']['weight'] = 0.4
    cfg['distill']['losses']['kd_pred']['weight'] = 0.6
    cfg['train']['epochs'] = 50
    cfg['train']['early_stopping']['patience'] = 10
    mae, metrics = test_config(cfg, teacher_names, student,
                               "test5_combined",
                               "Combined: [256, 128] + 0.4/0.6 + 50 epochs")
    results.append(("Combined improvements", mae))

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nBaseline: {baseline_mae:.4f} BDT/kg")
    print("\nAll results:")
    for name, mae_val in results:
        if mae_val < 1.0:
            status = "🎉 SUCCESS"
        elif mae_val < baseline_mae:
            improvement = ((baseline_mae - mae_val) / baseline_mae) * 100
            status = f"✓ Better ({improvement:.1f}% improvement)"
        elif mae_val == baseline_mae:
            status = "= Same"
        else:
            regression = ((mae_val - baseline_mae) / baseline_mae) * 100
            status = f"✗ Worse ({regression:.1f}% regression)"

        print(f"  {mae_val:.4f} - {name:30s} {status}")

    best_result = min(results, key=lambda x: x[1])
    print(f"\nBest result: {best_result[0]} with MAE = {best_result[1]:.4f}")

    if best_result[1] < 1.0:
        print(f"\n🎉 TARGET ACHIEVED! MAE < 1.0")
    else:
        gap = best_result[1] - 1.0
        print(f"\nGap remaining: {gap:.4f} BDT/kg ({gap/best_result[1]*100:.1f}%)")
        print("\nNext steps:")
        print("  - Try [384, 192, 96] (even bigger)")
        print("  - Try 0.3 hard + 0.7 KD (more aggressive)")
        print("  - Try 100 epochs with patience 15")
        print("  - Try n_commodities = 5 (more data)")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
