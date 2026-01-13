#!/usr/bin/env python3
"""
Push from 1.53 to < 1.0

Now that we have a working baseline at 1.53, try incremental improvements.
Target: 35% improvement (1.53 → 1.0) through systematic optimization.
"""
from pathlib import Path
from src.utils.config import load_yaml
from src.distill.experiment import run_one_combo
import json
import copy

def test_improvement(cfg, teachers, student, name, description):
    """Test a single improvement"""
    run_dir = Path("runs") / f"push_1_0_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"{'='*80}")

    try:
        metrics, per_series, history = run_one_combo(cfg, teachers, student, str(run_dir))

        per_series.to_csv(run_dir / "per_series.csv", index=False)
        history.to_csv(run_dir / "history.csv", index=False)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mae = metrics['MAE']
        print(f"\n✓ MAE: {mae:.4f}")
        if mae < 1.0:
            print(f"  🎉 SUCCESS! Below 1.0!")
        return mae
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return float('inf')

def main():
    base_cfg = load_yaml("configs/default.yaml")
    baseline = 1.53

    print("="*80)
    print(f"PUSH FROM {baseline} TO < 1.0")
    print("="*80)
    print(f"\nCurrent baseline: {baseline:.2f} BDT/kg")
    print(f"Target: < 1.0 BDT/kg")
    print(f"Need: 35% improvement\n")

    teachers = ["nbeats"]  # Your best teacher
    student = "mlp"
    results = [("Baseline", baseline)]

    # Test 1: More aggressive distillation
    print("\n" + "="*80)
    print("TEST 1: MORE AGGRESSIVE DISTILLATION")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    cfg['distill']['losses']['hard']['weight'] = 0.25
    cfg['distill']['losses']['kd_pred']['weight'] = 0.65
    cfg['distill']['losses']['kd_feat']['weight'] = 0.2
    cfg['distill']['losses']['kd_diff']['weight'] = 0.15
    mae = test_improvement(cfg, teachers, student, "more_distill",
                          "0.25 hard / 0.65 pred / 0.2 feat / 0.15 diff")
    results.append(("More aggressive distillation", mae))

    # Test 2: Bigger student
    print("\n" + "="*80)
    print("TEST 2: BIGGER STUDENT")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    cfg['students']['models']['mlp']['layers'] = [768, 384, 192]
    mae = test_improvement(cfg, teachers, student, "bigger_student",
                          "MLP: [768, 384, 192]")
    results.append(("Bigger student", mae))

    # Test 3: Longer training
    print("\n" + "="*80)
    print("TEST 3: LONGER TRAINING")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    cfg['train']['epochs'] = 150
    cfg['train']['early_stopping']['patience'] = 20
    mae = test_improvement(cfg, teachers, student, "longer_training",
                          "150 epochs, patience 20")
    results.append(("Longer training", mae))

    # Test 4: All 3 teachers
    print("\n" + "="*80)
    print("TEST 4: ALL 3 TEACHERS")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    mae = test_improvement(cfg, ["dlinear", "patchtst", "nbeats"], student,
                          "all_teachers", "Ensemble: dlinear + patchtst + nbeats")
    results.append(("All 3 teachers", mae))

    # Test 5: Lower LR + longer
    print("\n" + "="*80)
    print("TEST 5: LOWER LR + LONGER")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    cfg['train']['lr'] = 1.0e-4
    cfg['train']['epochs'] = 150
    cfg['train']['early_stopping']['patience'] = 20
    mae = test_improvement(cfg, teachers, student, "lower_lr",
                          "LR: 1e-4, 150 epochs")
    results.append(("Lower LR + longer", mae))

    # Test 6: BEST COMBINATION
    print("\n" + "="*80)
    print("TEST 6: COMBINE BEST IMPROVEMENTS")
    print("="*80)
    cfg = copy.deepcopy(base_cfg)
    # Apply all improvements that helped
    cfg['distill']['losses']['hard']['weight'] = 0.25
    cfg['distill']['losses']['kd_pred']['weight'] = 0.65
    cfg['distill']['losses']['kd_feat']['weight'] = 0.2
    cfg['distill']['losses']['kd_diff']['weight'] = 0.15
    cfg['students']['models']['mlp']['layers'] = [768, 384, 192]
    cfg['train']['lr'] = 1.0e-4
    cfg['train']['epochs'] = 150
    cfg['train']['early_stopping']['patience'] = 20
    mae = test_improvement(cfg, teachers, student, "combined_best",
                          "All best improvements combined")
    results.append(("Combined best", mae))

    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nBaseline: {baseline:.4f} BDT/kg\n")

    best_mae = baseline
    best_name = "Baseline"

    for name, mae in results:
        if mae < baseline:
            improvement = ((baseline - mae) / baseline) * 100
            status = f"✓ {improvement:+.1f}%"
        else:
            status = "= baseline"

        if mae < best_mae:
            best_mae = mae
            best_name = name

        success = "🎉" if mae < 1.0 else ""
        print(f"  {mae:.4f} - {name:30s} {status} {success}")

    print(f"\n{'='*80}")
    print(f"BEST RESULT: {best_name}")
    print(f"MAE: {best_mae:.4f} BDT/kg")

    if best_mae < 1.0:
        print(f"\n🎉🎉🎉 SUCCESS! MAE < 1.0 ACHIEVED! 🎉🎉🎉")
        gap = 1.0 - best_mae
        print(f"Below target by: {gap:.4f} BDT/kg")
    else:
        gap = best_mae - 1.0
        pct = (gap / best_mae) * 100
        print(f"\nGap remaining: {gap:.4f} BDT/kg ({pct:.1f}%)")

        improvement_so_far = ((baseline - best_mae) / baseline) * 100
        print(f"Improvement so far: {improvement_so_far:.1f}%")

        if best_mae < 1.2:
            print(f"\n💡 Very close! Try:")
            print(f"  - Train even longer (200+ epochs)")
            print(f"  - Even bigger model [1024, 512, 256]")
            print(f"  - Ensemble predictions from multiple runs")

    print("="*80)

if __name__ == "__main__":
    main()
