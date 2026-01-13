#!/usr/bin/env python3
"""
Debug: Why are only 2 commodities showing up when 7 are configured?
"""
from src.utils.config import load_yaml
from src.data.wfp import load_and_prepare
from src.distill.trainer import build_loaders_for_series

def main():
    cfg = load_yaml("configs/default.yaml")

    print("="*80)
    print("COMMODITY DEBUG")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  n_commodities: {cfg['data']['n_commodities']}")
    print(f"  input_len: {cfg['task']['input_len']}")
    print(f"  horizon: {cfg['task']['horizon']}")
    print(f"  train/val/test split: {cfg['split']['train']}/{cfg['split']['val']}/{cfg['split']['test']}")

    # Load data
    prep = load_and_prepare(**cfg['data'])

    print(f"\n📊 Commodities loaded: {len(prep.series)}")
    print(f"\nDetails:")

    usable_count = 0
    for i, (key, series) in enumerate(prep.series.items(), 1):
        commodity = prep.meta.loc[prep.meta['key'] == key, 'commodity'].values[0]
        n_points = len(series)

        print(f"\n{i}. {commodity}")
        print(f"   Key: {key}")
        print(f"   Total points: {n_points}")

        # Calculate minimum points needed
        input_len = cfg['task']['input_len']
        horizon = cfg['task']['horizon']
        window_size = input_len + horizon

        train_split = cfg['split']['train']
        val_split = cfg['split']['val']
        test_split = cfg['split']['test']

        n_train = int(n_points * train_split)
        n_val = int(n_points * val_split)
        n_test = n_points - n_train - n_val

        print(f"   Split sizes: train={n_train}, val={n_val}, test={n_test}")
        print(f"   Window size needed: {window_size} (input={input_len} + horizon={horizon})")

        # Try creating loaders
        try:
            loaders, scaler, splits = build_loaders_for_series(series, cfg)

            train_batches = len(loaders['train'])
            val_batches = len(loaders['val'])
            test_batches = len(loaders['test'])

            print(f"   Loader batches: train={train_batches}, val={val_batches}, test={test_batches}")

            if train_batches > 0 and val_batches > 0 and test_batches > 0:
                print(f"   ✓ USABLE")
                usable_count += 1
            else:
                print(f"   ✗ SKIPPED (not enough data for all splits)")
                if train_batches == 0:
                    print(f"      - Train split too small ({n_train} points < {window_size} window)")
                if val_batches == 0:
                    print(f"      - Val split too small ({n_val} points < {window_size} window)")
                if test_batches == 0:
                    print(f"      - Test split too small ({n_test} points < {window_size} window)")
        except Exception as e:
            print(f"   ✗ ERROR: {e}")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Configured: {cfg['data']['n_commodities']} commodities")
    print(f"Loaded: {len(prep.series)} commodities")
    print(f"Usable: {usable_count} commodities")
    print(f"\n⚠️  Only {usable_count} commodities have enough data for training!")

    if usable_count < cfg['data']['n_commodities']:
        print(f"\n💡 Solutions:")
        print(f"   1. Reduce input_len: {cfg['task']['input_len']} → 12 or 15")
        print(f"   2. Change split ratio: more train, less val/test")
        print(f"   3. Check if commodities have sparse/missing data")
        print(f"   4. Use commodities with more data points")

    print("="*80)

if __name__ == "__main__":
    main()
