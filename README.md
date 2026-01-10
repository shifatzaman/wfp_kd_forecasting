# WFP Bangladesh Commodity Price Forecasting with Multi-Teacher Knowledge Distillation

This project trains strong **teacher** time-series forecasting models and distills their knowledge into lightweight **student** models to achieve low MAE on commodity price forecasting.

It is designed to be:
- **Colab-friendly** (single entrypoint: `run.py`)
- **Config-driven** (all experiments controlled via YAML)
- **Experimentable** (grid-search over teacher/student/distillation options)
- **Reproducible** (seeded runs + saved configs + metrics)

## Dataset
WFP food price dataset (Bangladesh):
- Source CSV: `https://raw.githubusercontent.com/shifatzaman/datasets/refs/heads/main/wfp_food_prices_bgd_updated.csv`

### Dataset preparation rules implemented
1. **Normalize prices to BDT per kg** by parsing the `unit` column (e.g., `100kg` → price/100).
2. Evaluate on **5 commodities** within **one market** (default: `Dhaka`) using the 5 commodities with the most observations in that market.

## Knowledge Distillation (KD)
We implement KD sources/algorithms aligned with the survey paper:
- **Prediction (logit) distillation**: student mimics teacher predictions (regression analog of logits).
- **Feature distillation**: align intermediate representations with optional projection (general form in Eq. (4)).
- **Contrastive distillation**: align student and teacher feature spaces with an InfoNCE-style objective (Eq. (7)).
- **Multi-teacher distillation**: ensemble multiple teachers; optionally weight teachers based on validation error.

(See `docs/Survey on KD.pdf` for background.)

## Models
### Teachers
- **DLinear** (decomposition + linear heads)
- **PatchTST** (patch embedding + Transformer encoder)
- **N-BEATS** (stacked residual blocks)
You can enable any subset as a teacher ensemble.

### Students
- **MLP**
- **GRU**
- **KAN** (a small, readable spline-based KAN-inspired layer; intended as an approximation for experimentation)

## Quickstart (Colab)
```bash
!pip -q install -r requirements.txt
!python run.py --config configs/default.yaml
```

## Outputs
Each run creates a timestamped folder in `runs/`, including:
- `config_resolved.yaml` (exact config used)
- `metrics.json` (aggregate metrics)
- `per_series.csv` (metrics per commodity series)
- `history.csv` (train/val curves)
- A row appended to `runs/summary.csv` with the combination and performance

## Configuration
Edit `configs/default.yaml` to control:
- market + commodities
- input length / horizon
- train/val/test splits
- teacher models and hyperparameters
- student models and hyperparameters
- KD losses and weights
- grid search combinations

## Notes on MAE < 1
Achieving MAE < 1 depends on:
- unit normalization correctness
- commodity volatility
- forecast horizon
- scaling choices

This repo logs all combinations so you can iterate toward the best setup.

---
## Project structure
```
wfp_kd_forecasting/
  run.py
  configs/
  src/
    data/
    models/
    distill/
    utils/
  runs/
  requirements.txt
```
