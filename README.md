# Loyalty Fraud Detection (Final Repository)

Unsupervised detection of internal abuse in a loyalty system (`!Fest` scenario).
The project identifies suspicious behavior at multiple aggregation levels and produces a final waiter-level risk ranking for manual review.

## What this repository contains

- Production-oriented Python pipeline in `models/`
- Feature engineering notebooks in `parquet/`
- Final ensemble logic in `models/waiter_ensemble.py`
- Reproducible dependencies in `requirements.txt`

## Detection pipeline

The modeling flow has three levels:

1. Card level (`person_id`) - anomaly signals on loyalty-card behavior.
2. Waiter-week level - weekly abnormal patterns per waiter.
3. Waiter-month level - monthly abnormal patterns per waiter.

These signals are unified and combined in a final ensemble at waiter level.

## Main scripts

- `models/models.py` - card-level IF / OCSVM / LOF experiments (+ synthetic support).
- `models/waiter_week_models.py` - waiter-week models (+ real vs synthetic evaluation).
- `models/waiter_month_models.py` - waiter-month models (+ real vs synthetic evaluation).
- `models/waiter_ensemble.py` - final waiter-level ensemble, including real vs synthetic comparison.

## Synthetic data support

Synthetic fraud generation is implemented in `models/synt_data_generation.py`.
Final waiter-level ensemble now supports synthetic evaluation through:

- `compare_waiter_ensemble_real_vs_synthetic(...)` function
- CLI flag `--synthetic` in `models/waiter_ensemble.py`

Example:

`python3 models/waiter_ensemble.py --synthetic --n-synthetic 500 --noise-scale 0.1`

## Reproducibility

1. Create environment and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Provide required input datasets locally (not committed here).
3. Run target script (example):
   - `python3 models/waiter_ensemble.py --top-n 20`
4. Output risk ranking is saved to:
   - `waiter_ensemble_risk.csv` (real)
   - `waiter_ensemble_risk_synthetic.csv` (synthetic mode)

## Data and artifacts policy

- Raw data files (`*.csv`, `*.parquet`) are intentionally excluded from Git.
- Generated artifacts and local environment files are excluded via `.gitignore`.
- Notebook outputs are cleared to keep the repository lightweight and review-friendly.