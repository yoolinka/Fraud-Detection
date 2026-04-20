"""
Final waiter-level ensemble.

Signals used:
  Card level  (precomputed in waiter_level_data):
    iso_90, ocsvm_90, lof_90, share_active_clients_only_this_waiter,
    share_anomaly_weeks_iso, share_anomaly_weeks_ocsvm, share_anomaly_weeks_lof
  Waiter-week (computed fresh via IF/OCSVM/LOF on week features):
    week_iso_max, week_ocsvm_max, week_lof_max,
    week_iso_mean, week_ocsvm_mean, week_n_top5pct
  Waiter-month (computed fresh via IF/OCSVM/LOF on month features):
    month_iso_max, month_ocsvm_max, month_lof_max,
    month_iso_mean, month_ocsvm_mean, month_n_top5pct

Ensemble approaches compared:
  IF          — Isolation Forest on all unified features
  OCSVM       — One-Class SVM on all unified features
  LOF         — LOF on all unified features
  Fusion-2    — Rank fusion: IF + OCSVM on unified (50/50)
  Fusion-sig  — Rank fusion of raw sub-signals, no re-fitting
"""

import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import load_data, FEATURES
from fit_and_evaluate import fit_and_evaluate, fit_and_evaluate_per_model, _top_k_recall, _top_k_precision

import importlib.util

_spec = importlib.util.spec_from_file_location("scaling", os.path.join(_script_dir, "scaling.py"))
_scaling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scaling)
scale_features = _scaling.scale_features

# fmt: off
# Unified feature matrix columns (order defines how they appear in the output DataFrame)
WAITER_UNIFIED_FEATURES = [
    # --- card-level signals (from waiter_level_data, precomputed) ---
    "iso_90", "ocsvm_90", "lof_90",
    "share_active_clients_only_this_waiter",
    "share_anomaly_weeks_iso", "share_anomaly_weeks_ocsvm", "share_anomaly_weeks_lof",
    # --- waiter-week signals (aggregated from fresh model run) ---
    "week_iso_max", "week_ocsvm_max", "week_lof_max",
    "week_iso_mean", "week_ocsvm_mean",
    "week_n_top5pct",
    # --- waiter-month signals (aggregated from fresh model run) ---
    "month_iso_max", "month_ocsvm_max", "month_lof_max",
    "month_iso_mean", "month_ocsvm_mean",
    "month_n_top5pct",
]

# Fusion-sig: weights proportional to actual precision@100 per model per granularity.
#
# Card  (precision@100): IF=0.27, OCSVM=0.23, LOF=0.01
# Week  (precision@100): LOF=0.14, IF=0.12, OCSVM=0.05
# Month (precision@100): IF=0.17, LOF=0.13, OCSVM=0.07
#
# share_anomaly_weeks_* are binary (contamination-dependent) → half weight of max scores.
# *_mean scores carry less info than max → half weight of max.
_SIGNAL_WEIGHTS = {
    # card — IF and OCSVM strong, LOF near-zero
    "iso_90":                              2.7,
    "ocsvm_90":                            2.3,
    "lof_90":                              0.1,
    "share_active_clients_only_this_waiter": 1.0,
    "share_anomaly_weeks_iso":             0.6,
    "share_anomaly_weeks_ocsvm":           0.25,
    "share_anomaly_weeks_lof":             0.7,
    # week — LOF best, OCSVM weakest
    "week_lof_max":                        1.4,
    "week_iso_max":                        1.2,
    "week_ocsvm_max":                      0.5,
    "week_iso_mean":                       0.6,
    "week_ocsvm_mean":                     0.25,
    "week_n_top5pct":                      0.6,
    # month — IF best, OCSVM weakest
    "month_iso_max":                       1.7,
    "month_lof_max":                       1.3,
    "month_ocsvm_max":                     0.7,
    "month_iso_mean":                      0.85,
    "month_ocsvm_mean":                    0.35,
    "month_n_top5pct":                     0.85,
}
from waiter_month_models import WAITER_MONTH_FEATURES_ISO, WAITER_MONTH_FEATURES_OCSVM, WAITER_MONTH_FEATURES_LOF
from waiter_week_models import WAITER_WEEK_FEATURES_ISO, WAITER_WEEK_FEATURES_OCSVM, WAITER_WEEK_FEATURES_LOF
# Week / month feature sets for the per-granularity models (from tuned waiter_week/month_models).
# fmt: on

TOP_K_LIST = [5, 10, 20, 50]  # 14 = total known fraud waiters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank(arr: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1]; higher = more anomalous."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1) / n
    return ranks


def _extract_waiter_id(data: pd.DataFrame) -> pd.Series:
    """Return waiter_id for each row (str). Parses index if column absent."""
    if "waiter_id" in data.columns:
        return data["waiter_id"].astype(str)
    # index format: {place_id}_{waiter_num}_{date}  →  rsplit("_", 1) strips date
    return pd.Series(
        data.index.astype(str).str.rsplit("_", n=1).str[0],
        index=data.index,
    )


def _top5pct_threshold(scores: np.ndarray) -> float:
    return float(np.percentile(scores, 95))


def _metrics_row(name: str, score: np.ndarray, y_true: np.ndarray) -> dict:
    row = {"approach": name}
    row.update(_top_k_recall(score, y_true, TOP_K_LIST))
    row.update(_top_k_precision(score, y_true, TOP_K_LIST))
    return row


# ---------------------------------------------------------------------------
# Step 1: week-level signals → aggregate per waiter
# ---------------------------------------------------------------------------

def _run_week_model(
    waiter_week_data: pd.DataFrame,
    y_week: np.ndarray,
    n_estimators: int,
    n_neighbors: int,
) -> dict:
    """Run IF/OCSVM/LOF on waiter-week data; return raw score arrays."""
    train_mask = y_week == 0
    fit_sub = waiter_week_data.loc[train_mask]

    X_fit = {}
    X_eval = {}
    for key, feats in [("iso", WAITER_WEEK_FEATURES_ISO), ("ocsvm", WAITER_WEEK_FEATURES_OCSVM), ("lof", WAITER_WEEK_FEATURES_LOF)]:
        fit_df, eval_df = scale_features(
            data=waiter_week_data, features=feats, scaler_type="standard", fit_data=fit_sub
        )
        X_fit[key] = fit_df.values.astype(np.float64)
        X_eval[key] = eval_df.values.astype(np.float64)

    _, _, scores = fit_and_evaluate_per_model(
        X_fit, X_eval, y_week, n_neighbors=5, n_estimators=100
    )
    return scores


def _aggregate_week_signals(waiter_week_data: pd.DataFrame, scores: dict) -> pd.DataFrame:
    """Aggregate week-level scores per waiter_id."""
    wid = _extract_waiter_id(waiter_week_data)
    thr_iso = _top5pct_threshold(scores["iso"])

    agg = pd.DataFrame(
        {
            "waiter_id": wid.values,
            "iso": scores["iso"],
            "ocsvm": scores["ocsvm"],
            "lof": scores["lof"],
            "top5pct": (scores["iso"] > thr_iso).astype(int),
        }
    )
    result = (
        agg.groupby("waiter_id")
        .agg(
            week_iso_max=("iso", "max"),
            week_ocsvm_max=("ocsvm", "max"),
            week_lof_max=("lof", "max"),
            week_iso_mean=("iso", "mean"),
            week_ocsvm_mean=("ocsvm", "mean"),
            week_n_top5pct=("top5pct", "sum"),
        )
    )
    return result


# ---------------------------------------------------------------------------
# Step 2: month-level signals → aggregate per waiter
# ---------------------------------------------------------------------------

def _run_month_model(
    waiter_month_data: pd.DataFrame,
    y_month: np.ndarray,
    n_estimators: int,
    n_neighbors: int,
) -> dict:
    train_mask = y_month == 0
    fit_sub = waiter_month_data.loc[train_mask]

    X_fit = {}
    X_eval = {}
    for key, feats in [("iso", WAITER_MONTH_FEATURES_ISO), ("ocsvm", WAITER_MONTH_FEATURES_OCSVM), ("lof", WAITER_MONTH_FEATURES_LOF)]:
        fit_df, eval_df = scale_features(
            data=waiter_month_data, features=feats, scaler_type="standard", fit_data=fit_sub
        )
        X_fit[key] = fit_df.values.astype(np.float64)
        X_eval[key] = eval_df.values.astype(np.float64)

    _, _, scores = fit_and_evaluate_per_model(
        X_fit, X_eval, y_month, n_neighbors=20, n_estimators=500
    )
    return scores


def _aggregate_month_signals(waiter_month_data: pd.DataFrame, scores: dict) -> pd.DataFrame:
    wid = _extract_waiter_id(waiter_month_data)
    thr_iso = _top5pct_threshold(scores["iso"])

    agg = pd.DataFrame(
        {
            "waiter_id": wid.values,
            "iso": scores["iso"],
            "ocsvm": scores["ocsvm"],
            "lof": scores["lof"],
            "top5pct": (scores["iso"] > thr_iso).astype(int),
        }
    )
    result = (
        agg.groupby("waiter_id")
        .agg(
            month_iso_max=("iso", "max"),
            month_ocsvm_max=("ocsvm", "max"),
            month_lof_max=("lof", "max"),
            month_iso_mean=("iso", "mean"),
            month_ocsvm_mean=("ocsvm", "mean"),
            month_n_top5pct=("top5pct", "sum"),
        )
    )
    return result


# ---------------------------------------------------------------------------
# Step 3: build unified feature matrix
# ---------------------------------------------------------------------------

def _build_unified(
    waiter_data: pd.DataFrame,
    week_agg: pd.DataFrame,
    month_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join waiter_level_data (card signals) with week/month aggregates.
    Missing week/month entries (waiter not in those datasets) → filled with 0.
    """
    card_cols = [c for c in WAITER_UNIFIED_FEATURES if c in waiter_data.columns]
    unified = waiter_data[card_cols].copy()

    unified = unified.join(week_agg, how="left")
    unified = unified.join(month_agg, how="left")

    # Waiters with no matching weeks/months get 0 (no anomalous signal)
    week_cols = week_agg.columns.tolist()
    month_cols = month_agg.columns.tolist()
    unified[week_cols] = unified[week_cols].fillna(0)
    unified[month_cols] = unified[month_cols].fillna(0)

    # Keep only features that actually exist after the join
    available = [f for f in WAITER_UNIFIED_FEATURES if f in unified.columns]
    return unified[available]


# ---------------------------------------------------------------------------
# Step 4: five ensemble approaches
# ---------------------------------------------------------------------------

def _run_unified_models(
    unified: pd.DataFrame,
    y_fraud: np.ndarray,
    features: list,
    n_estimators: int,
    n_neighbors: int,
) -> tuple[pd.DataFrame, dict, dict]:
    """Fit IF/OCSVM/LOF on unified features; return results_df, predictions, scores."""
    train_mask = y_fraud == 0
    train_data = unified.loc[train_mask]

    X_fit, X_eval = scale_features(
        data=unified, features=features, scaler_type="standard", fit_data=train_data
    )
    return fit_and_evaluate(
        X_fit.values.astype(np.float64),
        y_fraud,
        X_eval=X_eval.values.astype(np.float64),
        n_neighbors=n_neighbors,
        n_estimators=n_estimators,
    )


def _fusion2_score(scores: dict) -> np.ndarray:
    """Rank fusion of IF + OCSVM on unified features (50/50)."""
    return 0.5 * _rank(scores["iso"]) + 0.5 * _rank(scores["ocsvm"])


def _fusion_signals_score(unified: pd.DataFrame, features: list) -> np.ndarray:
    """
    Rank fusion of raw sub-signals without re-fitting any model on unified features.
    Each signal is rank-normalised; weighted sum uses _SIGNAL_WEIGHTS.
    """
    total_w = sum(_SIGNAL_WEIGHTS.get(f, 1.0) for f in features)
    score = np.zeros(len(unified), dtype=np.float64)
    for f in features:
        col = unified[f].fillna(0).values.astype(np.float64)
        w = _SIGNAL_WEIGHTS.get(f, 1.0)
        score += (w / total_w) * _rank(col)
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare_waiter_ensemble(
    activity_state: int = 2,
    days_visits: int = 2,
    min_working_days: int = 5,
    min_num_of_trn_week: int = 8,
    min_num_of_trn_month: int = 10,
    n_estimators: int = 200,
    n_neighbors: int = 10,
    top_n: int = 20,
    scores_csv_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load all granularities → build unified features → compare 5 approaches.

    Returns
    -------
    metrics_df : pd.DataFrame
        One row per approach with recall@k and precision@k.
    risk_df : pd.DataFrame
        One row per waiter, sorted by Fusion-sig score (best ensemble).
        Columns: ensemble_rank, waiter_id, is_fraud, all unified features, all approach scores.
    """
    print("Loading data …")
    _, client_data, waiter_week_data, waiter_month_data, waiter_data = load_data(
        activity_state=activity_state,
        days_visits=days_visits,
        total_num_of_trn=8,

        num_of_trn = 8,
        min_working_days=5,
        place_num_of_waiters = 2
    )

    # --- filters ---
    waiter_week_data = waiter_week_data[waiter_week_data["num_of_trn"] >= min_num_of_trn_week].copy()
    waiter_month_data = waiter_month_data[waiter_month_data["num_of_trn"] >= min_num_of_trn_month].copy()

    y_waiter = waiter_data["is_fraud"].astype(int).values
    y_week = waiter_week_data["is_fraud"].astype(int).values
    y_month = waiter_month_data["is_fraud"].astype(int).values

    n_fraud = int(y_waiter.sum())
    n_total = len(waiter_data)
    print(f"Waiters: {n_total} total, {n_fraud} known fraud")
    print(f"Waiter-weeks: {len(waiter_week_data)} | Waiter-months: {len(waiter_month_data)}")

    # --- sub-models ---
    print("Running waiter-week model …")
    week_scores = _run_week_model(waiter_week_data, y_week, n_estimators, n_neighbors)
    week_agg = _aggregate_week_signals(waiter_week_data, week_scores)

    print("Running waiter-month model …")
    month_scores = _run_month_model(waiter_month_data, y_month, n_estimators, n_neighbors)
    month_agg = _aggregate_month_signals(waiter_month_data, month_scores)

    # --- unified features ---
    print("Building unified feature matrix …")
    unified = _build_unified(waiter_data, week_agg, month_agg)
    features = unified.columns.tolist()
    n_features = len(features)

    # --- unified models (IF / OCSVM / LOF) ---
    print("Running unified models …")
    results_base, _, scores_unified = _run_unified_models(
        unified, y_waiter, features, n_estimators, n_neighbors
    )

    # --- ensemble scores ---
    score_fusion2 = _fusion2_score(scores_unified)
    score_fusion_sig = _fusion_signals_score(unified, features)

    # --- metrics ---
    rows = []
    for name, score in [
        ("IF (unified)", scores_unified["iso"]),
        ("OCSVM (unified)", scores_unified["ocsvm"]),
        ("LOF (unified)", scores_unified["lof"]),
        ("Fusion-2 (IF+OCSVM)", score_fusion2),
        ("Fusion-sig (sub-signals)", score_fusion_sig),
    ]:
        rows.append(_metrics_row(name, score, y_waiter))
    metrics_df = pd.DataFrame(rows)

    # --- risk ranking (by Fusion-sig) ---
    risk_df = pd.DataFrame(
        {
            "waiter_id": waiter_data.index,
            "is_fraud": y_waiter,
            **{f: unified[f].values for f in features},
            "score_if": scores_unified["iso"],
            "score_ocsvm": scores_unified["ocsvm"],
            "score_lof": scores_unified["lof"],
            "score_fusion2": score_fusion2,
            "score_fusion_sig": score_fusion_sig,
        }
    )
    risk_df = risk_df.sort_values("score_fusion_sig", ascending=False).reset_index(drop=True)
    risk_df.insert(0, "ensemble_rank", risk_df.index + 1)

    # --- print results ---
    _print_results(metrics_df, n_total, n_fraud, n_features, top_n, risk_df)

    if scores_csv_path:
        d = os.path.dirname(os.path.abspath(scores_csv_path))
        if d:
            os.makedirs(d, exist_ok=True)
        risk_df.to_csv(scores_csv_path, index=False)
        print(f"\nRisk ranking saved to {scores_csv_path}")

    return metrics_df, risk_df


def _print_results(
    metrics_df: pd.DataFrame,
    n_total: int,
    n_fraud: int,
    n_features: int,
    top_n: int,
    risk_df: pd.DataFrame,
) -> None:
    print()
    print("=" * 70)
    print("Waiter-level ensemble — final comparison")
    print("=" * 70)
    print(f"Waiters: {n_total} total | Known fraud: {n_fraud} | Features: {n_features}")
    print()
    print(metrics_df.to_string(index=False))
    print()
    print(f"Top-{top_n} risk ranking (by OCSVM (unified)):")
    print("-" * 70)
    cols = ["ensemble_rank", "waiter_id", "is_fraud", "score_fusion_sig", "score_fusion2", "score_if", "score_ocsvm"]
    print(risk_df[cols].sort_values("score_ocsvm", ascending=False).head(top_n).to_string(index=False))
    print()
    n_fraud_in_top = int(risk_df.head(top_n)["is_fraud"].sum())
    print(f"Fraud in top-{top_n}: {n_fraud_in_top} / {n_fraud}  "
          f"(precision={n_fraud_in_top/top_n:.2f}, recall={n_fraud_in_top/n_fraud:.2f})")


def top_n_risk(risk_df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Return top-N rows (default 14 = total known fraud waiters)."""
    return risk_df.head(n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Waiter-level ensemble anomaly detection")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-working-days", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--scores-csv",
        type=str,
        default=os.path.join(_project_root, "waiter_ensemble_risk.csv"),
    )
    args = parser.parse_args()

    metrics, risk = compare_waiter_ensemble(
        min_working_days=args.min_working_days,
        n_estimators=args.n_estimators,
        top_n=args.top_n,
        scores_csv_path=args.scores_csv,
    )
