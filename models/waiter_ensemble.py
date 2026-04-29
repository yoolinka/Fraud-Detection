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

Final unified meta-models (trained on all unified features):
  IF (iso)    — Isolation Forest
  OCSVM       — One-Class SVM
  LOF         — Local Outlier Factor
"""

import os
import sys
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import load_data
from fit_and_evaluate import fit_and_evaluate, fit_and_evaluate_per_model, _top_k_recall, _top_k_precision

import importlib.util

_spec = importlib.util.spec_from_file_location("scaling", os.path.join(_script_dir, "scaling.py"))
_scaling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scaling)
scale_features = _scaling.scale_features

WAITER_UNIFIED_FEATURES = [
    # --- card-level signals ---
    "iso_90", "ocsvm_90",
    "share_active_clients_only_this_waiter",
    # --- waiter-month signals ---
    "month_iso_max", 
    "month_ocsvm_max", 
    "month_lof_max",
    "month_iso_mean", 
    "month_ocsvm_mean",
    "month_n_top5pct",
]

from waiter_month_models import WAITER_MONTH_FEATURES_ISO, WAITER_MONTH_FEATURES_OCSVM, WAITER_MONTH_FEATURES_LOF
from waiter_week_models import WAITER_WEEK_FEATURES_ISO, WAITER_WEEK_FEATURES_OCSVM, WAITER_WEEK_FEATURES_LOF

TOP_K_LIST = [5, 10, 14, 20, 50]

# Extended top-k for synthetic waiter evaluation (includes 50, 100, 200).
SYNTHETIC_TOP_K_LIST = [50, 100, 200]


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


def _metrics_row(
    name: str,
    score: np.ndarray,
    y_true: np.ndarray,
    top_k_list: Optional[list[int]] = None,
) -> dict:
    k_list = TOP_K_LIST if top_k_list is None else top_k_list
    row = {"approach": name}
    row.update(_top_k_recall(score, y_true, k_list))
    row.update(_top_k_precision(score, y_true, k_list))
    return row


def _build_unified_from_pipeline(
    activity_state: int,
    days_visits: int,
    min_working_days: int,
    min_num_of_trn_week: int,
    min_num_of_trn_month: int,
    n_estimators: int,
    n_neighbors: int,
) -> tuple[pd.DataFrame, np.ndarray, list, pd.DataFrame]:
    """
    Load data, run week/month sub-models, return unified feature matrix, y_waiter, feature names, waiter_data.
    """
    _, _, waiter_week_data, waiter_month_data, waiter_data = load_data(
        activity_state=activity_state,
        days_visits=days_visits,
        total_num_of_trn=8,
        num_of_trn=8,
        min_working_days=min_working_days,
        place_num_of_waiters=2,
    )
    waiter_week_data = waiter_week_data[waiter_week_data["num_of_trn"] >= min_num_of_trn_week].copy()
    waiter_month_data = waiter_month_data[waiter_month_data["num_of_trn"] >= min_num_of_trn_month].copy()
    waiter_data = waiter_data.copy()
    waiter_data["is_fraud"] = waiter_data["is_fraud"].astype(int)

    y_week = waiter_week_data["is_fraud"].astype(int).values
    y_month = waiter_month_data["is_fraud"].astype(int).values
    y_waiter = waiter_data["is_fraud"].astype(int).values

    week_scores = _run_week_model(waiter_week_data, y_week, n_estimators, n_neighbors)
    week_agg = _aggregate_week_signals(waiter_week_data, week_scores)
    month_scores = _run_month_model(waiter_month_data, y_month, n_estimators, n_neighbors)
    month_agg = _aggregate_month_signals(waiter_month_data, month_scores)
    unified = _build_unified(waiter_data, week_agg, month_agg)
    features = unified.columns.tolist()
    return unified, y_waiter, features, waiter_data


def _synthetic_unified_clamped(
    unified: pd.DataFrame,
    y_waiter: np.ndarray,
    features: list,
    n_synthetic: int,
    noise_scale: float,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Clamped additive noise on resampled real-fraud rows (same idea as card-level ``evaluate_card_synthetic_auc``).
    Stack: all non-fraud unified rows + synthetic fraud rows.
    """
    fraud_mask = y_waiter == 1
    nf_mask = y_waiter == 0
    fraud_u = unified.loc[fraud_mask, features]
    nf_u = unified.loc[nf_mask, features]
    if len(fraud_u) == 0:
        raise ValueError("No fraud waiters in unified matrix; cannot build clamped synthetic fraud.")
    rng = np.random.default_rng(random_state)
    sampled = fraud_u.sample(n=n_synthetic, replace=True, random_state=random_state)
    nf_std = nf_u.std().values
    f_min = fraud_u.min().values
    f_max = fraud_u.max().values
    noise = rng.normal(0, 1, sampled.shape) * (noise_scale * nf_std)
    synth_vals = np.clip(sampled.values + noise, f_min, f_max)
    synth_df = pd.DataFrame(synth_vals, columns=features)
    synth_df.index = [f"__synth_clamped_{random_state}_{i}" for i in range(n_synthetic)]

    nf_part = unified.loc[nf_mask, features].copy()
    unified_synt = pd.concat([nf_part, synth_df], axis=0)
    y_synt = np.concatenate([np.zeros(len(nf_part), dtype=int), np.ones(n_synthetic, dtype=int)])
    return unified_synt, y_synt


def _synthetic_unified_interp(
    unified: pd.DataFrame,
    y_waiter: np.ndarray,
    features: list,
    n_synthetic: int,
    alpha: float,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Interpolation (1-α)*fraud + α*non-fraud feature draws — ensemble_synthetic_test / card AUC style.
    """
    fraud_mask = y_waiter == 1
    nf_mask = y_waiter == 0
    fraud_u = unified.loc[fraud_mask, features]
    nf_u = unified.loc[nf_mask, features]
    if len(fraud_u) == 0 or len(nf_u) == 0:
        raise ValueError("Need both fraud and non-fraud waiters for interpolation synthetic data.")
    fraud_sampled = fraud_u.sample(n=n_synthetic, replace=True, random_state=random_state).values
    nf_sampled = nf_u.sample(n=n_synthetic, replace=True, random_state=random_state).values
    synth_vals = (1.0 - alpha) * fraud_sampled + alpha * nf_sampled
    synth_df = pd.DataFrame(synth_vals, columns=features)
    synth_df.index = [f"__synth_interp_{random_state}_{i}" for i in range(n_synthetic)]

    nf_part = unified.loc[nf_mask, features].copy()
    unified_synt = pd.concat([nf_part, synth_df], axis=0)
    y_synt = np.concatenate([np.zeros(len(nf_part), dtype=int), np.ones(n_synthetic, dtype=int)])
    return unified_synt, y_synt


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

    week_cols = week_agg.columns.tolist()
    month_cols = month_agg.columns.tolist()
    unified[week_cols] = unified[week_cols].fillna(0)
    unified[month_cols] = unified[month_cols].fillna(0)

    available = [f for f in WAITER_UNIFIED_FEATURES if f in unified.columns]
    return unified[available]

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
    Full pipeline: load all granularities → build unified features → IF / OCSVM / LOF on unified matrix.

    Returns
    -------
    metrics_df : pd.DataFrame
        One row per model with recall@k and precision@k.
    risk_df : pd.DataFrame
        One row per waiter, sorted by OCSVM score (descending).
        Columns: ensemble_rank, waiter_id, is_fraud, unified features, score_if, score_ocsvm, score_lof.
    """
    print("Loading data …")
    _, client_data, waiter_week_data, waiter_month_data, waiter_data = load_data(
        activity_state=activity_state,
        days_visits=days_visits,
        total_num_of_trn=8,
        num_of_trn=8,
        min_working_days=min_working_days,
        place_num_of_waiters=2,
    )
    waiter_week_data = waiter_week_data[waiter_week_data["num_of_trn"] >= min_num_of_trn_week].copy()
    waiter_month_data = waiter_month_data[waiter_month_data["num_of_trn"] >= min_num_of_trn_month].copy()

    y_waiter = waiter_data["is_fraud"].astype(int).values
    y_week = waiter_week_data["is_fraud"].astype(int).values
    y_month = waiter_month_data["is_fraud"].astype(int).values

    n_fraud = int(y_waiter.sum())
    n_total = len(waiter_data)
    print(f"Waiters: {n_total} total, {n_fraud} known fraud")
    print(f"Waiter-weeks: {len(waiter_week_data)} | Waiter-months: {len(waiter_month_data)}")

    print("Running waiter-week model …")
    week_scores = _run_week_model(waiter_week_data, y_week, n_estimators, n_neighbors)
    week_agg = _aggregate_week_signals(waiter_week_data, week_scores)

    print("Running waiter-month model …")
    month_scores = _run_month_model(waiter_month_data, y_month, n_estimators, n_neighbors)
    month_agg = _aggregate_month_signals(waiter_month_data, month_scores)

    print("Building unified feature matrix …")
    unified = _build_unified(waiter_data, week_agg, month_agg)
    features = unified.columns.tolist()
    n_features = len(features)

    print("Running unified models …")
    _, _, scores_unified = _run_unified_models(
        unified, y_waiter, features, n_estimators, n_neighbors
    )

    rows = []
    for name, score in [
        ("IF (unified)", scores_unified["iso"]),
        ("OCSVM (unified)", scores_unified["ocsvm"]),
        ("LOF (unified)", scores_unified["lof"]),
    ]:
        rows.append(_metrics_row(name, score, y_waiter))
    metrics_df = pd.DataFrame(rows)

    # --- risk ranking ---
    risk_df = pd.DataFrame(
        {
            "waiter_id": waiter_data.index,
            "is_fraud": y_waiter,
            **{f: unified[f].values for f in features},
            "score_if": scores_unified["iso"],
            "score_ocsvm": scores_unified["ocsvm"],
            "score_lof": scores_unified["lof"],
        }
    )
    risk_df = risk_df.sort_values("score_ocsvm", ascending=False).reset_index(drop=True)
    risk_df.insert(0, "ensemble_rank", risk_df.index + 1)

    _print_results(metrics_df, n_total, n_fraud, n_features, top_n, risk_df)

    if scores_csv_path:
        d = os.path.dirname(os.path.abspath(scores_csv_path))
        if d:
            os.makedirs(d, exist_ok=True)
        risk_df.to_csv(scores_csv_path, index=False)
        print(f"\nRisk ranking saved to {scores_csv_path}")

    return metrics_df, risk_df


def compare_waiter_ensemble_real_vs_synthetic(
    activity_state: int = 2,
    days_visits: int = 2,
    min_working_days: int = 5,
    min_num_of_trn_week: int = 8,
    min_num_of_trn_month: int = 10,
    n_estimators: int = 200,
    n_neighbors: int = 10,
    n_synthetic: int = 500,
    noise_scale: float = 0.1,
    random_state: int = 42,
    top_n: int = 20,
    real_scores_csv_path: Optional[str] = None,
    synthetic_scores_csv_path: Optional[str] = None,
    synthetic_mode: Literal["unified_interp", "unified_clamped"] = "unified_interp",
    interp_alpha: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare waiter-level ensemble metrics on real vs synthetic data.

    Default ``ensemble_synthetic_test`` / card-style generation on the **unified** feature matrix after the real
    pipeline:

    - ``unified_interp`` — blend ``(1-α)·fraud + α·non-fraud`` (interpolation).
    - ``unified_clamped`` — resample fraud rows + Gaussian noise clamped to fraud min/max (σ scales with ``noise_scale``).
    """
    metrics_real, risk_real = compare_waiter_ensemble(
        activity_state=activity_state,
        days_visits=days_visits,
        min_working_days=min_working_days,
        min_num_of_trn_week=min_num_of_trn_week,
        min_num_of_trn_month=min_num_of_trn_month,
        n_estimators=n_estimators,
        n_neighbors=n_neighbors,
        top_n=top_n,
        scores_csv_path=real_scores_csv_path,
    )
    metrics_real = metrics_real.copy()
    metrics_real.insert(0, "dataset", "Real")

    print("\nBuilding synthetic evaluation data …")
    unified, y_waiter, features, waiter_data = _build_unified_from_pipeline(
        activity_state=activity_state,
        days_visits=days_visits,
        min_working_days=min_working_days,
        min_num_of_trn_week=min_num_of_trn_week,
        min_num_of_trn_month=min_num_of_trn_month,
        n_estimators=n_estimators,
        n_neighbors=n_neighbors,
    )

    if synthetic_mode == "unified_interp":
        print(f"  mode=unified_interp (α={interp_alpha}) — same family as ensemble_synthetic_test / card synthetic AUC")
        unified_synt, y_waiter_synt = _synthetic_unified_interp(
            unified, y_waiter, features, n_synthetic, interp_alpha, random_state
        )
        features_synt = features
    elif synthetic_mode == "unified_clamped":
        print(f"  mode=unified_clamped (σ scale={noise_scale}) — clamped noise vs non-fraud std")
        unified_synt, y_waiter_synt = _synthetic_unified_clamped(
            unified, y_waiter, features, n_synthetic, noise_scale, random_state
        )
        features_synt = features

    _, _, scores_unified_synt = _run_unified_models(
        unified_synt, y_waiter_synt, features_synt, n_estimators, n_neighbors
    )

    rows_synt = []
    for name, score in [
        ("IF (unified)", scores_unified_synt["iso"]),
        ("OCSVM (unified)", scores_unified_synt["ocsvm"]),
        ("LOF (unified)", scores_unified_synt["lof"]),
    ]:
        rows_synt.append(
            _metrics_row(name, score, y_waiter_synt, top_k_list=SYNTHETIC_TOP_K_LIST)
        )
    metrics_synt = pd.DataFrame(rows_synt)
    metrics_synt.insert(0, "dataset", "Synthetic")

    waiter_ids = unified_synt.index
    risk_synt = pd.DataFrame(
        {
            "waiter_id": waiter_ids,
            "is_fraud": y_waiter_synt,
            **{f: unified_synt[f].values for f in features_synt},
            "score_if": scores_unified_synt["iso"],
            "score_ocsvm": scores_unified_synt["ocsvm"],
            "score_lof": scores_unified_synt["lof"],
        }
    )
    risk_synt = risk_synt.sort_values("score_ocsvm", ascending=False).reset_index(drop=True)
    risk_synt.insert(0, "ensemble_rank", risk_synt.index + 1)

    print()
    print("=" * 70)
    print("Waiter-level ensemble — real vs synthetic comparison")
    print("=" * 70)
    print(
        f"Real: n_total={len(risk_real)}, n_fraud={int(risk_real['is_fraud'].sum())} | "
        f"Synthetic: n_total={len(risk_synt)}, n_fraud={int(risk_synt['is_fraud'].sum())}"
    )
    print(
        f"Synthetic: mode={synthetic_mode}, n_synthetic={n_synthetic}, random_state={random_state}"
        + (f", interp_alpha={interp_alpha}" if synthetic_mode == "unified_interp" else "")
        + (f", noise_scale={noise_scale}" if synthetic_mode == "unified_clamped" else "")
    )
    print()
    print("Real metrics:")
    print(metrics_real.to_string(index=False))
    print()
    print("Synthetic metrics (top-k includes 50, 100, 200):")
    print(metrics_synt.to_string(index=False))

    if synthetic_scores_csv_path:
        d = os.path.dirname(os.path.abspath(synthetic_scores_csv_path))
        if d:
            os.makedirs(d, exist_ok=True)
        risk_synt.to_csv(synthetic_scores_csv_path, index=False)
        print(f"\nSynthetic risk ranking saved to {synthetic_scores_csv_path}")

    return metrics_real, metrics_synt, risk_real, risk_synt


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
    print(f"Top-{top_n} risk ranking (by OCSVM):")
    print("-" * 70)
    cols = ["ensemble_rank", "waiter_id", "is_fraud", "score_if", "score_ocsvm", "score_lof"]
    print(risk_df[cols].sort_values("score_ocsvm", ascending=False).head(top_n).to_string(index=False))
    print()
    n_fraud_in_top = int(risk_df.head(top_n)["is_fraud"].sum())
    print(f"Fraud in top-{top_n}: {n_fraud_in_top} / {n_fraud}  "
          f"(precision={n_fraud_in_top/top_n:.2f}, recall={n_fraud_in_top/n_fraud:.2f})")

    missed = risk_df[(risk_df["is_fraud"] == 1) & (risk_df["ensemble_rank"] > top_n)].copy()
    print()
    print(f"Known frauds outside OCSVM top-{top_n} (not in the table above):")
    print("-" * 70)
    if len(missed) == 0:
        print("(none — all known fraud waiters appear in the top list.)")
    else:
        print(missed[cols].sort_values("score_ocsvm", ascending=False).to_string(index=False))
        print(f"\nCount: {len(missed)} fraud waiter(s) ranked below #{top_n} by OCSVM.")


def top_n_risk(risk_df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Return top-N rows (default 14 = total known fraud waiters)."""
    return risk_df.head(n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Waiter-level ensemble anomaly detection")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-working-days", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--synthetic", action="store_true", help="Compare real vs synthetic ensemble results")
    parser.add_argument("--n-synthetic", type=int, default=200, help="Synthetic fraud samples for --synthetic")
    parser.add_argument("--noise-scale", type=float, default=0.02, help="Noise scale for unified_clamped")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for synthetic generation")
    parser.add_argument(
        "--synthetic-mode",
        type=str,
        default="unified_interp",
        choices=["unified_interp", "unified_clamped"],
        help="unified_interp / unified_clamped = ensemble_synthetic_test style on unified features",
    )
    parser.add_argument(
        "--interp-alpha",
        type=float,
        default=0.02,
        help="Interpolation α for unified_interp (blend fraud vs non-fraud draws)",
    )
    parser.add_argument(
        "--scores-csv",
        type=str,
        default=os.path.join(_project_root, "waiter_ensemble_risk.csv"),
    )
    parser.add_argument(
        "--synthetic-scores-csv",
        type=str,
        default=os.path.join(_project_root, "waiter_ensemble_risk_synthetic.csv"),
        help="Path for synthetic risk ranking CSV (used with --synthetic)",
    )
    args = parser.parse_args()

    if args.synthetic:
        compare_waiter_ensemble_real_vs_synthetic(
            min_working_days=args.min_working_days,
            n_estimators=args.n_estimators,
            top_n=args.top_n,
            n_synthetic=args.n_synthetic,
            noise_scale=args.noise_scale,
            random_state=args.random_state,
            real_scores_csv_path=args.scores_csv,
            synthetic_scores_csv_path=args.synthetic_scores_csv,
            synthetic_mode=args.synthetic_mode,
            interp_alpha=args.interp_alpha,
        )
    else:
        compare_waiter_ensemble(
            min_working_days=args.min_working_days,
            n_estimators=args.n_estimators,
            top_n=args.top_n,
            scores_csv_path=args.scores_csv,
        )
