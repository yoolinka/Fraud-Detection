"""
Compare Isolation Forest, One-Class SVM, and LOF on waiter–week level features
(waiter_week_features.parquet)
"""
import os
import sys
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import FRAUD_WAITER_IDS, load_data

import importlib.util

_spec = importlib.util.spec_from_file_location("scaling", os.path.join(_script_dir, "scaling.py"))
_scaling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scaling)
scale_features = _scaling.scale_features

_client_models_spec = importlib.util.spec_from_file_location(
    "client_models", os.path.join(_script_dir, "models.py")
)
_client_models = importlib.util.module_from_spec(_client_models_spec)
_client_models_spec.loader.exec_module(_client_models)

from fit_and_evaluate import fit_and_evaluate
from synt_data_generation import generate_synthetic_data

WAITER_WEEK_FEATURES = [
    "share_new_clients",
    "bonusses_accum",
    "new_clients",
    "mean_check",
    "trn_per_day",
    "top1_client_share_norm",
    "share_unique_clients",
    "diff_share_of_trn",
]

WAITER_WEEK_SKEWED = [
    "bonusses_accum",
    "new_clients",
    "mean_check",
    "trn_per_day",
    "top1_client_share_norm",
]

def compare_waiter_week_models(
    waiter_features: Sequence[str] = WAITER_WEEK_FEATURES,
    waiter_skewed: Sequence[str] = WAITER_WEEK_SKEWED,
    agg_data: Optional[pd.DataFrame] = None,
    fraud_waiter_week_ids: Optional[Union[Sequence, np.ndarray]] = None,
    min_num_of_trn: int = 8,
    exclude_fraud_from_training: bool = True,
    num_of_trn: int = 1,
    place_num_of_waiters: int = 1,
    plot_scores_path: Optional[str] = None,
):
    """
    Load or accept waiter–week aggregates, scale, fit IF / OCSVM / LOF, print metrics.

    Parameters
    ----------
    agg_data : optional pre-built DataFrame (e.g. filtered). Must include waiter_week
    column or index, feature columns, and is_fraud unless fraud_waiter_week_ids given.
    fraud_waiter_week_ids : optional index keys for known fraud rows; if None, uses is_fraud.
    min_num_of_trn : keep rows with num_of_trn >= this (after load), default 8 as in notebook.

    Returns
    -------
    results_df, predictions, waiter_week_data, X_out, scores
    """

    if agg_data is None:
        _, _, waiter_week_data = load_data(
            num_of_trn=num_of_trn,
            place_num_of_waiters=place_num_of_waiters,
        )
    else:
        waiter_week_data = agg_data.copy()

    missing = [c for c in waiter_features if c not in waiter_week_data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    waiter_week_data = waiter_week_data[waiter_week_data["num_of_trn"] >= min_num_of_trn].copy()

    if fraud_waiter_week_ids is not None:
        ids = set(fraud_waiter_week_ids)
        y_fraud = waiter_week_data.index.isin(ids).astype(int).values
    elif "is_fraud" in waiter_week_data.columns:
        y_fraud = waiter_week_data["is_fraud"].astype(int).values
    else:
        raise ValueError("Provide fraud_waiter_week_ids or an is_fraud column on agg_data")

    n_fraud = int(y_fraud.sum())
    n_total = len(waiter_week_data)

    if exclude_fraud_from_training:
        train_mask = y_fraud == 0
        fit_subset = waiter_week_data.loc[train_mask]
        X_fit_df, X_eval_df = scale_features(
            data=waiter_week_data,
            features=waiter_features,
            skewed=waiter_skewed,
            scaler_type="standard",
            fit_data=fit_subset,
        )
        X_fit = X_fit_df.values
        X_eval = X_eval_df.values
        n_train = int(train_mask.sum())
    else:
        X_full = scale_features(
            data=waiter_week_data,
            features=waiter_features,
            skewed=waiter_skewed,
            scaler_type="standard",
        )
        X_fit = X_eval = np.asarray(X_full.values, dtype=np.float64)
        n_train = n_total

    results_df, predictions, scores = fit_and_evaluate(
        X_fit,
        y_fraud,
        X_eval=X_eval
    )

    print("=" * 60)
    print("Waiter–week anomaly detection — model comparison")
    print("=" * 60)
    print(f"Samples (total): {n_total}  |  Known fraud waiter-weeks: {n_fraud}")
    if exclude_fraud_from_training:
        print(f"Training on non-fraud only: {n_train} samples")
    else:
        print("Training on full data (fraud included in training)")
    print(f"Features ({len(waiter_features)}): {waiter_features}")
    print(f"Filter: num_of_trn >= {min_num_of_trn}")
    print()
    print(results_df.to_string(index=False))
    print()
    print("Metrics:")
    print("  - n_anomalies: rows flagged as anomaly (-1)")
    print("  - pct_flagged: percentage of all rows flagged")
    print("  - fraud_hit_rate: fraction of known frauds flagged as anomaly")
    print("  - recall@k / precision@k: top-k by anomaly score (k=10,20,50,100)")
    print("  - time_sec: fit+predict time in seconds")
    print()

    fraud_index = np.where(y_fraud.astype(bool))[0]
    print("Known fraud waiter-weeks — which model flagged them (-1 = anomaly):")
    print("-" * 60)
    for i in fraud_index:
        wid = waiter_week_data.index[i]
        row = (
            f"  waiter_week={wid}: "
            f"IF={predictions['iso'][i]}, OCSVM={predictions['ocsvm'][i]}, LOF={predictions['lof'][i]}"
        )
        print(row)
    print()

    if plot_scores_path:
        _client_models._plot_anomaly_score_distributions(scores, y_fraud, plot_scores_path)
        print(f"Anomaly score distributions saved to {plot_scores_path}")

    X_out = pd.DataFrame(X_eval, index=waiter_week_data.index, columns=waiter_features)
    return results_df, predictions, waiter_week_data, X_out, scores


def compare_waiter_week_real_vs_synthetic(
    waiter_features: Sequence[str] = WAITER_WEEK_FEATURES,
    waiter_skewed: Sequence[str] = WAITER_WEEK_SKEWED,
    agg_data: Optional[pd.DataFrame] = None,
    fraud_waiter_week_ids: Optional[Union[Sequence, np.ndarray]] = None,
    min_num_of_trn: int = 8,
    num_of_trn: int = 1,
    place_num_of_waiters: int = 1,
    n_synthetic: int = 500,
    noise_scale: float = 0.1,
    random_state: int = 42,
    plot_scores_path: Optional[str] = None,
) -> tuple:
    """
    Run IF / OCSVM / LOF on real waiter–week data and on a synthetic dataset
    (non-fraud rows + noisy resamples of real fraud), training on non-fraud only.
    Mirrors ``compare_real_vs_synthetic`` in ``models.py``.
    """
    if agg_data is None:
        _, _, waiter_week_data = load_data(
            num_of_trn=num_of_trn,
            place_num_of_waiters=place_num_of_waiters,
        )
    else:
        waiter_week_data = agg_data.copy()

    missing = [c for c in waiter_features if c not in waiter_week_data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    waiter_week_data = waiter_week_data[waiter_week_data["num_of_trn"] >= min_num_of_trn].copy()

    if fraud_waiter_week_ids is not None:
        ids = set(fraud_waiter_week_ids)
        y_fraud = waiter_week_data.index.isin(ids).astype(int).values
    elif "is_fraud" in waiter_week_data.columns:
        y_fraud = waiter_week_data["is_fraud"].astype(int).values
    else:
        y_fraud = waiter_week_data.index.isin(FRAUD_WAITER_IDS).astype(int).values

    n_fraud_real = int(y_fraud.sum())
    if n_fraud_real == 0:
        raise ValueError("No known fraud waiter-weeks; cannot build synthetic fraud.")

    waiter_week_data = waiter_week_data.copy()
    waiter_week_data["is_fraud"] = y_fraud.astype(int)

    train_mask = y_fraud == 0
    train_data = waiter_week_data.loc[train_mask]
    X_fit_real, X_eval_real = scale_features(
        data=waiter_week_data,
        features=waiter_features,
        skewed=waiter_skewed,
        scaler_type="standard",
        fit_data=train_data,
    )
    results_real, _, scores_real = fit_and_evaluate(
        X_fit_real.values,
        y_fraud,
        X_eval=X_eval_real.values,
    )
    results_real.insert(0, "dataset", "Real")

    synthetic = generate_synthetic_data(
        waiter_week_data,
        n_synthetic=n_synthetic,
        noise_scale=noise_scale,
        random_state=random_state,
    )
    y_synt = synthetic["is_fraud"].astype(int).values
    n_fraud_synt = int(y_synt.sum())
    train_synt = synthetic[synthetic["is_fraud"] == 0]
    X_fit_synt, X_eval_synt = scale_features(
        data=synthetic,
        features=waiter_features,
        skewed=waiter_skewed,
        scaler_type="standard",
        fit_data=train_synt,
    )
    results_synt, _, scores_synt = fit_and_evaluate(
        X_fit_synt.values,
        y_synt,
        X_eval=X_eval_synt.values,
    )
    results_synt.insert(0, "dataset", "Synthetic")

    print("=" * 70)
    print("Waiter–week: real vs synthetic — model comparison (train on non-fraud only)")
    print("=" * 70)
    print(f"Real:      n_total={len(waiter_week_data)}, n_fraud={n_fraud_real}")
    print(f"Synthetic: n_total={len(synthetic)}, n_fraud={n_fraud_synt}")
    print(f"Features ({len(waiter_features)}): num_of_trn >= {min_num_of_trn}")
    print()
    combined = pd.concat([results_real, results_synt], ignore_index=True)
    print(combined.to_string(index=False))
    print()
    print(
        "Metrics: fraud_hit_rate = fraction of known frauds flagged; "
        "recall@k / precision@k = top-k by score."
    )
    if plot_scores_path:
        base = os.path.join(
            os.path.dirname(plot_scores_path),
            os.path.splitext(os.path.basename(plot_scores_path))[0],
        )
        real_path = base + "_real.png"
        synt_path = base + "_synthetic.png"
        _client_models._plot_anomaly_score_distributions(scores_real, y_fraud, real_path)
        _client_models._plot_anomaly_score_distributions(scores_synt, y_synt, synt_path)
        print(f"Anomaly score distributions saved to {real_path}")
        print(f"Anomaly score distributions saved to {synt_path}")

    return results_real, results_synt, waiter_week_data, synthetic


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Waiter–week anomaly detection: IF, OCSVM, LOF")
    parser.add_argument("--synthetic", action="store_true", help="Compare real vs synthetic (default: real only)")
    parser.add_argument("--activity-state", type=int, default=2)
    parser.add_argument("--days-visits", type=int, default=2)
    parser.add_argument("--min-trn", type=int, default=8, help="num_of_trn lower bound (inclusive)")
    parser.add_argument("--n-synthetic", type=int, default=6000, help="Synthetic fraud samples (with --synthetic)")
    parser.add_argument("--plot", type=str, default=None, help="Path to save score distribution plot")
    args = parser.parse_args()
    default_plot = os.path.join(_project_root, "waiter_week_anomaly_score_distributions.png")
    if args.synthetic:
        compare_waiter_week_real_vs_synthetic(
            min_num_of_trn=args.min_trn,
            n_synthetic=args.n_synthetic,
            plot_scores_path=args.plot or default_plot,
        )
    else:
        compare_waiter_week_models(
            min_num_of_trn=args.min_trn,
            exclude_fraud_from_training=True,
            plot_scores_path=args.plot or default_plot,
        )
