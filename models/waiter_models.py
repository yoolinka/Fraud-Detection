"""
Compare three anomaly detection models (Isolation Forest, One-Class SVM, LOF) for fraud waiters.
Uses waiter-level features: iso_90, ocsvm_90, lof_90, share_active_clients_only_this_waiter.
"""
import sys
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import load_data, FEATURES, SKEWED

import importlib.util
_spec = importlib.util.spec_from_file_location("scaling", os.path.join(_script_dir, "scaling.py"))
_scaling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scaling)
scale_features = _scaling.scale_features

# Import client-level fit_and_evaluate from models.py in the same directory
_client_models_spec = importlib.util.spec_from_file_location(
    "client_models", os.path.join(_script_dir, "models.py")
)
_client_models = importlib.util.module_from_spec(_client_models_spec)
_client_models_spec.loader.exec_module(_client_models)
from fit_and_evaluate import fit_and_evaluate

# Waiter-level features (from waiter_models.ipynb): client anomaly score 90th percentile + share
WAITER_FEATURES = [
    "iso_90",
    "ocsvm_90",
    "lof_90",
    "share_active_clients_only_this_waiter",
    'share_anomaly_weeks_iso',
    'share_anomaly_weeks_ocsvm',
    'share_anomaly_weeks_lof'
]
WAITER_SKEWED: list = []


def compare_waiter_models(
    activity_state: int = 2,
    days_visits: int = 2,
    min_working_days: int = 20,
    exclude_fraud_from_training: bool = True,
    plot_scores_path: Optional[str] = None,
    waiter_features: list = WAITER_FEATURES,
    waiter_skewed: list = WAITER_SKEWED,
    waiter_data: pd.DataFrame = None,
    total_num_of_trn: int = 8,
):
    """
    Load client data, compute client-level anomaly scores, build waiter-level features,
    then run Isolation Forest, One-Class SVM, and LOF on waiters. Print same metrics as models.py:
    hit_rate, recall@k, precision@k, n_anomalies, pct_flagged, time_sec.
    """
    df, client_data, _, _ = load_data(activity_state=activity_state, days_visits=days_visits)
    if "top_waiter_id" not in client_data.columns:
        raise ValueError("client_data must contain 'top_waiter_id' (from client_level_features)")

    # --- Client-level: get anomaly scores (fit on all or non-fraud only) ---
    train_mask = ~client_data["is_fraud"].values
    train_data = client_data.loc[train_mask]
    X_fit_cl, X_eval_cl = scale_features(
        data=client_data,
        scaler_type="standard",
        features=FEATURES,
        skewed=SKEWED,
        fit_data=train_data,
    )
    y_client = client_data["is_fraud"].astype(int).values
    _, _, scores_cl = fit_and_evaluate(
        X_fit_cl.values,
        y_client,
        X_eval=X_eval_cl.values,
    )
    anomaly_scores = pd.DataFrame(
        {
            "anomaly_score_iso": scores_cl["iso"],
            "anomaly_score_ocsvm": scores_cl["ocsvm"],
            "anomaly_score_lof": scores_cl["lof"],
        },
        index=client_data.index,
    )
    client_data = client_data.merge(anomaly_scores, left_index=True, right_index=True)
    df = df.merge(anomaly_scores, left_on="person_id", right_index=True, how="left")

    # --- Build waiter-level data and filter by working_days ---
    if waiter_data is None:
        _, _, _, waiter_data = load_data(
            activity_state=activity_state,
            days_visits=days_visits,
            total_num_of_trn=total_num_of_trn,
        )
    waiter_data = waiter_data[waiter_data["working_days"] > min_working_days].copy()

    y_fraud = waiter_data["is_fraud"].astype(int).values
    n_fraud = y_fraud.sum()
    n_total = len(waiter_data)

    if exclude_fraud_from_training:
        non_fraud_waiters = waiter_data[~waiter_data["is_fraud"]]
        X_fit_df, X_eval_df = scale_features(
            data=waiter_data,
            features=waiter_features,
            skewed=waiter_skewed,
            scaler_type="standard",
            fit_data=non_fraud_waiters,
        )
        X_fit = X_fit_df.values
        X_eval = X_eval_df.values
        n_train = len(non_fraud_waiters)
    else:
        X_full = scale_features(
            data=waiter_data,
            features=waiter_features,
            skewed=waiter_skewed,
            scaler_type="standard",
        )
        X_fit = X_eval = np.asarray(X_full.values, dtype=np.float64)
        n_train = n_total

    results_df, predictions, scores = fit_and_evaluate(
        X_fit,
        y_fraud,
        X_eval=X_eval,
    )

    print("=" * 60)
    print("Waiter anomaly detection — model comparison")
    print("=" * 60)
    print(f"Samples (total): {n_total}  |  Known fraud waiters: {n_fraud}")
    if exclude_fraud_from_training:
        print(f"Training on non-fraud waiters only: {n_train} samples")
    else:
        print("Training on full data (fraud included in training)")
    print(f"Features: {waiter_features}")
    print(f"Filter: working_days > {min_working_days}")
    print()
    print(results_df.to_string(index=False))
    print()
    print("Metrics:")
    print("  - n_anomalies: number of waiters flagged as anomaly (-1)")
    print("  - pct_flagged: percentage of all waiters flagged")
    print("  - fraud_hit_rate: fraction of known fraud waiters correctly flagged as anomaly")
    print("  - recall@k: fraction of known fraud waiters in the top k by anomaly score (k=10,20,50,100)")
    print("  - precision@k: fraction of top k by anomaly score that are known fraud waiters (k=10,20,50,100)")
    print("  - time_sec: fit+predict time in seconds")
    print()

    fraud_mask = waiter_data["is_fraud"].values
    fraud_index = np.where(fraud_mask)[0]
    print("Known fraud waiters — which model flagged them (-1 = anomaly):")
    print("-" * 60)
    for i in fraud_index:
        wid = waiter_data.index[i]
        row = (
            f"  waiter_id={wid}: "
            f"IF={predictions['iso'][i]}, OCSVM={predictions['ocsvm'][i]}, LOF={predictions['lof'][i]}"
        )
        print(row)
    print()

    if plot_scores_path:
        _client_models._plot_anomaly_score_distributions(scores, y_fraud, plot_scores_path)
        print(f"Anomaly score distributions saved to {plot_scores_path}")

    X_out = pd.DataFrame(X_eval, index=waiter_data.index, columns=waiter_features)
    return results_df, predictions, waiter_data, X_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Waiter anomaly detection: IF, OCSVM, LOF")
    parser.add_argument("--activity-state", type=int, default=2)
    parser.add_argument("--days-visits", type=int, default=2)
    parser.add_argument("--min-working-days", type=int, default=5)
    parser.add_argument("--plot", type=str, default=None, help="Path to save score distribution plot")
    args = parser.parse_args()
    compare_waiter_models(
        activity_state=args.activity_state,
        days_visits=args.days_visits,
        min_working_days=args.min_working_days,
        exclude_fraud_from_training=True,
        plot_scores_path=args.plot or os.path.join(_project_root, "waiter_anomaly_score_distributions.png"),
    )