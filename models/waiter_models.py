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

from config import load_data, FEATURES, SKEWED, FRAUD_IDS, FRAUD_WAITER_IDS

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
]
WAITER_SKEWED: list = []


def _build_waiter_data(df: pd.DataFrame, client_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build waiter-level DataFrame from transaction df and client_data (with anomaly scores).
    client_data must have anomaly_score_iso, anomaly_score_ocsvm, anomaly_score_lof, top_waiter_id, num_of_trn, days_visits.
    df must have person_id, waiter_id, place_id, date, trn_id and the same anomaly score columns after merge.
    """
    # Merge anomaly scores from client_data into df (df has person_id)
    if "anomaly_score_iso" not in df.columns and "anomaly_score_iso" in client_data.columns:
        scores = client_data[["anomaly_score_iso", "anomaly_score_ocsvm", "anomaly_score_lof"]]
        df = df.merge(scores, left_on="person_id", right_index=True, how="left")
    elif "anomaly_score_iso" not in df.columns:
        raise ValueError("Anomaly scores must be in df or client_data")

    waiter_data = df.groupby("waiter_id").agg(
        iso_90=("anomaly_score_iso", lambda x: x.quantile(0.9)),
        ocsvm_90=("anomaly_score_ocsvm", lambda x: x.quantile(0.9)),
        lof_90=("anomaly_score_lof", lambda x: x.quantile(0.9)),
        num_of_trn=("trn_id", "nunique"),
        num_of_clients=("person_id", "nunique"),
        working_days=("date", "nunique"),
    ).reset_index()

    fraud_waiter_ids = client_data[client_data["is_fraud"] == 1]["top_waiter_id"].dropna().unique()
    waiter_data["is_fraud"] = waiter_data["waiter_id"].isin(fraud_waiter_ids)

    active_person_ids = client_data[
        (client_data["num_of_trn"] > 5) & (client_data["days_visits"] > 5)
    ].index

    active_clients_per_waiter = (
        df[df["person_id"].isin(active_person_ids)]
        .groupby("waiter_id")["person_id"]
        .nunique()
        .rename("num_of_active_clients")
    )
    waiter_data = waiter_data.merge(
        active_clients_per_waiter, left_on="waiter_id", right_index=True, how="left"
    )
    waiter_data["num_of_active_clients"] = waiter_data["num_of_active_clients"].fillna(0).astype(int)

    person_place_waiter = (
        df.groupby(["person_id", "place_id"])["waiter_id"]
        .nunique()
        .reset_index(name="num_waiters_in_place")
    )
    single_waiter_person_place = person_place_waiter[
        person_place_waiter["num_waiters_in_place"] == 1
    ][["person_id", "place_id"]]
    person_place_waiter_map = df[["person_id", "place_id", "waiter_id"]].drop_duplicates()
    single_waiter_records = single_waiter_person_place.merge(
        person_place_waiter_map, on=["person_id", "place_id"], how="left"
    )

    clients_only_this_waiter = (
        single_waiter_records.groupby("waiter_id")["person_id"]
        .nunique()
        .rename("num_clients_only_this_waiter")
    )
    waiter_data = waiter_data.merge(
        clients_only_this_waiter, left_on="waiter_id", right_index=True, how="left"
    )
    waiter_data["num_clients_only_this_waiter"] = (
        waiter_data["num_clients_only_this_waiter"].fillna(0).astype(int)
    )

    single_waiter_active_records = single_waiter_records[
        single_waiter_records["person_id"].isin(active_person_ids)
    ]
    active_clients_only_this_waiter = (
        single_waiter_active_records.groupby("waiter_id")["person_id"]
        .nunique()
        .rename("num_active_clients_only_this_waiter")
    )
    waiter_data = waiter_data.merge(
        active_clients_only_this_waiter, left_on="waiter_id", right_index=True, how="left"
    )
    waiter_data["num_active_clients_only_this_waiter"] = (
        waiter_data["num_active_clients_only_this_waiter"].fillna(0).astype(int)
    )

    person_waiter = (
        df.groupby("person_id")["waiter_id"].nunique().reset_index(name="num_waiters_total")
    )
    single_waiter_total_persons = person_waiter[person_waiter["num_waiters_total"] == 1][["person_id"]]
    person_waiter_total_map = df[["person_id", "waiter_id"]].drop_duplicates()
    single_waiter_total_records = single_waiter_total_persons.merge(
        person_waiter_total_map, on="person_id", how="left"
    )

    clients_single_waiter_total = (
        single_waiter_total_records.groupby("waiter_id")["person_id"]
        .nunique()
        .rename("num_clients_single_waiter_total")
    )
    waiter_data = waiter_data.merge(
        clients_single_waiter_total, left_on="waiter_id", right_index=True, how="left"
    )
    waiter_data["num_clients_single_waiter_total"] = (
        waiter_data["num_clients_single_waiter_total"].fillna(0).astype(int)
    )

    single_waiter_total_active_records = single_waiter_total_records[
        single_waiter_total_records["person_id"].isin(active_person_ids)
    ]
    active_clients_single_waiter_total = (
        single_waiter_total_active_records.groupby("waiter_id")["person_id"]
        .nunique()
        .rename("num_active_clients_single_waiter_total")
    )
    waiter_data = waiter_data.merge(
        active_clients_single_waiter_total, left_on="waiter_id", right_index=True, how="left"
    )
    waiter_data["num_active_clients_single_waiter_total"] = (
        waiter_data["num_active_clients_single_waiter_total"].fillna(0).astype(int)
    )

    waiter_data["share_clients_only_this_waiter"] = (
        waiter_data["num_clients_only_this_waiter"] / waiter_data["num_of_clients"]
    ).fillna(0)
    waiter_data["share_active_clients_only_this_waiter"] = (
        waiter_data["num_active_clients_only_this_waiter"]
        / waiter_data["num_of_active_clients"].replace(0, np.nan)
    ).fillna(0)
    waiter_data["share_clients_single_waiter_total"] = (
        waiter_data["num_clients_single_waiter_total"] / waiter_data["num_of_clients"]
    ).fillna(0)
    waiter_data["share_active_clients_single_waiter_total"] = (
        waiter_data["num_active_clients_single_waiter_total"]
        / waiter_data["num_of_active_clients"].replace(0, np.nan)
    ).fillna(0)

    return waiter_data


def compare_waiter_models(
    activity_state: int = 2,
    days_visits: int = 2,
    min_working_days: int = 20,
    exclude_fraud_from_training: bool = True,
    plot_scores_path: Optional[str] = None,
    waiter_features: list = WAITER_FEATURES,
    waiter_skewed: list = WAITER_SKEWED,
    waiter_data: pd.DataFrame = None,
):
    """
    Load client data, compute client-level anomaly scores, build waiter-level features,
    then run Isolation Forest, One-Class SVM, and LOF on waiters. Print same metrics as models.py:
    hit_rate, recall@k, precision@k, n_anomalies, pct_flagged, time_sec.
    """
    df, client_data, _ = load_data(activity_state=activity_state, days_visits=days_visits)
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
        waiter_data = _build_waiter_data(df, client_data)
        waiter_data = waiter_data.set_index("waiter_id")
    
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
    parser.add_argument("--min-working-days", type=int, default=20)
    parser.add_argument("--plot", type=str, default=None, help="Path to save score distribution plot")
    args = parser.parse_args()
    compare_waiter_models(
        activity_state=args.activity_state,
        days_visits=args.days_visits,
        min_working_days=args.min_working_days,
        exclude_fraud_from_training=True,
        plot_scores_path=args.plot or os.path.join(_project_root, "waiter_anomaly_score_distributions.png"),
    )