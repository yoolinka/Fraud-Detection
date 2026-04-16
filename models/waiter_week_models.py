"""
Compare Isolation Forest, One-Class SVM, and LOF on waiter–week level features
(waiter_week_features.parquet)
"""
import os
import sys
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import load_data
from parquet.fraud_ids import FRAUD_WAITER_IDS

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

WAITER_WEEK_FEATURES_ISO = [
    # ISO precision@100 = 0.10, precision@50 = 0.18
    'trn_per_person_norm',
    'top1_client_trn',
    'bonusses_accum',
    'trn_per_person',
    'top1_client_share_norm',
    'top1_client_trn_diff_next',
    'top1_client_trn_diff_prev',
    'mean_check',
    'bonusses_used',
    'top1_client_trn_perc_diff_next'
]
WAITER_WEEK_FEATURES_OCSVM = [
    # OCSVM precision@100 = 0.09, precision@50 = 0.14
    'trn_per_person_norm',
    'trn_per_person_perc_diff_next',
    'top1_client_share_norm',
    'bonusses_accum_diff_prev',
    'share_loyal_trn'
]
WAITER_WEEK_FEATURES_LOF = [
   # LOF precision@100 = 0.12
    'trn_per_day_norm',
    'bonusses_accum_diff_next',
    'trn_per_person_perc_diff_next',
    'trn_per_person_norm_perc_diff_next',
    'trn_per_person',
    'unique_persons_diff_next',
    'trn_per_person_norm',
    'bonusses_accum',
    'mean_check',
    'top1_client_share_norm',
    'unique_clients_per_day',
    'unique_clients_per_day_diff_prev',
    'top1_client_trn_diff_next',
    'bonusses_trn',
    'unique_clients_per_day_perc_diff_prev',
    'trn_per_person_norm_perc_diff_prev',
    'share_bonusses_trn',
    'top1_client_trn_diff_prev',
    'share_loyal_trn'

    # LOF precision@100 = 0.11, precision@50 = 0.16
    # 'trn_per_person_norm',
    # 'top1_client_trn',
    # 'top1_client_share',
    # 'share_loyal_trn',
    # 'bonusses_accum',
    # 'trn_per_person',
    # 'top1_client_share_norm',
    # 'top1_client_trn_diff_next',
    # 'top1_client_trn_diff_prev',
    # 'trn_count_nonloyal_diff_prev',
    # 'mean_check',
    # 'bonusses_accum_diff_next',
    # 'bonusses_used',
    # 'trn_per_person_diff_prev',
    # 'trn_count_nonloyal',
    # 'bonusses_accum_diff_prev',
    # 'bonusses_trn',
    # 'trn_per_person_norm_perc_diff_next',
    # 'share_loyal_trn_perc_diff_next',
    # 'trn_per_person_perc_diff_prev',
    # 'unique_clients_per_day_diff_next',
    # 'share_new_clients_norm_diff_next',
    # 'share_of_trn_diff_next',
    # 'unique_clients_per_day',
    # 'share_new_clients',
    # 'trn_per_person_norm_diff_next',
    # 'bonusses_used_norm_l'
]
WAITER_WEEK_FEATURES = [

]
def _waiter_id_week_for_csv(waiter_week_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve waiter_id and week columns, or derive week from index + waiter_id (see parquet pipeline)."""
    idx = waiter_week_data.index
    n = len(waiter_week_data)
    if "waiter_id" in waiter_week_data.columns:
        waiter_id = waiter_week_data["waiter_id"].to_numpy()
        if "week" in waiter_week_data.columns:
            week = waiter_week_data["week"].to_numpy()
        else:
            week = np.empty(n, dtype=object)
            for i in range(n):
                ww = str(idx[i])
                p = str(waiter_id[i]) + "_"
                week[i] = ww[len(p) :] if ww.startswith(p) else np.nan
    else:
        waiter_id = np.empty(n, dtype=object)
        week = np.empty(n, dtype=object)
        for i in range(n):
            ww = str(idx[i])
            parts = ww.rsplit("_", 1)
            if len(parts) == 2:
                waiter_id[i], week[i] = parts[0], parts[1]
            else:
                waiter_id[i], week[i] = ww, np.nan
    return waiter_id, week


def _write_waiter_week_scores_csv(
    waiter_week_data: pd.DataFrame,
    scores: dict,
    path: str,
) -> None:
    """Write one row per waiter_week with IF / OCSVM / LOF anomaly scores (higher = more anomalous)."""
    waiter_id, week = _waiter_id_week_for_csv(waiter_week_data)
    out = pd.DataFrame(
        {
            "waiter_week": waiter_week_data.index,
            "waiter_id": waiter_id,
            "week": week,
            "iso_score": np.asarray(scores["iso"], dtype=np.float64),
            "ocsvm_score": np.asarray(scores["ocsvm"], dtype=np.float64),
            "lof_score": np.asarray(scores["lof"], dtype=np.float64),
        }
    )
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    out.to_csv(path, index=False)


def compare_waiter_week_models(
    waiter_features: Sequence[str] = WAITER_WEEK_FEATURES,
    agg_data: Optional[pd.DataFrame] = None,
    fraud_waiter_week_ids: Optional[Union[Sequence, np.ndarray]] = None,
    min_num_of_trn: int = 8,
    exclude_fraud_from_training: bool = True,
    num_of_trn: int = 1,
    place_num_of_waiters: int = 1,
    plot_scores_path: Optional[str] = None,
    scores_csv_path: Optional[str] = None,
):
    """
    Load or accept waiter–week aggregates, scale, fit IF / OCSVM / LOF, print metrics.

    Parameters
    ----------
    agg_data : optional pre-built DataFrame (e.g. filtered). Must include waiter_week
    column or index, feature columns, and is_fraud unless fraud_waiter_week_ids given.
    fraud_waiter_week_ids : optional index keys for known fraud rows; if None, uses is_fraud.
    min_num_of_trn : keep rows with num_of_trn >= this (after load), default 8 as in notebook.
    scores_csv_path : if set, write waiter_week × model anomaly scores to this CSV path.

    Returns
    -------
    results_df, predictions, waiter_week_data, X_out, scores
    """

    if agg_data is None:
        _, _, waiter_week_data, _ = load_data(
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
    print(f"Samples (total): {n_total}  |  Known fraud waiter-weeks: {n_fraud}  | Known fraud waiter: {len(FRAUD_WAITER_IDS)}")
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

    # fraud_index = np.where(y_fraud.astype(bool))[0]
    # print("Known fraud waiter-weeks — which model flagged them (-1 = anomaly):")
    # print("-" * 60)
    # for i in fraud_index:
    #     wid = waiter_week_data.index[i]
    #     row = (
    #         f"  waiter_week={wid}: "
    #         f"IF={predictions['iso'][i]}, OCSVM={predictions['ocsvm'][i]}, LOF={predictions['lof'][i]}"
    #     )
    #     print(row)
    # print()

    if plot_scores_path:
        _client_models._plot_anomaly_score_distributions(scores, y_fraud, plot_scores_path)
        print(f"Anomaly score distributions saved to {plot_scores_path}")

    if scores_csv_path:
        _write_waiter_week_scores_csv(waiter_week_data, scores, scores_csv_path)
        print(f"Per–waiter-week anomaly scores saved to {scores_csv_path}")

    X_out = pd.DataFrame(X_eval, index=waiter_week_data.index, columns=waiter_features)
    return results_df, predictions, waiter_week_data, X_out, scores


def compare_waiter_week_real_vs_synthetic(
    waiter_features: Sequence[str] = WAITER_WEEK_FEATURES,
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
        _, _, waiter_week_data, _ = load_data(
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
    parser.add_argument(
        "--scores-csv",
        type=str,
        default=None,
        help="Path for CSV of waiter_week + iso/ocsvm/lof scores (default: waiter_week_anomaly_scores.csv)",
    )
    parser.add_argument("--place-num-of-waiters", type=int, default=2, help="Place num of waiters lower bound (inclusive)")
    args = parser.parse_args()
    default_plot = os.path.join(_project_root, "waiter_week_anomaly_score_distributions.png")
    default_scores_csv = os.path.join(_project_root, "waiter_week_anomaly_scores.csv")
    scores_csv = args.scores_csv if args.scores_csv is not None else default_scores_csv
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
            scores_csv_path=scores_csv,
            place_num_of_waiters=args.place_num_of_waiters,
        )
