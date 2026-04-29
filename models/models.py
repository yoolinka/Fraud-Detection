"""
Compare three anomaly detection models: Isolation Forest, One-Class SVM, and LOF.
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
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from config import FEATURES, load_data_with_graph_features

import importlib.util
_spec = importlib.util.spec_from_file_location("scaling", os.path.join(_script_dir, "scaling.py"))
_scaling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scaling)
scale_features = _scaling.scale_features

from fit_and_evaluate import fit_and_evaluate

def _plot_anomaly_score_distributions(
    scores: dict,
    y_fraud: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot histograms of anomaly scores per model (fraud vs non-fraud).
    Y-axis normalized to 0-1 (heights scaled so max = 1).
    scores: dict with keys 'iso', 'ocsvm', 'lof' and 1d arrays (higher = more anomalous).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fraud_mask = y_fraud.astype(bool)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = {"iso": "Isolation Forest", "ocsvm": "One-Class SVM", "lof": "LOF"}
    for ax, (key, name) in zip(axes, names.items()):
        s = scores[key]
        bins = np.linspace(s.min(), s.max(), 51)
        c0, _ = np.histogram(s[~fraud_mask], bins=bins)
        c1, _ = np.histogram(s[fraud_mask], bins=bins)
        # Normalize each series by its own max so fraud (few samples) is visible alongside non-fraud
        scale0 = max(c0.max(), 1)
        scale1 = max(c1.max(), 1)
        w0 = c0 / scale0
        w1 = c1 / scale1
        x = (bins[:-1] + bins[1:]) / 2
        w = bins[1] - bins[0]
        ax.bar(x, w0, width=w * 0.9, alpha=0.6, label="Non-fraud", color="C0", align="center")
        ax.bar(x, w1, width=w * 0.9, alpha=0.6, label="Fraud", color="C1", align="center", bottom=0)
        ax.set_title(name)
        ax.set_xlabel("Anomaly score (higher = more anomalous)")
        ax.set_ylabel("Relative freq. (0–1 per group)")
        ax.legend()
        ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _write_person_anomaly_scores_csv(
    person_index: pd.Index,
    scores: dict,
    path: str,
) -> None:
    """One row per person_id with IF / OCSVM / LOF scores (higher = more anomalous)."""
    out = pd.DataFrame(
        {
            "person_id": person_index,
            "iso_score": np.asarray(scores["iso"], dtype=np.float64),
            "ocsvm_score": np.asarray(scores["ocsvm"], dtype=np.float64),
            "lof_score": np.asarray(scores["lof"], dtype=np.float64),
        }
    )
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    out.to_csv(path, index=False)


def compare_models(
    activity_state: int = 1,
    days_visits: int = 2,
    exclude_fraud_from_training: bool = True,
    compare_scalers: bool = False,
    plot_scores_path: Optional[str] = None,
    scores_csv_path: Optional[str] = None,
    n_neighbors: int = 5,
    n_estimators: int = 200,
):
    """Load data, scale, run all three models, and print comparison.

    If exclude_fraud_from_training=True, models are fit only on non-fraud samples
    and evaluated on the full set (fairer evaluation of anomaly detection).
    If compare_scalers=True, runs with both StandardScaler and RobustScaler and
    reports metrics for each (6 rows: 3 models x 2 scalers).
    If plot_scores_path is set, saves anomaly score distribution histograms (fraud vs non-fraud) to that file.
    If scores_csv_path is set, writes person_id and the three anomaly scores to CSV (standard scaler when
    compare_scalers=True, same scores as the distribution plot).
    """
    # _, client_data, _, _, _ = load_data(activity_state=activity_state, days_visits=days_visits)
    _, client_data = load_data_with_graph_features(activity_state=activity_state, days_visits=days_visits)
    y_fraud = client_data["is_fraud"].astype(int).values
    n_fraud = y_fraud.sum()
    n_total = len(client_data)

    if compare_scalers:
        # Run with both standard and robust scaling
        if exclude_fraud_from_training:
            train_mask = ~client_data["is_fraud"].values
            train_data = client_data.loc[train_mask]
            X_fit_std, X_eval_std = scale_features(
                data=client_data, scaler_type="standard", features=FEATURES, fit_data=train_data,
            )
            X_fit_rob, X_eval_rob = scale_features(
                data=client_data, scaler_type="robust", features=FEATURES, fit_data=train_data,
            )
            X_fit_std, X_eval_std = X_fit_std.values, X_eval_std.values
            X_fit_rob, X_eval_rob = X_fit_rob.values, X_eval_rob.values
            n_train = len(train_data)
        else:
            X_std = scale_features(
                data=client_data, scaler_type="standard", features=FEATURES,
            )
            X_rob = scale_features(
                data=client_data, scaler_type="robust", features=FEATURES,
            )
            X_fit_std = X_eval_std = X_std.values
            X_fit_rob = X_eval_rob = X_rob.values
            n_train = n_total

        results_std, pred_std, scores_std = fit_and_evaluate(
            X_fit_std,
            y_fraud,
            X_eval=X_eval_std,
            n_neighbors=n_neighbors,
            n_estimators=n_estimators,
        )
        results_rob, pred_rob, _ = fit_and_evaluate(
            X_fit_rob,
            y_fraud,
            X_eval=X_eval_rob,
            n_neighbors=n_neighbors,
            n_estimators=n_estimators,
        )
        results_std["scaler"] = "standard"
        results_rob["scaler"] = "robust"
        results_df = pd.concat([results_std, results_rob], ignore_index=True)
        # Reorder: model, scaler, then rest
        cols = ["model", "scaler"] + [c for c in results_df.columns if c not in ("model", "scaler")]
        results_df = results_df[cols]
        predictions = {"standard": pred_std, "robust": pred_rob}
        X_eval = X_eval_std  # for return
    else:
        # Single scaling (standard)
        if exclude_fraud_from_training:
            train_mask = ~client_data["is_fraud"].values
            train_data = client_data.loc[train_mask]
            X_fit, X_eval = scale_features(
                data=client_data,
                scaler_type="standard",
                features=FEATURES,
                fit_data=train_data,
            )
            X_fit, X_eval = X_fit.values, X_eval.values
            n_train = len(train_data)
        else:
            X = scale_features(
                data=client_data,
                scaler_type="standard",
                features=FEATURES,
            )
            X_fit = X_eval = X.values
            n_train = n_total

        results_df, pred_single, scores = fit_and_evaluate(
            X_fit, y_fraud, X_eval=X_eval,
            n_neighbors=n_neighbors,
            n_estimators=n_estimators,
        )
        predictions = {"standard": pred_single}

    print("=" * 60)
    print("Anomaly detection model comparison")
    if compare_scalers:
        print("(Standard vs Robust scaling)")
    print("=" * 60)
    print(f"Samples (total): {n_total}  |  Known frauds: {n_fraud}")
    if exclude_fraud_from_training:
        print(f"Training on non-fraud only: {n_train} samples")
    else:
        print("Training on full data (fraud included in training)")
    print(f"Features: {len(FEATURES)}")
    print()

    print(results_df.to_string(index=False))
    print()
    if plot_scores_path:
        _scores = scores_std if compare_scalers else scores
        _plot_anomaly_score_distributions(_scores, y_fraud, plot_scores_path)
        print(f"Anomaly score distributions saved to {plot_scores_path}")
    if scores_csv_path:
        _scores_csv = scores_std if compare_scalers else scores
        _write_person_anomaly_scores_csv(client_data.index, _scores_csv, scores_csv_path)
        print(f"Per-person anomaly scores saved to {scores_csv_path}")
    print()
    print("Metrics:")
    print("  - n_anomalies: number of points flagged as anomaly (-1)")
    print("  - pct_flagged: percentage of all samples flagged")
    print("  - fraud_hit_rate: fraction of known frauds correctly flagged as anomaly")
    print("  - recall@k: fraction of known frauds in the top k by anomaly score (k=10,20,50,100)")
    print("  - precision@k: fraction of top k by anomaly score that are known frauds (k=10,20,50,100)")
    print("  - time_sec: fit+predict time in seconds")
    if compare_scalers:
        print("  - scaler: StandardScaler vs RobustScaler (median/IQR-based)")
    print()
    X_out = pd.DataFrame(X_eval, index=client_data.index, columns=FEATURES)
    return results_df, predictions, client_data, X_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Anomaly detection model comparison")
    parser.add_argument("--activity-state", type=int, default=1, help="Filter clients with num_of_trn > this")
    parser.add_argument(
        "--scores-csv",
        type=str,
        default=None,
        help="Path for CSV of person_id + iso/ocsvm/lof scores (default: client_anomaly_scores.csv)",
    )
    args = parser.parse_args()
    default_plot = os.path.join(_project_root, "anomaly_score_distributions.png")
    default_scores_csv = os.path.join(_project_root, "client_anomaly_scores.csv")
    scores_csv = args.scores_csv if args.scores_csv is not None else default_scores_csv
    compare_models(
        activity_state=args.activity_state,
        exclude_fraud_from_training=True,
        compare_scalers=False,
        plot_scores_path=default_plot,
        scores_csv_path=scores_csv,
    )
