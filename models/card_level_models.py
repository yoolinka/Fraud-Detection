"""
Compare Isolation Forest, One-Class SVM, and LOF on card-level features.
"""
import os
import sys
import warnings
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from config import load_data_with_graph_features

import importlib.util
_spec = importlib.util.spec_from_file_location("scaling", os.path.join(_script_dir, "scaling.py"))
_scaling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scaling)
scale_features = _scaling.scale_features

from fit_and_evaluate import fit_and_evaluate, fit_and_evaluate_per_model

# Card-level features (copied from config.FEATURES to keep this module self-contained).
CARD_LEVEL_FEATURES = [
    "share_top_waiter",
    "gross_amount_mean_prcnt",
    "num_of_trn_prcnt",
    "first_last_trn_diff_prcnt",
]
CARD_LEVEL_FEATURES_ISO = list(CARD_LEVEL_FEATURES)
CARD_LEVEL_FEATURES_OCSVM = list(CARD_LEVEL_FEATURES)
CARD_LEVEL_FEATURES_LOF = list(CARD_LEVEL_FEATURES)

CARD_LEVEL_FEATURES_BY_MODEL: Dict[str, list] = {
    "iso": list(CARD_LEVEL_FEATURES_ISO),
    "ocsvm": list(CARD_LEVEL_FEATURES_OCSVM),
    "lof": list(CARD_LEVEL_FEATURES_LOF),
}

_MODEL_KEYS: Tuple[str, ...] = ("iso", "ocsvm", "lof")


def _resolve_features_by_model(
    features_by_model: Optional[Mapping[str, Sequence[str]]],
    card_features: Optional[Sequence[str]],
) -> Dict[str, list]:
    if features_by_model is not None and card_features is not None:
        raise ValueError("Pass either features_by_model or card_features, not both.")
    if card_features is not None:
        shared = list(card_features)
        return {k: list(shared) for k in _MODEL_KEYS}
    if features_by_model is None:
        return {k: list(v) for k, v in CARD_LEVEL_FEATURES_BY_MODEL.items()}
    out = {k: list(v) for k, v in CARD_LEVEL_FEATURES_BY_MODEL.items()}
    for k, v in features_by_model.items():
        if k not in out:
            raise ValueError(f"Unknown model key {k!r}; expected one of {list(_MODEL_KEYS)}")
        out[k] = list(v)
    return out


def _scale_per_model(
    card_data: pd.DataFrame,
    features_by_model: Mapping[str, Sequence[str]],
    exclude_fraud_from_training: bool,
    y_fraud: np.ndarray,
    scaler_type: str = "standard",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, pd.DataFrame], int]:
    train_mask = y_fraud == 0
    fit_subset = card_data.loc[train_mask] if exclude_fraud_from_training else None
    n_train = int(train_mask.sum()) if exclude_fraud_from_training else len(card_data)
    X_fit: Dict[str, np.ndarray] = {}
    X_eval: Dict[str, np.ndarray] = {}
    X_out: Dict[str, pd.DataFrame] = {}

    for key in _MODEL_KEYS:
        feats = list(features_by_model[key])
        if exclude_fraud_from_training:
            X_fit_df, X_eval_df = scale_features(
                data=card_data,
                features=feats,
                scaler_type=scaler_type,
                fit_data=fit_subset,
            )
            X_fit[key] = np.asarray(X_fit_df.values, dtype=np.float64)
            X_eval[key] = np.asarray(X_eval_df.values, dtype=np.float64)
        else:
            X_full = scale_features(
                data=card_data,
                features=feats,
                scaler_type=scaler_type,
            )
            arr = np.asarray(X_full.values, dtype=np.float64)
            X_fit[key] = arr
            X_eval[key] = arr
        X_out[key] = pd.DataFrame(X_eval[key], index=card_data.index, columns=feats)
    return X_fit, X_eval, X_out, n_train


def _plot_anomaly_score_distributions(
    scores: dict,
    y_fraud: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
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
    features_by_model: Optional[Mapping[str, Sequence[str]]] = None,
    card_features: Optional[Sequence[str]] = None,
    plot_scores_path: Optional[str] = None,
    scores_csv_path: Optional[str] = None,
    n_neighbors: int = 5,
    n_estimators: int = 200,
):
    _, card_data = load_data_with_graph_features(activity_state=activity_state, days_visits=days_visits)
    y_fraud = card_data["is_fraud"].astype(int).values
    n_fraud = int(y_fraud.sum())
    n_total = len(card_data)
    resolved = _resolve_features_by_model(features_by_model, card_features)

    if compare_scalers:
        shared = list(resolved["iso"])
        if any(list(resolved[k]) != shared for k in _MODEL_KEYS):
            raise ValueError("compare_scalers=True supports only shared feature list across all models.")
        if exclude_fraud_from_training:
            train_mask = ~card_data["is_fraud"].values
            train_data = card_data.loc[train_mask]
            X_fit_std, X_eval_std = scale_features(
                data=card_data, scaler_type="standard", features=shared, fit_data=train_data,
            )
            X_fit_rob, X_eval_rob = scale_features(
                data=card_data, scaler_type="robust", features=shared, fit_data=train_data,
            )
            X_fit_std, X_eval_std = X_fit_std.values, X_eval_std.values
            X_fit_rob, X_eval_rob = X_fit_rob.values, X_eval_rob.values
            n_train = len(train_data)
        else:
            X_std = scale_features(data=card_data, scaler_type="standard", features=shared)
            X_rob = scale_features(data=card_data, scaler_type="robust", features=shared)
            X_fit_std = X_eval_std = X_std.values
            X_fit_rob = X_eval_rob = X_rob.values
            n_train = n_total

        results_std, pred_std, scores_std = fit_and_evaluate(
            X_fit_std, y_fraud, X_eval=X_eval_std, n_neighbors=n_neighbors, n_estimators=n_estimators
        )
        results_rob, pred_rob, _ = fit_and_evaluate(
            X_fit_rob, y_fraud, X_eval=X_eval_rob, n_neighbors=n_neighbors, n_estimators=n_estimators
        )
        results_std["scaler"] = "standard"
        results_rob["scaler"] = "robust"
        results_df = pd.concat([results_std, results_rob], ignore_index=True)
        cols = ["model", "scaler"] + [c for c in results_df.columns if c not in ("model", "scaler")]
        results_df = results_df[cols]
        predictions = {"standard": pred_std, "robust": pred_rob}
        scores = scores_std
        X_out = pd.DataFrame(X_eval_std, index=card_data.index, columns=shared)
    else:
        X_fit, X_eval, X_out_by_model, n_train = _scale_per_model(
            card_data=card_data,
            features_by_model=resolved,
            exclude_fraud_from_training=exclude_fraud_from_training,
            y_fraud=y_fraud,
            scaler_type="standard",
        )
        results_df, predictions, scores = fit_and_evaluate_per_model(
            X_fit, X_eval, y_fraud, n_neighbors=n_neighbors, n_estimators=n_estimators
        )
        X_out = X_out_by_model["iso"]

    print("=" * 60)
    print("Card-level anomaly detection — model comparison")
    if compare_scalers:
        print("(Standard vs Robust scaling)")
    print("=" * 60)
    print(f"Samples (total): {n_total}  |  Known frauds: {n_fraud}")
    if exclude_fraud_from_training:
        print(f"Training on non-fraud only: {n_train} samples")
    else:
        print("Training on full data (fraud included in training)")
    for key in _MODEL_KEYS:
        print(f"Features {key.upper()} ({len(resolved[key])}): {resolved[key]}")
    print()

    print(results_df.to_string(index=False))
    print()
    if plot_scores_path:
        _plot_anomaly_score_distributions(scores, y_fraud, plot_scores_path)
        print(f"Anomaly score distributions saved to {plot_scores_path}")
    if scores_csv_path:
        _write_person_anomaly_scores_csv(card_data.index, scores, scores_csv_path)
        print(f"Per-person anomaly scores saved to {scores_csv_path}")
    print()
    return results_df, predictions, card_data, X_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Card-level anomaly detection model comparison")
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
