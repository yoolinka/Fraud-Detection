"""
Fit and evaluate anomaly detection models: Isolation Forest, One-Class SVM, and LOF.
"""
import sys
import os
import time
import warnings
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

def _hit_rate(pred: np.ndarray, fraud_binary: np.ndarray) -> float:
    """Fraction of known frauds that were flagged as anomaly (-1)."""
    if fraud_binary.sum() == 0:
        return np.nan
    fraud_mask = fraud_binary.astype(bool)
    return (pred[fraud_mask] == -1).mean()


def _top_k_recall(
    anomaly_scores: np.ndarray,
    fraud_binary: np.ndarray,
    k_list: list,
) -> dict:
    """
    anomaly_scores: higher = more anomalous.
    Returns dict of recall@k = (frauds in top k by score) / total_frauds.
    """
    n_fraud = int(fraud_binary.sum())
    if n_fraud == 0:
        return {f"recall@{k}": np.nan for k in k_list}
    order = np.argsort(-anomaly_scores)
    fraud_mask = fraud_binary.astype(bool)
    out = {}
    for k in k_list:
        top_k = order[: min(k, len(order))]
        n_fraud_in_top_k = fraud_mask[top_k].sum()
        out[f"recall@{k}"] = n_fraud_in_top_k / n_fraud
    return out


def _top_k_precision(
    anomaly_scores: np.ndarray,
    fraud_binary: np.ndarray,
    k_list: list,
) -> dict:
    """
    anomaly_scores: higher = more anomalous.
    Returns dict of precision@k = (frauds in top k by score) / k.
    """
    order = np.argsort(-anomaly_scores)
    fraud_mask = fraud_binary.astype(bool)
    out = {}
    for k in k_list:
        top_k = order[: min(k, len(order))]
        n_fraud_in_top_k = fraud_mask[top_k].sum()
        out[f"precision@{k}"] = n_fraud_in_top_k / k
    return out

def fit_and_evaluate(
    X_fit: np.ndarray,
    y_true_binary: np.ndarray,
    X_eval: Optional[np.ndarray] = None,
    max_ocsvm_train: int = 4000,
    n_neighbors: int = 5,
    n_estimators: int = 100
):
    """
    Fit each model on X_fit, predict on X_eval (or X_fit if X_eval is None).
    y_true_binary: 1 for known fraud, 0 otherwise; length must match X_eval (or X_fit).
    max_ocsvm_train: One-Class SVM is slow; if n_fit > this, fit OCSVM on a subsample (default 4000).
    """
    X_eval = X_eval if X_eval is not None else X_fit
    n_fit, n_eval = len(X_fit), len(X_eval)
    # n_fraud = y_true_binary.sum()
    top_k_list = [10, 20, 50, 100]
    results = []

    # --- Isolation Forest ---
    t0 = time.perf_counter()
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=0.0005,
        max_samples=1.0,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_fit)
    pred_iso = iso.predict(X_eval)
    # score_samples: lower = more anomalous; use negative so higher = more anomalous
    scores_iso = -iso.score_samples(X_eval)
    t_iso = time.perf_counter() - t0
    n_anom_iso = (pred_iso == -1).sum()
    hit_iso = _hit_rate(pred_iso, y_true_binary)
    row_iso = {
        "model": "Isolation Forest",
        "n_anomalies": n_anom_iso,
        "pct_flagged": 100 * n_anom_iso / n_eval,
        "fraud_hit_rate": hit_iso,
        "time_sec": round(t_iso, 3),
    }
    row_iso.update(_top_k_recall(scores_iso, y_true_binary, top_k_list))
    row_iso.update(_top_k_precision(scores_iso, y_true_binary, top_k_list))
    results.append(row_iso)

    # --- One-Class SVM (subsample if large: RBF fit is O(n²)) ---
    nu = 0.0005
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    if n_fit > max_ocsvm_train:
        idx = rng.choice(n_fit, size=max_ocsvm_train, replace=False)
        X_fit_ocsvm = X_fit[idx]
    else:
        X_fit_ocsvm = X_fit
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    ocsvm.fit(X_fit_ocsvm)
    pred_ocsvm = ocsvm.predict(X_eval)
    # decision_function: negative = anomaly; negate so higher = more anomalous
    scores_ocsvm = -ocsvm.decision_function(X_eval)
    t_ocsvm = time.perf_counter() - t0
    n_anom_ocsvm = (pred_ocsvm == -1).sum()
    hit_ocsvm = _hit_rate(pred_ocsvm, y_true_binary)
    row_ocsvm = {
        "model": "One-Class SVM",
        "n_anomalies": n_anom_ocsvm,
        "pct_flagged": 100 * n_anom_ocsvm / n_eval,
        "fraud_hit_rate": hit_ocsvm,
        "time_sec": round(t_ocsvm, 3),
    }
    row_ocsvm.update(_top_k_recall(scores_ocsvm, y_true_binary, top_k_list))
    row_ocsvm.update(_top_k_precision(scores_ocsvm, y_true_binary, top_k_list))
    results.append(row_ocsvm)

    # --- Local Outlier Factor (novelty=True so we can predict on X_eval) ---
    # n_neighbors = min(50, n_fit - 1)
    # if n_neighbors < 5:
    t0 = time.perf_counter()
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=0.01,
        metric="minkowski",
        p=2,
        novelty=True,
    )
    lof.fit(X_fit)
    pred_lof = lof.predict(X_eval)
    # score_samples: more negative = more anomalous; negate so higher = more anomalous
    scores_lof = -lof.score_samples(X_eval)
    t_lof = time.perf_counter() - t0
    n_anom_lof = (pred_lof == -1).sum()
    hit_lof = _hit_rate(pred_lof, y_true_binary)
    row_lof = {
        "model": "LOF",
        "n_anomalies": n_anom_lof,
        "pct_flagged": 100 * n_anom_lof / n_eval,
        "fraud_hit_rate": hit_lof,
        "time_sec": round(t_lof, 3),
    }
    row_lof.update(_top_k_recall(scores_lof, y_true_binary, top_k_list))
    row_lof.update(_top_k_precision(scores_lof, y_true_binary, top_k_list))
    results.append(row_lof)

    scores = {"iso": scores_iso, "ocsvm": scores_ocsvm, "lof": scores_lof}
    return pd.DataFrame(results), {"iso": pred_iso, "ocsvm": pred_ocsvm, "lof": pred_lof}, scores


_MODEL_KEYS = ("iso", "ocsvm", "lof")


def fit_and_evaluate_per_model(
    X_fit: Mapping[str, np.ndarray],
    X_eval: Mapping[str, np.ndarray],
    y_true_binary: np.ndarray,
    max_ocsvm_train: int = 4000,
    n_neighbors: int = 5,
    n_estimators: int = 100,
):
    """
    Same as ``fit_and_evaluate`` but each model gets its own feature matrix.
    ``X_fit`` / ``X_eval`` must contain keys 'iso', 'ocsvm', 'lof'; row order must
    match ``y_true_binary`` for every key.
    """
    y_true_binary = np.asarray(y_true_binary)
    n_eval = len(y_true_binary)
    for key in _MODEL_KEYS:
        if key not in X_fit or key not in X_eval:
            raise ValueError(f"X_fit and X_eval must contain key {key!r}")
        if len(X_eval[key]) != n_eval:
            raise ValueError(
                f"X_eval[{key!r}] has {len(X_eval[key])} rows; y_true_binary has {n_eval}"
            )
    top_k_list = [10, 20, 50, 100]
    results = []

    # --- Isolation Forest ---
    X_fit_iso = np.asarray(X_fit["iso"], dtype=np.float64)
    X_eval_iso = np.asarray(X_eval["iso"], dtype=np.float64)
    t0 = time.perf_counter()
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=0.0005,
        max_samples=1.0,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_fit_iso)
    pred_iso = iso.predict(X_eval_iso)
    scores_iso = -iso.score_samples(X_eval_iso)
    t_iso = time.perf_counter() - t0
    n_anom_iso = (pred_iso == -1).sum()
    hit_iso = _hit_rate(pred_iso, y_true_binary)
    row_iso = {
        "model": "Isolation Forest",
        "n_anomalies": n_anom_iso,
        "pct_flagged": 100 * n_anom_iso / n_eval,
        "fraud_hit_rate": hit_iso,
        "time_sec": round(t_iso, 3),
    }
    row_iso.update(_top_k_recall(scores_iso, y_true_binary, top_k_list))
    row_iso.update(_top_k_precision(scores_iso, y_true_binary, top_k_list))
    results.append(row_iso)

    # --- One-Class SVM ---
    X_fit_ocsvm = np.asarray(X_fit["ocsvm"], dtype=np.float64)
    X_eval_ocsvm = np.asarray(X_eval["ocsvm"], dtype=np.float64)
    n_fit_ocsvm = len(X_fit_ocsvm)
    nu = 0.0005
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    if n_fit_ocsvm > max_ocsvm_train:
        idx = rng.choice(n_fit_ocsvm, size=max_ocsvm_train, replace=False)
        X_fit_ocsvm_sub = X_fit_ocsvm[idx]
    else:
        X_fit_ocsvm_sub = X_fit_ocsvm
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    ocsvm.fit(X_fit_ocsvm_sub)
    pred_ocsvm = ocsvm.predict(X_eval_ocsvm)
    scores_ocsvm = -ocsvm.decision_function(X_eval_ocsvm)
    t_ocsvm = time.perf_counter() - t0
    n_anom_ocsvm = (pred_ocsvm == -1).sum()
    hit_ocsvm = _hit_rate(pred_ocsvm, y_true_binary)
    row_ocsvm = {
        "model": "One-Class SVM",
        "n_anomalies": n_anom_ocsvm,
        "pct_flagged": 100 * n_anom_ocsvm / n_eval,
        "fraud_hit_rate": hit_ocsvm,
        "time_sec": round(t_ocsvm, 3),
    }
    row_ocsvm.update(_top_k_recall(scores_ocsvm, y_true_binary, top_k_list))
    row_ocsvm.update(_top_k_precision(scores_ocsvm, y_true_binary, top_k_list))
    results.append(row_ocsvm)

    # --- LOF ---
    X_fit_lof = np.asarray(X_fit["lof"], dtype=np.float64)
    X_eval_lof = np.asarray(X_eval["lof"], dtype=np.float64)
    t0 = time.perf_counter()
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=0.01,
        metric="minkowski",
        p=2,
        novelty=True,
    )
    lof.fit(X_fit_lof)
    pred_lof = lof.predict(X_eval_lof)
    scores_lof = -lof.score_samples(X_eval_lof)
    t_lof = time.perf_counter() - t0
    n_anom_lof = (pred_lof == -1).sum()
    hit_lof = _hit_rate(pred_lof, y_true_binary)
    row_lof = {
        "model": "LOF",
        "n_anomalies": n_anom_lof,
        "pct_flagged": 100 * n_anom_lof / n_eval,
        "fraud_hit_rate": hit_lof,
        "time_sec": round(t_lof, 3),
    }
    row_lof.update(_top_k_recall(scores_lof, y_true_binary, top_k_list))
    row_lof.update(_top_k_precision(scores_lof, y_true_binary, top_k_list))
    results.append(row_lof)

    scores = {"iso": scores_iso, "ocsvm": scores_ocsvm, "lof": scores_lof}
    return pd.DataFrame(results), {"iso": pred_iso, "ocsvm": pred_ocsvm, "lof": pred_lof}, scores
