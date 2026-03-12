"""
Compare three anomaly detection models: Isolation Forest, One-Class SVM, and LOF.
"""
import sys
import os
import time
import warnings
from typing import Optional

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

from config import load_data, FEATURES, SKEWED, FRAUD_IDS, DATA_PATH

import importlib.util
_spec = importlib.util.spec_from_file_location("scaling", os.path.join(_script_dir, "scaling.py"))
_scaling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scaling)
scale_features = _scaling.scale_features


def generate_synthetic_data(
    client_data: pd.DataFrame,
    n_synthetic: int = 500,
    noise_scale: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic fraud samples from real fraud: resample with replacement,
    add Gaussian noise to numeric features, assign new person_id indices.
    Returns a DataFrame with same structure as client_data: non-fraud rows +
    synthetic fraud (is_fraud=1), index = person_id.
    """
    fraud_real = client_data[client_data["is_fraud"] == 1]
    if len(fraud_real) == 0:
        raise ValueError("No fraud rows in client_data; cannot generate synthetic fraud.")
    synthetic = fraud_real.sample(n=min(n_synthetic, len(fraud_real)), replace=True, random_state=random_state).copy()
    num_cols = synthetic.select_dtypes("number").columns.difference(["is_fraud"], sort=False)
    if "person_id" in synthetic.columns:
        num_cols = num_cols.difference(["person_id"], sort=False)
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, noise_scale, (len(synthetic), len(num_cols)))
    synthetic.loc[:, num_cols] = synthetic[num_cols].values * (1 + noise)
    start = int(client_data.index.max()) + 1
    synthetic.index = np.arange(start, start + len(synthetic))
    synthetic.index.name = client_data.index.name
    synthetic["is_fraud"] = 1
    non_fraud = client_data[client_data["is_fraud"] == 0]
    non_fraud['synthetic'] = 0
    synthetic['synthetic'] = 1
    return pd.concat([non_fraud, synthetic], axis=0)


def fit_and_evaluate(
    X_fit: np.ndarray,
    y_true_binary: np.ndarray,
    fraud_ids: list,
    index,
    X_eval: Optional[np.ndarray] = None,
    max_ocsvm_train: int = 4000,
):
    """
    Fit each model on X_fit, predict on X_eval (or X_fit if X_eval is None).
    y_true_binary: 1 for known fraud, 0 otherwise; length must match X_eval (or X_fit).
    max_ocsvm_train: One-Class SVM is slow; if n_fit > this, fit OCSVM on a subsample (default 4000).
    """
    X_eval = X_eval if X_eval is not None else X_fit
    n_fit, n_eval = len(X_fit), len(X_eval)
    n_fraud = y_true_binary.sum()
    top_k_list = [10, 20, 50, 100]
    results = []

    # --- Isolation Forest ---
    contamination = 0.001
    t0 = time.perf_counter()
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
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
    nu = 0.001
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
    n_neighbors = min(50, n_fit - 1)
    if n_neighbors < 5:
        n_neighbors = 5
    t0 = time.perf_counter()
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
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


def compare_models(
    activity_state: int = 1,
    exclude_fraud_from_training: bool = True,
    compare_scalers: bool = False,
    plot_scores_path: Optional[str] = None,
):
    """Load data, scale, run all three models, and print comparison.

    If exclude_fraud_from_training=True, models are fit only on non-fraud samples
    and evaluated on the full set (fairer evaluation of anomaly detection).
    If compare_scalers=True, runs with both StandardScaler and RobustScaler and
    reports metrics for each (6 rows: 3 models x 2 scalers).
    If plot_scores_path is set, saves anomaly score distribution histograms (fraud vs non-fraud) to that file.
    """
    df, client_data = load_data(activity_state=activity_state)
    y_fraud = client_data["is_fraud"].astype(int).values
    n_fraud = y_fraud.sum()
    n_total = len(client_data)

    if compare_scalers:
        # Run with both standard and robust scaling
        if exclude_fraud_from_training:
            train_mask = ~client_data["is_fraud"].values
            train_data = client_data.loc[train_mask]
            X_fit_std, X_eval_std = scale_features(
                data=client_data, scaler_type="standard", features=FEATURES, skewed=SKEWED, fit_data=train_data,
            )
            X_fit_rob, X_eval_rob = scale_features(
                data=client_data, scaler_type="robust", features=FEATURES, skewed=SKEWED, fit_data=train_data,
            )
            X_fit_std, X_eval_std = X_fit_std.values, X_eval_std.values
            X_fit_rob, X_eval_rob = X_fit_rob.values, X_eval_rob.values
            n_train = len(train_data)
        else:
            X_std = scale_features(
                data=client_data, scaler_type="standard", features=FEATURES, skewed=SKEWED,
            )
            X_rob = scale_features(
                data=client_data, scaler_type="robust", features=FEATURES, skewed=SKEWED,
            )
            X_fit_std = X_eval_std = X_std.values
            X_fit_rob = X_eval_rob = X_rob.values
            n_train = n_total

        results_std, pred_std, scores_std = fit_and_evaluate(
            X_fit_std, y_fraud, FRAUD_IDS, client_data.index, X_eval=X_eval_std,
        )
        results_rob, pred_rob, scores_rob = fit_and_evaluate(
            X_fit_rob, y_fraud, FRAUD_IDS, client_data.index, X_eval=X_eval_rob,
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
                skewed=SKEWED,
                fit_data=train_data,
            )
            X_fit, X_eval = X_fit.values, X_eval.values
            n_train = len(train_data)
        else:
            X = scale_features(
                data=client_data,
                scaler_type="standard",
                features=FEATURES,
                skewed=SKEWED,
            )
            X_fit = X_eval = X.values
            n_train = n_total

        results_df, pred_single, scores = fit_and_evaluate(
            X_fit, y_fraud, FRAUD_IDS, client_data.index, X_eval=X_eval,
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

    fraud_mask = client_data["is_fraud"].values
    fraud_index = np.where(fraud_mask)[0]
    print("Known frauds — which model flagged them (-1 = anomaly):")
    print("-" * 60)
    if compare_scalers:
        for i in fraud_index:
            pid = client_data.index[i]
            row = (
                f"  person_id={pid}: "
                f"IF_std={pred_std['iso'][i]} IF_rob={pred_rob['iso'][i]} | "
                f"OCSVM_std={pred_std['ocsvm'][i]} OCSVM_rob={pred_rob['ocsvm'][i]} | "
                f"LOF_std={pred_std['lof'][i]} LOF_rob={pred_rob['lof'][i]}"
            )
            print(row)
    else:
        pred = predictions["standard"]
        for i in fraud_index:
            pid = client_data.index[i]
            row = (
                f"  person_id={pid}: "
                f"IF={pred['iso'][i]}, OCSVM={pred['ocsvm'][i]}, LOF={pred['lof'][i]}"
            )
            print(row)
    print()
    X_out = pd.DataFrame(X_eval, index=client_data.index, columns=FEATURES)
    return results_df, predictions, client_data, X_out


def compare_real_vs_synthetic(
    activity_state: int = 1,
    n_synthetic: int = 500,
    noise_scale: float = 0.1,
    random_state: int = 42,
    plot_scores_path: Optional[str] = None,
) -> tuple:
    """
    Load real data, generate synthetic data (non-fraud + synthetic fraud),
    run the same three models on both with training on non-fraud only.
    Returns (results_real, results_synthetic, client_data_real, client_data_synthetic).
    """
    df, client_data = load_data(activity_state=activity_state)
    client_data = client_data[client_data["num_of_trn"] > activity_state]
    if "is_fraud" not in client_data.columns:
        client_data["is_fraud"] = client_data.index.isin(FRAUD_IDS).astype(int)
    y_real = client_data["is_fraud"].astype(int).values
    n_fraud_real = y_real.sum()

    # --- Real: fit on non-fraud, evaluate on full ---
    train_mask = ~client_data["is_fraud"].astype(bool).values
    train_data = client_data.loc[train_mask]
    X_fit_real, X_eval_real = scale_features(
        data=client_data,
        scaler_type="standard",
        features=FEATURES,
        skewed=SKEWED,
        fit_data=train_data,
    )
    results_real, _, scores_real = fit_and_evaluate(
        X_fit_real.values,
        y_real,
        FRAUD_IDS,
        client_data.index,
        X_eval=X_eval_real.values,
    )
    results_real.insert(0, "dataset", "Real")

    # --- Synthetic: generate then fit on non-fraud synthetic, evaluate on full synthetic ---
    synthetic = generate_synthetic_data(
        client_data, n_synthetic=n_synthetic, noise_scale=noise_scale, random_state=random_state
    )
    synthetic = synthetic[synthetic["num_of_trn"] > activity_state]
    y_synt = synthetic["is_fraud"].astype(int).values
    n_fraud_synt = y_synt.sum()
    train_synt = synthetic[synthetic["is_fraud"] == 0]
    X_fit_synt, X_eval_synt = scale_features(
        data=synthetic,
        scaler_type="standard",
        features=FEATURES,
        skewed=SKEWED,
        fit_data=train_synt,
    )
    results_synt, _, scores_synt = fit_and_evaluate(
        X_fit_synt.values,
        y_synt,
        [],  # no named fraud IDs for synthetic
        synthetic.index,
        X_eval=X_eval_synt.values,
    )
    results_synt.insert(0, "dataset", "Synthetic")

    # --- Print comparison ---
    print("=" * 70)
    print("Real vs Synthetic data — model comparison (train on non-fraud only)")
    print("=" * 70)
    print(f"Real:      n_total={len(client_data)}, n_fraud={n_fraud_real}")
    print(f"Synthetic: n_total={len(synthetic)}, n_fraud={n_fraud_synt}")
    print()
    combined = pd.concat([results_real, results_synt], ignore_index=True)
    print(combined.to_string(index=False))
    print()
    print("Metrics: fraud_hit_rate = fraction of known frauds flagged; recall@k / precision@k = top-k by score.")
    if plot_scores_path:
        base = os.path.join(os.path.dirname(plot_scores_path), os.path.splitext(os.path.basename(plot_scores_path))[0])
        real_path = base + "_real.png"
        synt_path = base + "_synthetic.png"
        _plot_anomaly_score_distributions(scores_real, y_real, real_path)
        _plot_anomaly_score_distributions(scores_synt, y_synt, synt_path)
        print(f"Anomaly score distributions saved to {real_path}")
        print(f"Anomaly score distributions saved to {synt_path}")
    return results_real, results_synt, client_data, synthetic


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Anomaly detection: compare models and/or real vs synthetic")
    parser.add_argument("--synthetic", action="store_true", help="Compare real vs synthetic data (default: only real)")
    parser.add_argument("--activity-state", type=int, default=1, help="Filter clients with num_of_trn > this")
    parser.add_argument("--n-synthetic", type=int, default=500, help="Number of synthetic fraud samples")
    args = parser.parse_args()
    if args.synthetic:
        compare_real_vs_synthetic(
            activity_state=args.activity_state,
            n_synthetic=args.n_synthetic,
            plot_scores_path=os.path.join(_project_root, "anomaly_score_distributions.png"),
        )
    else:
        compare_models(
            activity_state=args.activity_state,
            exclude_fraud_from_training=True,
            compare_scalers=False,
            plot_scores_path=os.path.join(_project_root, "anomaly_score_distributions.png"),
        )
