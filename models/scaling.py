import math
import os
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import load_data, FEATURES

df, client_data, waiter_week_data, waiter_level_data = load_data()

_RATE_BOUNDS_ATOL = 1e-8


def _column_values_in_unit_interval(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    lo, hi = float(s.min()), float(s.max())
    return lo >= -_RATE_BOUNDS_ATOL and hi <= 1.0 + _RATE_BOUNDS_ATOL


def _use_log1p_for_skew(
    col: str,
    df: pd.DataFrame,
    nf_mask: np.ndarray,
    skew_abs_threshold: float,
) -> bool:
    """Same rule as `waiter_week_use_log1p` in features_selection / gini notebooks."""
    if col == "is_fraud" or col not in df.columns:
        return False
    if _column_values_in_unit_interval(df[col]):
        return False
    s = pd.to_numeric(df.loc[nf_mask, col], errors="coerce").dropna()
    v = np.asarray(s, dtype=np.float64)
    s = s[np.isfinite(v)]
    if len(s) < 3:
        return False
    return abs(float(s.skew())) >= skew_abs_threshold


def _auto_skewed_columns(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    skew_abs_threshold: float,
) -> list[str]:
    if "is_fraud" in df.columns:
        nf_mask = df["is_fraud"].to_numpy() == 0
    else:
        nf_mask = np.ones(len(df), dtype=bool)
    return [
        c
        for c in feature_names
        if _use_log1p_for_skew(c, df, nf_mask, skew_abs_threshold)
    ]


def _resolve_skewed(
    skew_reference: pd.DataFrame,
    feature_names: Sequence[str],
    skew_abs_threshold: float,
) -> list[str]:
    return _auto_skewed_columns(skew_reference, feature_names, skew_abs_threshold)


def scale_features(
        data = client_data,
        scaler_type = "standard",
        features = FEATURES,
        skew_abs_threshold: float = 1.0,
        show_charts = False,
        fit_data = None,
        impute_reference = None,
):
    """
    Scale features (log1p for skewed, then Standard or Robust scaler).
    If skewed is None, log1p targets are chosen like features_selection.ipynb: among `features`,
    use non-fraud rows' skew when is_fraud exists; skip [0,1]-bounded columns; |skew| >= skew_abs_threshold.
    If fit_data is None: fit and transform on data (returns one DataFrame).
    If fit_data is not None: fit on fit_data, transform data; returns (scaled_fit_data, scaled_data).
    If impute_reference is not None, NaN filling uses medians from that frame only (e.g. train
    non-fraud) so test rows do not leak into imputation statistics.
    """
    skew_cols = _resolve_skewed(data, features, skew_abs_threshold)
    X = data[features].replace([np.inf, -np.inf], np.nan)
    if impute_reference is not None:
        med = impute_reference[features].median()
        X = X.fillna(med).fillna(0)
    else:
        X = X.fillna(X.median()).fillna(0)
    for col in skew_cols:
        X[col] = np.log1p(X[col])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median()).fillna(0)

    if fit_data is not None:
        X_fit = fit_data[features].copy().replace([np.inf, -np.inf], np.nan)
        if impute_reference is not None:
            med = impute_reference[features].median()
            X_fit = X_fit.fillna(med).fillna(0)
        else:
            X_fit = X_fit.fillna(fit_data[features].median()).fillna(0)
        for col in skew_cols:
            X_fit[col] = np.log1p(X_fit[col])
        X_fit = X_fit.replace([np.inf, -np.inf], np.nan).fillna(X_fit.median()).fillna(0)
        std_scaler = StandardScaler()
        rob_scaler = RobustScaler()
        std_scaler.fit(X_fit)
        rob_scaler.fit(X_fit)
        X_std_fit = std_scaler.transform(X_fit)
        X_rob_fit = rob_scaler.transform(X_fit)
        X_std = std_scaler.transform(X)
        X_rob = rob_scaler.transform(X)
        X_std = pd.DataFrame(X_std, columns=features, index=data.index)
        X_rob = pd.DataFrame(X_rob, columns=features, index=data.index)
        X_std_fit = pd.DataFrame(X_std_fit, columns=features, index=fit_data.index)
        X_rob_fit = pd.DataFrame(X_rob_fit, columns=features, index=fit_data.index)
        if scaler_type == "standard":
            return X_std_fit, X_std
        elif scaler_type == "robust":
            return X_rob_fit, X_rob
        raise ValueError("SORRYYY PLS PAST scaler_type value 'standard' or 'robust'")

    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()
    X_std = pd.DataFrame(std_scaler.fit_transform(X), columns=features, index=data.index)
    X_rob = pd.DataFrame(rob_scaler.fit_transform(X), columns=features, index=data.index)

    # Show visualization of raw vs scaled features
    if show_charts:
        n_cols = 3
        n_rows = math.ceil(len(features) / n_cols)

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=features,
            horizontal_spacing=0.03,
            vertical_spacing=0.03
        )

        for i, feat in enumerate(features):
            r = i // n_cols + 1
            c = i % n_cols + 1

            # Raw
            fig.add_trace(
                go.Box(
                    y=X[feat],
                    name="Raw",
                    marker_color="#636EFA",
                    boxpoints=False,
                    width=0.4,
                    legendgroup="Raw",
                    showlegend=(i == 0)
                ),
                row=r, col=c
            )

            # Standard
            fig.add_trace(
                go.Box(
                    y=X_std[feat],
                    name="Standard",
                    marker_color="#EF553B",
                    boxpoints=False,
                    width=0.4,
                    legendgroup="Standard",
                    showlegend=(i == 0)
                ),
                row=r, col=c
            )

            # Robust
            fig.add_trace(
                go.Box(
                    y=X_rob[feat],
                    name="Robust",
                    marker_color="#00CC96",
                    boxpoints=False,
                    width=0.4,
                    legendgroup="Robust",
                    showlegend=(i == 0)
                ),
                row=r, col=c
            )

        fig.update_layout(
            height=400 * n_rows,
            width=1800,
            boxmode="group",
            template="plotly_white",
            title="Raw vs StandardScaler vs RobustScaler (Side-by-Side)"
        )
        fig.update_yaxes(type="log")
        fig.show()
    
    if scaler_type == "standard":
        return X_std
    elif scaler_type == "robust":
        return X_rob
    else: 
        raise ValueError("SORRYYY PLS PAST scaler_type value 'standard' or 'robust'")