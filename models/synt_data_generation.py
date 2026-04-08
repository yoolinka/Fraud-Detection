from typing import Optional

import numpy as np
import pandas as pd

_RATE_BOUNDS_ATOL = 1e-8


def _column_values_in_unit_interval(series: pd.Series) -> bool:
    """True if non-null values lie in [0, 1] (rate-like metrics)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    lo, hi = float(s.min()), float(s.max())
    return lo >= -_RATE_BOUNDS_ATOL and hi <= 1.0 + _RATE_BOUNDS_ATOL


def generate_synthetic_data(
    agg_data: pd.DataFrame,
    n_synthetic: int = 500,
    noise_scale: float = 0.1,
    rate_noise_scale: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic fraud samples from real fraud: draw ``n_synthetic`` rows
    with replacement, add Gaussian noise to numeric features, assign new indices.
    Columns whose real-fraud values fall in [0, 1] (rate metrics) use
    ``rate_noise_scale`` (default: 25% of ``noise_scale``) instead of ``noise_scale``.
    Returns a DataFrame with same structure as agg_data: all original non-fraud
    rows plus ``n_synthetic`` noisy fraud copies (is_fraud=1).
    """
    fraud_real = agg_data[agg_data["is_fraud"] == 1]
    if len(fraud_real) == 0:
        raise ValueError("No fraud rows in agg_data; cannot generate synthetic fraud.")
    # With replace=True, draw n_synthetic rows (not capped by len(fraud_real)).
    synthetic = fraud_real.sample(n=n_synthetic, replace=True, random_state=random_state).copy()
    num_cols = synthetic.select_dtypes("number").columns.difference(["is_fraud"], sort=False)
    if "person_id" in synthetic.columns:
        num_cols = num_cols.difference(["person_id"], sort=False)
    num_cols = list(num_cols)
    if rate_noise_scale is None:
        rate_noise_scale = noise_scale * 0.25
    col_std = np.array(
        [
            rate_noise_scale if _column_values_in_unit_interval(fraud_real[c]) else noise_scale
            for c in num_cols
        ],
        dtype=float,
    )
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, 1, (len(synthetic), len(num_cols))) * col_std
    synthetic.loc[:, num_cols] = synthetic[num_cols].values * (1 + noise)
    idx = agg_data.index
    if pd.api.types.is_integer_dtype(idx.dtype) or pd.api.types.is_float_dtype(idx.dtype):
        start = int(np.nanmax(pd.to_numeric(idx, errors="coerce"))) + 1
        synthetic.index = np.arange(start, start + len(synthetic))
    else:
        existing = set(idx.astype(str))
        new_ids = []
        n = 0
        while len(new_ids) < len(synthetic):
            cand = f"__synthetic_fraud__{random_state}_{n}"
            if cand not in existing:
                new_ids.append(cand)
                existing.add(cand)
            n += 1
        synthetic.index = new_ids
    synthetic.index.name = agg_data.index.name
    synthetic["is_fraud"] = 1
    non_fraud = agg_data[agg_data["is_fraud"] == 0]
    non_fraud['synthetic'] = 0
    synthetic['synthetic'] = 1
    return pd.concat([non_fraud, synthetic], axis=0)

