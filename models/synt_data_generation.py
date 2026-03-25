import numpy as np
import pandas as pd

def generate_synthetic_data(
    agg_data: pd.DataFrame,
    n_synthetic: int = 500,
    noise_scale: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic fraud samples from real fraud: resample with replacement,
    add Gaussian noise to numeric features, assign new person_id indices.
    Returns a DataFrame with same structure as agg_data: non-fraud rows +
    synthetic fraud (is_fraud=1), index = person_id.
    """
    fraud_real = agg_data[agg_data["is_fraud"] == 1]
    if len(fraud_real) == 0:
        raise ValueError("No fraud rows in agg_data; cannot generate synthetic fraud.")
    synthetic = fraud_real.sample(n=min(n_synthetic, len(fraud_real)), replace=True, random_state=random_state).copy()
    num_cols = synthetic.select_dtypes("number").columns.difference(["is_fraud"], sort=False)
    if "person_id" in synthetic.columns:
        num_cols = num_cols.difference(["person_id"], sort=False)
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0, noise_scale, (len(synthetic), len(num_cols)))
    synthetic.loc[:, num_cols] = synthetic[num_cols].values * (1 + noise)
    start = int(agg_data.index.max()) + 1
    synthetic.index = np.arange(start, start + len(synthetic))
    synthetic.index.name = agg_data.index.name
    synthetic["is_fraud"] = 1
    non_fraud = agg_data[agg_data["is_fraud"] == 0]
    non_fraud['synthetic'] = 0
    synthetic['synthetic'] = 1
    return pd.concat([non_fraud, synthetic], axis=0)

