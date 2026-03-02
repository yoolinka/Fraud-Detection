import math
import os
import sys
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import RobustScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

sys.path.append(os.path.abspath(".."))
from config import load_data, FEATURES, SKEWED
df, client_data = load_data()

def scale_features(
        data = client_data,
        scaler_type = "standard",
        features = FEATURES,
        skewed = SKEWED,
        show_charts = False,
):
    X = data[features]
    X = X.fillna(X.median())
    
    for col in skewed:
        X[col] = np.log1p(X[col])

    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    X_std = pd.DataFrame(std_scaler.fit_transform(X), columns=features)
    X_rob = pd.DataFrame(rob_scaler.fit_transform(X), columns=features)

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