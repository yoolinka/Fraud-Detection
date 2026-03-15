"""Main pipeline for waiter anomaly detection."""

from typing import List, Tuple
import pandas as pd

from .graph import WaiterCardGraph
from .features import extract_transaction_features, get_fraud_labels
from .detector import AnomalyDetector, evaluate


def run_pipeline(
    df: pd.DataFrame,
    fraud_person_ids: List[int],
    min_transactions: int = 1,
    method: str = 'isolation_forest',
    contamination: float = 0.01,
    scaler_type: str = 'standard',
    use_fraud_labels: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete graph-based waiter anomaly detection pipeline.
    
    Args:
        df: Processed transaction dataframe
        fraud_person_ids: List of known fraud person_ids (for evaluation only)
        min_transactions: Minimum transactions to include an edge
        method: Anomaly detection method
        contamination: Expected proportion of anomalies
        scaler_type: Feature scaling method
        use_fraud_labels: Whether to add fraud labels for evaluation
        
    Returns:
        Tuple of (results_df, metrics_df)
    """
    # Build graph
    # graph_builder = WaiterCardGraph(df)
    # graph = graph_builder.build(min_transactions=min_transactions)
    
    # Extract features (NO fraud information)
    # graph features
    transaction_features = extract_transaction_features(df)
    
    # Combine features
    # combined_features = transaction_features.merge(
    #     graph_features, on='waiter_id', how='outer'
    # ).fillna(0)

    combined_features = transaction_features

    # Get fraud labels (ONLY for evaluation, not features)
    if use_fraud_labels:
        fraud_labels = get_fraud_labels(df, fraud_person_ids)
        print("Fraud waiters: ", len(fraud_labels))
        combined_features = combined_features.merge(
            fraud_labels, on='waiter_id', how='left'
        ).fillna(0)
    else:
        combined_features['is_fraud_waiter'] = 0
    
    # Prepare features for detection
    exclude_cols = ['waiter_id', 'is_fraud_waiter', 'first_trn_date', 'last_trn_date']
    feature_cols = [
        c for c in combined_features.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(combined_features[c])
    ]

    print("Feature columns: ", feature_cols)
    print("Combined features shape: ", combined_features.shape)
    
    X = combined_features[feature_cols].values
    
    # Detect anomalies
    detector = AnomalyDetector(
        method=method,
        contamination=contamination,
        scaler_type=scaler_type
    )
    detector.fit(X)
    
    predictions = detector.predict(X)
    
    # Create results dataframe
    results = combined_features[['waiter_id', 'is_fraud_waiter']].copy()
    
    for method_name, (scores, labels) in predictions.items():
        results[f'{method_name}_score'] = scores
        results[f'{method_name}_label'] = labels
    
    # Evaluate
    if use_fraud_labels and results['is_fraud_waiter'].sum() > 0:
        metrics = evaluate(results)
    else:
        metrics = pd.DataFrame()
    
    return results, metrics

