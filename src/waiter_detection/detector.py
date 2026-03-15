"""Anomaly detection models and evaluation."""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


class AnomalyDetector:
    """Anomaly detection for waiters."""
    
    def __init__(self, method: str = 'isolation_forest', 
                 contamination: float = 0.01,
                 scaler_type: str = 'standard'):
        """
        Initialize detector.
        
        Args:
            method: 'isolation_forest', 'one_class_svm', 'lof', or 'all'
            contamination: Expected proportion of anomalies
            scaler_type: 'standard' or 'robust'
        """
        self.method = method
        self.contamination = contamination
        self.scaler_type = scaler_type
        self.scaler = None
        self.models = {}
        
    def fit(self, X: np.ndarray):
        """Fit the detector(s) on data."""
        # Scale features
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle infinite and NaN values
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.method in ['isolation_forest', 'all']:
            iso = IsolationForest(
                n_estimators=300,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            iso.fit(X_scaled)
            self.models['iso'] = iso
        
        if self.method in ['one_class_svm', 'all']:
            ocsvm = OneClassSVM(kernel='rbf', nu=self.contamination, gamma='scale')
            ocsvm.fit(X_scaled)
            self.models['ocsvm'] = ocsvm
        
        if self.method in ['lof', 'all']:
            n_neighbors = min(50, len(X_scaled) - 1)
            if n_neighbors < 5:
                n_neighbors = 5
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self.contamination,
                novelty=True
            )
            lof.fit(X_scaled)
            self.models['lof'] = lof
    
    def predict(self, X: np.ndarray) -> dict:
        """
        Predict anomalies.
        
        Returns:
            Dictionary with method names as keys and (scores, labels) as values
        """
        if self.scaler is None:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        results = {}
        
        if 'iso' in self.models:
            scores = -self.models['iso'].score_samples(X_scaled)
            labels = self.models['iso'].predict(X_scaled)
            results['iso'] = (scores, labels)
        
        if 'ocsvm' in self.models:
            scores = -self.models['ocsvm'].decision_function(X_scaled)
            labels = self.models['ocsvm'].predict(X_scaled)
            results['ocsvm'] = (scores, labels)
        
        if 'lof' in self.models:
            scores = -self.models['lof'].score_samples(X_scaled)
            labels = self.models['lof'].predict(X_scaled)
            results['lof'] = (scores, labels)
        
        return results


def evaluate(results_df: pd.DataFrame, label_col: str = 'is_fraud_waiter') -> pd.DataFrame:
    """
    Evaluate detection performance.
    
    Args:
        results_df: DataFrame with anomaly scores, labels, and ground truth
        label_col: Column name with ground truth labels
        
    Returns:
        Evaluation metrics
    """
    print("Evaluating detection performance...")
    
    y_true = results_df[label_col].values
    
    metrics = []
    
    for method in ['iso', 'ocsvm', 'lof']:
        score_col = f'{method}_score'
        label_col_method = f'{method}_label'
        
        if score_col not in results_df.columns:
            continue
        
        scores = results_df[score_col].values
        labels = results_df[label_col_method].values
        
        # Basic metrics
        n_anomalies = (labels == -1).sum()
        n_fraud = y_true.sum()
        fraud_flagged = ((labels == -1) & (y_true == 1)).sum()
        
        hit_rate = fraud_flagged / n_fraud if n_fraud > 0 else 0
        precision = fraud_flagged / n_anomalies if n_anomalies > 0 else 0
        
        # Top-K metrics
        top_k_list = [10, 20, 50, 100]
        top_k_recall = {}
        top_k_precision = {}
        
        for k in top_k_list:
            top_k_indices = np.argsort(-scores)[:min(k, len(scores))]
            fraud_in_top_k = y_true[top_k_indices].sum()
            top_k_recall[f'recall@{k}'] = fraud_in_top_k / n_fraud if n_fraud > 0 else 0
            top_k_precision[f'precision@{k}'] = fraud_in_top_k / k
        
        metrics.append({
            'method': method.upper(),
            'n_anomalies': n_anomalies,
            'pct_flagged': 100 * n_anomalies / len(results_df),
            'hit_rate': hit_rate,
            'precision': precision,
            **top_k_recall,
            **top_k_precision,
        })
    
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

