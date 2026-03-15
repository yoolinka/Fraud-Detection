"""Main training script with Hydra configuration."""

import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath('..'))

from src.waiter_detection.pipeline import run_pipeline


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run waiter anomaly detection pipeline."""
    
    # Load data
    print(f"Loading data from {cfg.input.processed_path}...")
    df = pd.read_parquet(cfg.input.processed_path)
    
    print(f"Loaded {len(df):,} transactions")
    print(f"Unique waiters: {df['waiter_id'].nunique():,}")
    print(f"Unique loyalty cards: {df['person_id'].nunique():,}")
    print(f"Known fraud cards: {len(cfg.input.fraud_person_ids)}")
    
    # Run pipeline
    results, metrics = run_pipeline(
        df=df,
        fraud_person_ids=cfg.input.fraud_person_ids,
        min_transactions=cfg.graph.min_transactions,
        method=cfg.model.method,
        contamination=cfg.model.contamination,
        scaler_type=cfg.model.scaler_type,
        use_fraud_labels=True,
    )
    
    # Save results
    if cfg.output.save_results:
        output_dir = cfg.output.dir
        os.makedirs(output_dir, exist_ok=True)
        
        results_path = os.path.join(output_dir, "waiter_anomaly_results.parquet")
        results.to_parquet(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        if cfg.output.save_metrics and len(metrics) > 0:
            metrics_path = os.path.join(output_dir, "waiter_anomaly_metrics.csv")
            metrics.to_csv(metrics_path, index=False)
            print(f"Metrics saved to {metrics_path}")
            
            print("\nEvaluation Metrics:")
            print("=" * 60)
            print(metrics.to_string(index=False))
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

