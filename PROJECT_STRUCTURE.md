# Project Structure

## Overview

This project has been refactored into a clean, modular structure with Hydra configuration management.

## Key Changes

### 1. **Removed Data Leakage**
   - ❌ **REMOVED**: All fraud card features from feature extraction
   - ✅ **KEPT**: Fraud cards only for evaluation/target labels
   - Features now capture behavioral patterns without using fraud information

### 2. **Clean Architecture**
   ```
   src/waiter_detection/
   ├── graph.py      # Graph construction & graph features
   ├── features.py   # Transaction features (NO fraud info)
   ├── detector.py   # Anomaly detection models
   └── pipeline.py   # Main pipeline orchestration
   ```

### 3. **Hydra Configuration**
   - All parameters in `config/` directory
   - Easy to switch models: `python scripts/train.py model=one_class_svm`
   - Override any parameter: `python scripts/train.py model.contamination=0.02`

### 4. **Removed Redundant Files**
   - Old `models/graph_waiter_detection.py` → Refactored into modules
   - Old documentation files → Consolidated into README.md
   - Old notebooks → Moved to `notebooks/` directory

## Usage

### Basic Usage
```bash
python scripts/train.py
```

### With Different Model
```bash
python scripts/train.py model=isolation_forest
python scripts/train.py model=one_class_svm
python scripts/train.py model=lof
python scripts/train.py model=all
```

### Override Parameters
```bash
python scripts/train.py model.contamination=0.02 model.scaler_type=robust
```

## Feature Engineering

### Graph Features (NO fraud info)
- Degree (number of cards)
- Transaction patterns per card
- Amount patterns
- Bonus usage patterns
- Centrality measures
- Clustering coefficients
- Card sharing patterns
- First transaction patterns

### Transaction Features (NO fraud info)
- Volume statistics
- Financial aggregations
- Bonus patterns
- Time-based features
- Behavioral ratios

### Fraud Labels (Evaluation Only)
- `is_fraud_waiter`: Binary label (1 if waiter connected to fraud cards)
- Used ONLY for evaluation metrics
- NOT included in feature set

## Outputs

Results saved to `outputs/`:
- `waiter_anomaly_results.parquet` - Full results with scores
- `waiter_anomaly_metrics.csv` - Evaluation metrics

## Configuration Files

- `config/config.yaml` - Main configuration
- `config/data/default.yaml` - Data paths
- `config/model/*.yaml` - Model-specific configs

## Important Notes

⚠️ **No Data Leakage**: Fraud card information is completely excluded from features. It's only used for:
- Evaluation (calculating metrics)
- Target labels (if using supervised learning)

This ensures realistic performance evaluation and prevents overfitting.

