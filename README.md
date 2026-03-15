# Waiter Fraud Detection

Graph-based anomaly detection system for identifying fraudulent waiters who create fake loyalty cards.

## Project Structure

```
.
├── src/
│   └── waiter_detection/      # Main package
│       ├── graph.py           # Graph construction and graph features
│       ├── features.py        # Transaction-based features
│       ├── detector.py        # Anomaly detection models
│       └── pipeline.py        # Main pipeline
├── config/                     # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── data/                  # Data configurations
│   └── model/                 # Model configurations
├── scripts/
│   └── train.py              # Main training script
├── notebooks/                 # Jupyter notebooks for exploration
├── data/                      # Data files (gitignored)
└── outputs/                   # Results (gitignored)
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run with default configuration

```bash
python scripts/train.py
```

### Run with different model

```bash
python scripts/train.py model=one_class_svm
python scripts/train.py model=lof
python scripts/train.py model=all
```

### Override parameters

```bash
python scripts/train.py model.contamination=0.02 model.scaler_type=robust
```

## Key Features

- **Graph-based features**: Captures relationships between waiters and loyalty cards
- **No data leakage**: Fraud card information is NOT used as features, only for evaluation
- **Multiple algorithms**: Isolation Forest, One-Class SVM, LOF
- **Hydra configuration**: Easy parameter management
- **Clean architecture**: Modular, testable code

## Configuration

Edit `config/config.yaml` to change:
- Data paths
- Fraud person IDs (for evaluation only)
- Graph parameters
- Model parameters
- Output settings

## Outputs

Results are saved to `outputs/`:
- `waiter_anomaly_results.parquet` - Full results with scores
- `waiter_anomaly_metrics.csv` - Evaluation metrics

## Important Notes

⚠️ **Fraud cards are NOT used as features** - they are only used for:
- Evaluation (calculating metrics)
- Target labels (if using supervised learning)

This ensures no data leakage and realistic performance evaluation.

