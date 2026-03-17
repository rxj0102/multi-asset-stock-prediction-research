# Multi-Asset Stock Prediction Research

[![CI](https://github.com/rxj0102/multi-asset-stock-prediction-research/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/multi-asset-stock-prediction-research/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end research pipeline for predicting daily stock returns across multiple assets. Goes beyond standard forecasting to explore **causal relationships**, **feature stability**, **market regime detection**, and **cross-asset transfer learning**.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Features Engineered](#features-engineered)
- [Models Implemented](#models-implemented)
- [Research Extensions](#research-extensions)
- [Key Findings](#key-findings)
- [Development](#development)

---

## Overview

| Dimension | Detail |
|-----------|--------|
| **Assets** | AAPL, MSFT, JPM, XOM, AMZN + S&P 500 + VIX |
| **Period** | 2015-01-01 → present (live via yfinance) |
| **Features** | 21 engineered features per asset |
| **Models** | 11+ algorithms (linear → stacking ensembles) |
| **Research** | Granger causality · regime detection · transfer learning · concept drift |

---

## Project Structure

```
multi-asset-stock-prediction-research/
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI (lint + test)
├── config/
│   └── config.yaml                  # All hyperparameters & experiment settings
├── notebooks/
│   └── 01_stock_prediction_research.ipynb   # Full end-to-end research notebook
├── src/
│   └── stock_prediction/
│       ├── data/
│       │   └── loader.py            # yfinance download & alignment
│       ├── features/
│       │   └── engineering.py       # Full feature pipeline
│       ├── models/
│       │   └── factory.py           # Model constructors, training, tuning
│       └── utils/
│           ├── config.py            # YAML config loader
│           └── evaluation.py        # MSE / RMSE / MAE / R² metrics
├── tests/
│   ├── test_config.py
│   ├── test_features.py
│   └── test_models.py
├── results/                         # Saved plots & artefacts (git-ignored)
├── .gitignore
├── LICENSE
├── Makefile                         # Developer shortcuts
├── pyproject.toml                   # Package metadata & tool config
├── requirements.txt                 # Core dependencies
└── requirements-dev.txt             # Dev / test dependencies
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/rxj0102/multi-asset-stock-prediction-research.git
cd multi-asset-stock-prediction-research
pip install -e ".[dev]"
```

### 2. Run the notebook

```bash
make notebook
# Opens notebooks/01_stock_prediction_research.ipynb
```

### 3. Use the Python API directly

```python
from stock_prediction.utils.config import load_config
from stock_prediction.data.loader import download_market_data, compute_returns_and_align
from stock_prediction.features.engineering import build_features
from stock_prediction.models.factory import (
    get_linear_models, train_and_evaluate, train_test_split_time
)

cfg = load_config()
dc = cfg["data"]

stock_data, market_data, vix_data = download_market_data(
    stocks=dc["stocks"],
    market_index=dc["market_index"],
    vix_index=dc["vix_index"],
    start_date=dc["start_date"],
)
processed = compute_returns_and_align(stock_data, market_data, vix_data)
model_data = build_features(processed, cfg["features"])

ticker = "AAPL"
X, y = model_data[ticker]["X"], model_data[ticker]["y"]
X_tr, X_te, y_tr, y_te = train_test_split_time(X, y)

results = train_and_evaluate(get_linear_models(cfg["modelling"]["linear"]), X_tr, X_te, y_tr, y_te)
print(results)
```

---

## Configuration

All experiment settings live in `config/config.yaml`. No code changes needed to adjust assets, date range, model hyperparameters, or research settings:

```yaml
data:
  stocks: ["AAPL", "MSFT", "JPM", "XOM", "AMZN"]
  start_date: "2015-01-01"

modelling:
  train_test_split: 0.8
  tree:
    n_estimators: 200
    learning_rate: 0.05
```

---

## Features Engineered

### Price-Based (6)

| Feature | Description |
|---------|-------------|
| `Return_Lag_1` | Previous day's log-return |
| `Return_Lag_3` | 3-day rolling mean of returns |
| `Return_Lag_5` | 5-day rolling mean of returns |
| `MA_5` | 5-day simple moving average |
| `MA_20` | 20-day simple moving average |
| `Momentum` | MA_5 − MA_20 |

### Volume-Based (5 core + 4 enhanced = 9)

| Feature | Description |
|---------|-------------|
| `Volume_Change` | Daily volume % change |
| `Avg_Volume_5/20` | Rolling average volume |
| `Volume_Surprise` | Volume / 20-day avg (liquidity shock) |
| `Dollar_Volume` | Close × Volume |
| `Return_x_Volume` | Return_Lag_1 × Volume_Surprise |
| `Momentum_x_Volume` | Momentum × Volume_Surprise |
| `Dollar_Volume_Norm` | Dollar volume / rolling 20-day avg |
| `High_Volume_Day` | Binary: Volume_Surprise > 1.2 |

### Volatility (4)

| Feature | Description |
|---------|-------------|
| `Daily_Range` | (High − Low) / Close |
| `RV_5 / RV_20` | 5/20-day realised volatility |
| `ATR_5` | 5-day average true range |

### Market (2)

| Feature | Description |
|---------|-------------|
| `Market_Return` | S&P 500 daily log-return |
| `VIX_Change` | VIX daily % change |

---

## Models Implemented

| Category | Models |
|----------|--------|
| **Linear** | LinearRegression · Ridge · Lasso · ElasticNet |
| **Tree** | RandomForest · GradientBoosting · ExtraTrees · XGBoost · LightGBM |
| **Ensemble** | VotingRegressor · StackingRegressor (LR meta-model) |
| **Tuned** | RandomizedSearchCV hybrid ensemble |

### Benchmark Results (MSE, test set)

| Model | AAPL | MSFT | JPM | XOM | AMZN |
|-------|------|------|-----|-----|------|
| Linear Regression | 0.000297 | 0.000194 | 0.000255 | 0.000184 | 0.000381 |
| Lasso | 0.000289 | 0.000188 | 0.000232 | 0.000187 | 0.000375 |
| Random Forest | 0.000362 | 0.000212 | 0.000400 | 0.000248 | 0.001052 |
| LightGBM | 0.000337 | 0.000230 | 0.000393 | 0.000241 | 0.000469 |
| Stacking Ensemble | 0.000299 | 0.000196 | 0.000288 | 0.000194 | 0.000393 |

---

## Research Extensions

### 1. Granger Causality Analysis
Tests which features statistically Granger-cause next-day returns.
**Finding**: `Market_Return` (p=0.0007) and `RV_5` (p=0.0166) are causal drivers for AAPL.

### 2. Feature Stability & Concept Drift
Rolling 2-year windows track how feature importance evolves.
**Finding**: Importances spike during market turmoil (COVID 2020, bear market 2022).

### 3. Transfer Learning
Cross-asset prediction: train on one stock, predict another.
**Finding**: Largely ineffective; return correlation (r=0.267) and feature similarity (r=0.144) are the key transferability proxies.

### 4. Market Regime Detection
GMM/KMeans clustering on return distributions identifies 3 regimes (normal / high-return / low-return).
**Finding**: ~85% of days fall in the normal regime; regime-conditional models outperform on tail regimes.

### 5. Temporal Dynamics
Time-varying coefficient analysis reveals non-stationarity.
**Finding**: Feature importance structure shifts significantly post-2020.

---

## Key Findings

1. **Simple beats complex** — Regularised linear models (Lasso/ElasticNet) generally outperform tree-based models, implying a strong linear component in daily returns.
2. **Market context dominates** — `Market_Return` and `VIX_Change` are consistently the highest-importance features.
3. **Regimes matter** — A single model fitted on all regimes leaves alpha on the table for tail environments.
4. **Transferability is low** — Equity return distributions are largely asset-specific; cross-stock models add limited value.

---

## Development

```bash
make install-dev   # install package + dev tools
make lint          # ruff linting
make format        # black auto-format
make test          # pytest
make test-cov      # pytest + coverage report
make clean         # remove build artefacts
```
