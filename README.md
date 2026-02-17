# 📈 Stock Market Prediction with Causal Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project for predicting stock returns that goes beyond simple forecasting to explore causal relationships, feature stability, and transfer learning across multiple assets.

## 🎯 Overview

This project implements an end-to-end data science pipeline for financial time series prediction, featuring:

- **Data Acquisition**: Real-time data from Yahoo Finance for 5 major stocks (AAPL, MSFT, JPM, XOM, AMZN)
- **Feature Engineering**: 21 predictive features including technical indicators, volume metrics, and volatility measures
- **Model Comparison**: 15+ machine learning models from simple linear regression to advanced ensembles
- **Causal Analysis**: Granger causality tests to identify true predictive drivers
- **Stability Analysis**: Tracking feature importance evolution over time
- **Transfer Learning**: Cross-asset prediction experiments and meta-learning

## 📊 Key Results

| Model | AAPL | MSFT | JPM | XOM | AMZN |
|-------|------|------|-----|-----|------|
| Linear Regression | 0.000297 | 0.000194 | 0.000255 | 0.000184 | 0.000381 |
| Lasso | 0.000289 | 0.000188 | 0.000232 | 0.000187 | 0.000375 |
| Random Forest | 0.000362 | 0.000212 | 0.000400 | 0.000248 | 0.001052 |
| LightGBM | 0.000337 | 0.000230 | 0.000393 | 0.000241 | 0.000469 |
| Stacking Ensemble | 0.000299 | 0.000196 | 0.000288 | 0.000194 | 0.000393 |

**Regularized linear models (Lasso/ElasticNet) generally outperform complex tree-based models**, suggesting a strong linear component in daily returns.

# 🎯 Features Engineered

## Price-Based Features

| Feature | Description |
|---------|-------------|
| Return_Lag_1 | Previous day's return |
| Return_Lag_3 | 3-day moving average of returns |
| Return_Lag_5 | 5-day moving average of returns |
| MA_5 | 5-day simple moving average |
| MA_20 | 20-day simple moving average |
| Momentum | MA_5 - MA_20 |

## Volume-Based Features

| Feature | Description |
|---------|-------------|
| Volume_Change | Daily volume % change |
| Avg_Volume_5 | 5-day average volume |
| Avg_Volume_20 | 20-day average volume |
| Volume_Surprise | Volume / 20-day avg volume |
| Dollar_Volume | Close × Volume |
| Return_x_Volume | Return_Lag_1 × Volume_Surprise |
| Momentum_x_Volume | Momentum × Volume_Surprise |
| Dollar_Volume_Norm | Dollar volume / 20-day avg |
| High_Volume_Day | Volume_Surprise > 1.2 |

## Volatility Features

| Feature | Description |
|---------|-------------|
| Daily_Range | (High - Low) / Close |
| RV_5 | 5-day rolling std of returns |
| RV_20 | 20-day rolling std of returns |
| ATR_5 | 5-day average true range |

## Market Features

| Feature | Description |
|---------|-------------|
| Market_Return | S&P 500 daily return |
| VIX_Change | VIX index daily % change |

# 🤖 Models Implemented

## Linear Models

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization + feature selection |
| ElasticNet | Combined L1/L2 regularization |

## Tree-Based Models

| Model | Description |
|-------|-------------|
| Random Forest | 200 trees with bootstrap aggregation |
| Gradient Boosting | Sequential error correction |
| Extra Trees | Random splits for variance reduction |
| XGBoost | Optimized gradient boosting |
| LightGBM | Lightweight leaf-wise growth |

## Ensemble Methods

| Model | Description |
|-------|-------------|
| Voting Regressor | Average of all tree-based models |
| Stacking Regressor | Linear regression as meta-model |
| Hybrid Ensemble | Tuned models + stacking (Cell 15) |

# 🔬 Advanced Analysis

## 1. Causal Inference (Cell 17)

```python
# Tests if features Granger-cause returns
grangercausalitytests(data, maxlag=5)
```

**Key Finding**: For AAPL, `Market_Return` (p=0.0007) and `RV_5` (p=0.0166) Granger-cause returns at 5% significance.

## 2. Feature Stability (Cell 18)

- Rolling 2-year windows track importance evolution
- Coefficient of variation measures stability
- Structural break detection via statistical tests

**Key Finding**: Feature importances spike during market turmoil (2020 COVID crash, 2022 bear market).

## 3. Transfer Learning (Cell 19)

- Cross-stock prediction experiments
- Transferability heatmap visualization
- Meta-learning to predict transfer success

**Key Finding**: Cross-asset prediction is largely ineffective, but return correlation (r=0.267) and feature similarity (r=0.144) are key transferability factors.

# 📊 Visualizations

| Visualization | Description | Location |
|---------------|-------------|----------|
| Model Comparison | MSE metrics for all models | Cell 14 |
| Feature Importance | Top features per stock | Cell 14 |
| Granger Causality | Causal driver summary | Cell 17 |
| Feature Stability | Importance evolution over time | Cell 18 |
| Instability Timeline | Volatility of feature importance | Cell 18 |
| Transfer Heatmap | Cross-stock R² scores | Cell 19 |
| Meta-Feature Importance | Transfer success factors | Cell 19 |
