"""Tests for model factory and evaluation utilities."""

import numpy as np
import pandas as pd
import pytest

from stock_prediction.models.factory import (
    get_linear_models,
    get_tree_models,
    train_and_evaluate,
    train_test_split_time,
)
from stock_prediction.utils.evaluation import evaluate_predictions, results_to_dataframe


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_dataset():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.standard_normal(n), name="target")
    return X, y


# ── evaluate_predictions ──────────────────────────────────────────────────────

def test_evaluate_predictions_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    metrics = evaluate_predictions(y_true, y_pred)
    assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2"}


def test_evaluate_predictions_perfect():
    y = np.array([1.0, 2.0, 3.0])
    metrics = evaluate_predictions(y, y)
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_rmse_is_sqrt_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.5])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


# ── results_to_dataframe ──────────────────────────────────────────────────────

def test_results_to_dataframe_shape():
    results = {
        "AAPL": {"Linear_MSE": 0.01, "Ridge_MSE": 0.009},
        "MSFT": {"Linear_MSE": 0.012, "Ridge_MSE": 0.011},
    }
    df = results_to_dataframe(results)
    assert df.shape == (2, 2)
    assert list(df.index) == ["AAPL", "MSFT"]


# ── train_test_split_time ─────────────────────────────────────────────────────

def test_train_test_split_time_ratio(dummy_dataset):
    X, y = dummy_dataset
    X_tr, X_te, y_tr, y_te = train_test_split_time(X, y, train_ratio=0.8)
    assert len(X_tr) == 160
    assert len(X_te) == 40


def test_train_test_split_chronological(dummy_dataset):
    X, y = dummy_dataset
    X_tr, X_te, _, _ = train_test_split_time(X, y)
    assert X_tr.index[-1] < X_te.index[0]


# ── get_linear_models ─────────────────────────────────────────────────────────

def test_get_linear_models_keys():
    models = get_linear_models()
    assert set(models.keys()) == {"LinearRegression", "Ridge", "Lasso", "ElasticNet"}


def test_linear_models_fit_predict(dummy_dataset):
    X, y = dummy_dataset
    X_tr, X_te, y_tr, y_te = train_test_split_time(X, y)
    for name, model in get_linear_models().items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(X_te), f"{name} prediction length mismatch"


# ── train_and_evaluate ────────────────────────────────────────────────────────

def test_train_and_evaluate_returns_metrics(dummy_dataset):
    X, y = dummy_dataset
    X_tr, X_te, y_tr, y_te = train_test_split_time(X, y)
    results = train_and_evaluate(get_linear_models(), X_tr, X_te, y_tr, y_te)
    for name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
        assert name in results
        assert "mse" in results[name]
        assert results[name]["mse"] >= 0


def test_tree_models_keys():
    models = get_tree_models()
    assert set(models.keys()) == {"RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost", "LightGBM"}
