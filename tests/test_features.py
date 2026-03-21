"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from stock_prediction.features.engineering import (
    ALL_FEATURES,
    add_price_features,
    add_volatility_features,
    add_volume_enhanced_features,
    add_volume_features,
    build_features,
    build_target,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Minimal synthetic OHLCV + Return + market columns."""
    n = 100
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.uniform(-0.005, 0.005, n)),
            "High": close * (1 + rng.uniform(0.005, 0.015, n)),
            "Low": close * (1 - rng.uniform(0.005, 0.015, n)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            "Return": np.concatenate([[np.nan], np.log(close[1:] / close[:-1])]),
            "Market_Return": rng.normal(0, 0.01, n),
            "VIX_Change": rng.normal(0, 0.02, n),
        },
        index=idx,
    )


@pytest.fixture
def processed_data(sample_ohlcv):
    return {"TEST": sample_ohlcv.dropna()}


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_add_price_features_columns(sample_ohlcv):
    result = add_price_features(sample_ohlcv.dropna())
    for col in ["Return_Lag_1", "Return_RollMean_3", "Return_RollMean_5", "MA_5", "MA_20", "Momentum"]:
        assert col in result.columns, f"Missing column: {col}"


def test_add_price_features_no_nans(sample_ohlcv):
    result = add_price_features(sample_ohlcv.dropna())
    assert result[["Return_Lag_1", "MA_5", "MA_20", "Momentum"]].isna().sum().sum() == 0


def test_add_volume_features_columns(sample_ohlcv):
    temp = add_price_features(sample_ohlcv.dropna())
    result = add_volume_features(temp)
    for col in ["Volume_Change", "Avg_Volume_5", "Avg_Volume_20", "Volume_Surprise", "Dollar_Volume"]:
        assert col in result.columns, f"Missing column: {col}"


def test_add_volatility_features_columns(sample_ohlcv):
    temp = add_price_features(sample_ohlcv.dropna())
    temp = add_volume_features(temp)
    result = add_volatility_features(temp)
    for col in ["Daily_Range", "RV_5", "RV_20", "ATR_5"]:
        assert col in result.columns, f"Missing column: {col}"


def test_add_volume_enhanced_features(sample_ohlcv):
    temp = add_price_features(sample_ohlcv.dropna())
    temp = add_volume_features(temp)
    temp = add_volatility_features(temp)
    result = add_volume_enhanced_features(temp)
    for col in ["Return_x_Volume", "Momentum_x_Volume", "Dollar_Volume_Norm", "High_Volume_Day"]:
        assert col in result.columns


def test_high_volume_day_binary(sample_ohlcv):
    temp = add_price_features(sample_ohlcv.dropna())
    temp = add_volume_features(temp)
    temp = add_volatility_features(temp)
    result = add_volume_enhanced_features(temp)
    assert set(result["High_Volume_Day"].unique()).issubset({0, 1})


def test_build_target_shift(sample_ohlcv):
    temp = add_price_features(sample_ohlcv.dropna())
    temp = add_volume_features(temp)
    temp = add_volatility_features(temp)
    temp = add_volume_enhanced_features(temp)
    result = build_target(temp)
    assert "Target" in result.columns
    assert "Target_Dir" in result.columns
    assert result["Target"].isna().sum() == 0


def test_build_features_pipeline(processed_data):
    result = build_features(processed_data)
    assert "TEST" in result
    entry = result["TEST"]
    assert "X" in entry and "y" in entry and "y_dir" in entry
    assert entry["X"].shape[1] == len(ALL_FEATURES)
    assert len(entry["X"]) == len(entry["y"])


def test_build_features_no_nans(processed_data):
    result = build_features(processed_data)
    X = result["TEST"]["X"]
    assert X.isna().sum().sum() == 0
