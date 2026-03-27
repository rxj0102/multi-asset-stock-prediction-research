"""Feature engineering pipeline for financial time series."""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


# ── Base feature sets (default windows) ──────────────────────────────────────

PRICE_FEATURES = [
    "Return_Lag_1", "Return_RollMean_3", "Return_RollMean_5",
    "MA_5", "MA_20", "Momentum",
]

VOLUME_FEATURES = [
    "Volume_Change", "Avg_Volume_5", "Avg_Volume_20",
    "Volume_Surprise", "Dollar_Volume",
]

VOLATILITY_FEATURES = [
    "Daily_Range", "RV_5", "RV_20", "ATR_5",
]

MARKET_FEATURES = ["Market_Return", "VIX_Change"]

VOLUME_ENHANCED_FEATURES = [
    "Return_x_Volume", "Momentum_x_Volume",
    "Dollar_Volume_Norm", "High_Volume_Day",
]

ALL_FEATURES = (
    PRICE_FEATURES
    + VOLUME_FEATURES
    + VOLATILITY_FEATURES
    + MARKET_FEATURES
    + VOLUME_ENHANCED_FEATURES
)


# ── Individual engineering steps ─────────────────────────────────────────────

def add_price_features(
    df: pd.DataFrame,
    lag_windows: list[int] | None = None,
    ma_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged returns and moving-average momentum features.

    For ``lag_windows``, window=1 produces a true lag (shift); all other
    windows produce a rolling mean (``Return_RollMean_<w>``).
    """
    lag_windows = lag_windows or [1, 3, 5]
    ma_windows = ma_windows or [5, 20]

    n_before = len(df)
    temp = df.copy()
    for w in lag_windows:
        if w == 1:
            temp["Return_Lag_1"] = temp["Return"].shift(1)
        else:
            temp[f"Return_RollMean_{w}"] = temp["Return"].rolling(w).mean()

    short_w, long_w = ma_windows[0], ma_windows[1]
    temp[f"MA_{short_w}"] = temp["Close"].rolling(short_w).mean()
    temp[f"MA_{long_w}"] = temp["Close"].rolling(long_w).mean()
    temp["Momentum"] = temp[f"MA_{short_w}"] - temp[f"MA_{long_w}"]

    result = temp.dropna()
    logger.debug("add_price_features: %d \u2192 %d rows (dropped %d)", n_before, len(result), n_before - len(result))
    return result


def add_volume_features(
    df: pd.DataFrame,
    avg_windows: list[int] | None = None,
    high_volume_threshold: float = 1.2,
) -> pd.DataFrame:
    """Add volume-based liquidity and activity features."""
    avg_windows = avg_windows or [5, 20]

    n_before = len(df)
    temp = df.copy()
    temp["Volume_Change"] = temp["Volume"].pct_change()

    for w in avg_windows:
        temp[f"Avg_Volume_{w}"] = temp["Volume"].rolling(w).mean()

    long_w = avg_windows[-1]
    temp["Volume_Surprise"] = temp["Volume"] / temp[f"Avg_Volume_{long_w}"]
    temp["Dollar_Volume"] = temp["Close"] * temp["Volume"]

    result = temp.dropna()
    logger.debug("add_volume_features: %d \u2192 %d rows (dropped %d)", n_before, len(result), n_before - len(result))
    return result


def add_volatility_features(
    df: pd.DataFrame,
    rv_windows: list[int] | None = None,
    atr_window: int = 5,
) -> pd.DataFrame:
    """Add realised-volatility and price-range features.

    ATR is computed as the rolling mean of the True Range, where True Range
    = max(High - Low, |High - prev_Close|, |Low - prev_Close|).
    """
    rv_windows = rv_windows or [5, 20]

    n_before = len(df)
    temp = df.copy()
    temp["Daily_Range"] = (temp["High"] - temp["Low"]) / temp["Close"]

    for w in rv_windows:
        temp[f"RV_{w}"] = temp["Return"].rolling(w).std()

    prev_close = temp["Close"].shift(1)
    true_range = pd.concat(
        [
            temp["High"] - temp["Low"],
            (temp["High"] - prev_close).abs(),
            (temp["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    temp[f"ATR_{atr_window}"] = true_range.rolling(atr_window).mean()

    result = temp.dropna()
    logger.debug("add_volatility_features: %d \u2192 %d rows (dropped %d)", n_before, len(result), n_before - len(result))
    return result


def add_volume_enhanced_features(
    df: pd.DataFrame,
    high_volume_threshold: float = 1.2,
) -> pd.DataFrame:
    """Add interaction and normalisation features that combine volume and price."""
    n_before = len(df)
    temp = df.copy()
    temp["Return_x_Volume"] = temp["Return_Lag_1"] * temp["Volume_Surprise"]
    temp["Momentum_x_Volume"] = temp["Momentum"] * temp["Volume_Surprise"]
    temp["Dollar_Volume_Norm"] = temp["Dollar_Volume"] / temp["Dollar_Volume"].rolling(20).mean()
    temp["High_Volume_Day"] = (temp["Volume_Surprise"] > high_volume_threshold).astype(int)
    result = temp.dropna()
    logger.debug(
        "add_volume_enhanced_features: %d \u2192 %d rows (dropped %d)",
        n_before, len(result), n_before - len(result),
    )
    return result


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Append next-day return (regression) and direction (classification) targets."""
    temp = df.copy()
    temp["Target"] = temp["Return"].shift(-1)
    temp["Target_Dir"] = (temp["Target"] > 0).astype(int)
    return temp.dropna()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_features(
    processed_data: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Run the complete feature-engineering pipeline for all tickers.

    Parameters
    ----------
    processed_data:
        ``{ticker: aligned OHLCV + Return + Market_Return + VIX_Change}``
    cfg:
        Optional feature config sub-dict (``config["features"]``).  Uses
        sensible defaults when not provided.

    Returns
    -------
    dict
        ``{ticker: {"X": features_df, "y": regression_target, "y_dir": binary_target}}``
    """
    cfg = cfg or {}
    price_cfg = cfg.get("price", {})
    vol_cfg = cfg.get("volume", {})
    vola_cfg = cfg.get("volatility", {})
    high_vol_thr = vol_cfg.get("high_volume_threshold", 1.2)

    lag_windows = price_cfg.get("lag_windows") or [1, 3, 5]
    ma_windows = price_cfg.get("ma_windows") or [5, 20]
    avg_windows = vol_cfg.get("avg_windows") or [5, 20]
    rv_windows = vola_cfg.get("rv_windows") or [5, 20]
    atr_window = vola_cfg.get("atr_window", 5)

    # Build feature column list dynamically from the configured windows so
    # that changes to window sizes in config are always reflected in X.
    lag_feats = ["Return_Lag_1"] + [f"Return_RollMean_{w}" for w in lag_windows if w != 1]
    ma_feats = [f"MA_{w}" for w in ma_windows] + ["Momentum"]
    vol_feats = (
        ["Volume_Change"]
        + [f"Avg_Volume_{w}" for w in avg_windows]
        + ["Volume_Surprise", "Dollar_Volume"]
    )
    vola_feats = ["Daily_Range"] + [f"RV_{w}" for w in rv_windows] + [f"ATR_{atr_window}"]
    feature_cols = lag_feats + ma_feats + vol_feats + vola_feats + MARKET_FEATURES + VOLUME_ENHANCED_FEATURES

    final_model_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    for ticker, df in processed_data.items():
        logger.info("Building features for %s (input rows: %d)", ticker, len(df))
        temp = add_price_features(df, lag_windows=lag_windows, ma_windows=ma_windows)
        temp = add_volume_features(temp, avg_windows=avg_windows, high_volume_threshold=high_vol_thr)
        temp = add_volatility_features(temp, rv_windows=rv_windows, atr_window=atr_window)
        temp = add_volume_enhanced_features(temp, high_vol_thr)
        temp = build_target(temp)

        X = temp[feature_cols].copy()
        logger.info("Features built for %s: %d rows \u00d7 %d cols", ticker, len(X), X.shape[1])

        final_model_data[ticker] = {
            "X": X,
            "y": temp["Target"],
            "y_dir": temp["Target_Dir"],
        }

    return final_model_data
