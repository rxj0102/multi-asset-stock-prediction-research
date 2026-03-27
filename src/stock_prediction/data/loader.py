"""Market data downloading and preprocessing."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns produced by yfinance to a flat Index."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_stock(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download OHLCV data for a single ticker via yfinance."""
    logger.info("Downloading %s from %s to %s", ticker, start_date, end_date or "today")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df = _flatten_columns(df)
    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start_date} and "
            f"{end_date or 'today'}. Check the ticker symbol and date range."
        )
    n_before = len(df)
    df.dropna(inplace=True)
    if len(df) < n_before:
        logger.warning("%s: dropped %d rows with NaN values (%d rows remain)", ticker, n_before - len(df), len(df))
    logger.info("%s: %d rows downloaded", ticker, len(df))
    return df


def download_market_data(
    stocks: list[str],
    market_index: str,
    vix_index: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Download price data for all assets and macro indices.

    Parameters
    ----------
    stocks:
        List of equity tickers.
    market_index:
        Broad market index ticker (e.g. ``"^GSPC"``).
    vix_index:
        Volatility index ticker (e.g. ``"^VIX"``).
    start_date:
        Download start in ``YYYY-MM-DD`` format.
    end_date:
        Download end (``None`` = today).

    Returns
    -------
    stock_data:
        ``{ticker: OHLCV DataFrame}``
    market_data:
        S&P 500 OHLCV with an added ``Market_Return`` column.
    vix_data:
        VIX OHLCV with an added ``VIX_Change`` column.
    """
    stock_data: Dict[str, pd.DataFrame] = {}
    for ticker in stocks:
        stock_data[ticker] = download_stock(ticker, start_date, end_date)

    market_data = download_stock(market_index, start_date, end_date)
    vix_data = download_stock(vix_index, start_date, end_date)

    market_data["Market_Return"] = np.log(
        market_data["Close"] / market_data["Close"].shift(1)
    )
    vix_data["VIX_Change"] = vix_data["Close"].pct_change()

    return stock_data, market_data, vix_data


def compute_returns_and_align(
    stock_data: Dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    vix_data: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Compute log-returns for each stock and align with market/VIX.

    Parameters
    ----------
    stock_data:
        Raw OHLCV DataFrames keyed by ticker.
    market_data:
        Market index DataFrame with ``Market_Return`` column.
    vix_data:
        VIX DataFrame with ``VIX_Change`` column.

    Returns
    -------
    dict
        ``{ticker: aligned DataFrame}``
    """
    processed: Dict[str, pd.DataFrame] = {}
    for ticker, df in stock_data.items():
        temp = df.copy()
        temp["Return"] = np.log(temp["Close"] / temp["Close"].shift(1))
        temp = temp.join(market_data[["Market_Return"]], how="inner")
        temp = temp.join(vix_data[["VIX_Change"]], how="inner")
        n_before = len(temp)
        temp.dropna(inplace=True)
        if len(temp) < n_before:
            logger.warning(
                "%s: dropped %d rows during alignment (%d rows remain)",
                ticker, n_before - len(temp), len(temp),
            )
        logger.info("%s: %d rows after alignment", ticker, len(temp))
        processed[ticker] = temp
    return processed
