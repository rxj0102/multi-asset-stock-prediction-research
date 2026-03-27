"""Model evaluation utilities."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics for a set of predictions.

    Returns
    -------
    dict
        Keys: ``mse``, ``rmse``, ``mae``, ``r2``.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def results_to_dataframe(
    results: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Convert a nested results dict to a tidy DataFrame.

    Parameters
    ----------
    results:
        ``{ticker: {model_name: score}}``

    Returns
    -------
    pd.DataFrame
        Index = tickers, columns = model names.
    """
    return pd.DataFrame(results).T
