"""Model factory and training utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

import lightgbm as lgb
import xgboost as xgb

from ..utils.evaluation import evaluate_predictions

logger = logging.getLogger(__name__)


# ── Model constructors ────────────────────────────────────────────────────────

def get_linear_models(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a dict of configured linear regression models."""
    cfg = cfg or {}
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=cfg.get("ridge_alpha", 1.0)),
        "Lasso": Lasso(alpha=cfg.get("lasso_alpha", 0.001)),
        "ElasticNet": ElasticNet(
            alpha=cfg.get("elasticnet_alpha", 0.001),
            l1_ratio=cfg.get("elasticnet_l1_ratio", 0.5),
        ),
    }


def get_tree_models(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a dict of configured tree-based models."""
    cfg = cfg or {}
    n = cfg.get("n_estimators", 200)
    lr = cfg.get("learning_rate", 0.05)
    rs = cfg.get("random_state", 42)
    max_depth = cfg.get("max_depth", None)
    return {
        "RandomForest": RandomForestRegressor(n_estimators=n, max_depth=max_depth, random_state=rs),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=n, learning_rate=lr, max_depth=max_depth or 3, random_state=rs
        ),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=n, max_depth=max_depth, random_state=rs),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=n, learning_rate=lr, max_depth=max_depth or 6, random_state=rs, verbosity=0
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=n, learning_rate=lr, max_depth=max_depth or -1, random_state=rs, verbose=-1
        ),
    }


def get_ensemble_models(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return Voting and Stacking ensemble models."""
    cfg = cfg or {}
    n = cfg.get("voting_n_estimators", 100)
    rs = cfg.get("random_state", 42)

    base = [
        ("rf", RandomForestRegressor(n_estimators=n, random_state=rs)),
        ("gb", GradientBoostingRegressor(n_estimators=n, learning_rate=0.05, random_state=rs)),
        ("et", ExtraTreesRegressor(n_estimators=n, random_state=rs)),
        ("xg", xgb.XGBRegressor(n_estimators=n, learning_rate=0.05, random_state=rs, verbosity=0)),
        ("lgbm", lgb.LGBMRegressor(n_estimators=n, learning_rate=0.05, random_state=rs, verbose=-1)),
    ]
    return {
        "VotingRegressor": VotingRegressor(base),
        "StackingRegressor": StackingRegressor(
            estimators=base,
            final_estimator=LinearRegression(),
            passthrough=cfg.get("stacking_passthrough", True),
        ),
    }


# ── Training helpers ──────────────────────────────────────────────────────────

def train_test_split_time(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological train/test split (no shuffling)."""
    split = int(len(X) * train_ratio)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def train_and_evaluate(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Fit every model and return evaluation metrics.

    Returns
    -------
    dict
        ``{model_name: {"mse": ..., "rmse": ..., "mae": ..., "r2": ...}}``
    """
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_predictions(y_test, preds)
        results[name] = metrics
        logger.info("%s \u2014 RMSE=%.6f  R\u00b2=%.4f", name, metrics["rmse"], metrics["r2"])
    return results


def tune_model(
    model,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
) -> Any:
    """Randomised hyperparameter search with time-series-safe CV.

    Uses :class:`~sklearn.model_selection.TimeSeriesSplit` to avoid
    look-ahead bias during cross-validation.

    Returns
    -------
    Best estimator (fitted).
    """
    tscv = TimeSeriesSplit(n_splits=cv)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info("Tuning %s with %d iterations and %d CV splits", type(model).__name__, n_iter, cv)
    search.fit(X_train, y_train)
    logger.info("Best params: %s", search.best_params_)
    return search.best_estimator_
