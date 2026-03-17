"""Tests for configuration loading."""

from pathlib import Path

import pytest

from stock_prediction.utils.config import load_config


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)


def test_config_top_level_keys():
    cfg = load_config()
    for key in ("data", "features", "modelling", "research", "plotting"):
        assert key in cfg, f"Missing top-level config key: {key}"


def test_config_stocks_list():
    cfg = load_config()
    stocks = cfg["data"]["stocks"]
    assert isinstance(stocks, list)
    assert len(stocks) > 0


def test_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")
