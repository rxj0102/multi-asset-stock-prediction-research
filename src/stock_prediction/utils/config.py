"""Configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parents[3] / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML config file. Defaults to ``config/config.yaml`` at the
        project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as fh:
        return yaml.safe_load(fh)
