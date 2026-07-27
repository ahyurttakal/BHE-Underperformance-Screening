"""Shared configuration and package paths for the BHE analysis workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PARAMETER_FILE = PACKAGE_ROOT / "model_parameters.json"
DATA_DIR = PACKAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
MODEL_DIR = PACKAGE_ROOT / "models"
FIGURE_DIR = PACKAGE_ROOT / "figures"
LOG_DIR = PACKAGE_ROOT / "logs"

SEEDS = {
    "sampling": 240517,
    "lightgbm": 240518,
    "bootstrap": 240519,
    "shap": 240520,
    "synthetic": 240521,
}


def load_parameters() -> dict[str, Any]:
    """Load the JSON parameter file distributed with the package."""
    with PARAMETER_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_directories() -> None:
    """Create directories used by generated intermediate and output files."""
    for directory in (
        PROCESSED_DIR,
        INTERMEDIATE_DIR,
        MODEL_DIR,
        FIGURE_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
