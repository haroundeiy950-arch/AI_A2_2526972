"""Shared fixtures for the test suite."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


@pytest.fixture(scope="session")
def artifacts() -> dict:
    """Load saved models, training them first if they are missing."""
    from src.inference import load_artifacts

    if not (config.MODELS_DIR / "regression_pipeline.joblib").exists():
        import run_all

        run_all.run(config.INSTITUTIONAL_USERNAME, config.SEED)
    return load_artifacts(config.MODELS_DIR)


@pytest.fixture(scope="session")
def valid_record() -> dict:
    return json.loads((config.SAMPLES_DIR / "valid.json").read_text(encoding="utf-8"))
