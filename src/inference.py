"""Artifact loading and single-record inference shared by the CLI and the API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src import config


class RecordValidationError(ValueError):
    """Raised when an input record does not satisfy the schema and ranges."""


REQUIRED_RECORD_FIELDS = list(config.FEATURE_COLUMNS)


def validate_record(record: Any) -> dict:
    """Type-, completeness- and range-check one raw input record.

    Raises ``RecordValidationError`` with a message that names the offending
    field and the rule it broke. Returns a normalised record.
    """
    if not isinstance(record, dict):
        raise RecordValidationError(
            f"record must be a JSON object with fields {REQUIRED_RECORD_FIELDS}, "
            f"got {type(record).__name__}"
        )

    missing = [f for f in REQUIRED_RECORD_FIELDS if f not in record]
    if missing:
        raise RecordValidationError(
            f"missing required field(s): {missing}. Required fields are "
            f"{REQUIRED_RECORD_FIELDS}"
        )

    unknown = [f for f in record if f not in REQUIRED_RECORD_FIELDS]
    if unknown:
        raise RecordValidationError(f"unknown field(s): {unknown}")

    cleaned: dict = {}
    for field in REQUIRED_RECORD_FIELDS:
        value = record[field]
        if field in config.RANGES:
            low, high = config.RANGES[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RecordValidationError(
                    f"field '{field}' must be a number between {low} and {high}, "
                    f"got {value!r} ({type(value).__name__})"
                )
            value = float(value)
            if not (low <= value <= high):
                raise RecordValidationError(
                    f"field '{field}' must be between {low} and {high}, got {value}"
                )
            cleaned[field] = value
        elif field == "season":
            if not isinstance(value, str) or value not in config.ALLOWED_VALUES["season"]:
                raise RecordValidationError(
                    f"field 'season' must be one of "
                    f"{list(config.ALLOWED_VALUES['season'])}, got {value!r}"
                )
            cleaned[field] = value
        elif field == "cooperative_support":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RecordValidationError(
                    f"field 'cooperative_support' must be 0 or 1, got {value!r}"
                )
            if float(value) not in (0.0, 1.0):
                raise RecordValidationError(
                    f"field 'cooperative_support' must be 0 or 1, got {value!r}"
                )
            cleaned[field] = int(value)
    return cleaned


def load_artifacts(models_dir: Path | str = config.MODELS_DIR) -> dict:
    """Load saved pipelines and metadata from ``models/``."""
    models_dir = Path(models_dir)
    required = {
        "regression": models_dir / "regression_pipeline.joblib",
        "classification": models_dir / "classification_pipeline.joblib",
        "clustering": models_dir / "clustering_pipeline.joblib",
        "preprocessor": models_dir / "preprocessor.joblib",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing saved model(s): "
            + ", ".join(missing)
            + " - run 'python run_all.py' first"
        )

    artifacts = {name: joblib.load(path) for name, path in required.items()}
    metadata_path = models_dir / "model_metadata.json"
    artifacts["metadata"] = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    return artifacts


def predict_record(record: dict, artifacts: dict | None = None) -> dict:
    """Validate one record and return yield, risk, probability and cluster."""
    cleaned = validate_record(record)
    if artifacts is None:
        artifacts = load_artifacts()

    frame = pd.DataFrame([cleaned], columns=config.FEATURE_COLUMNS)

    yield_tons = float(artifacts["regression"].predict(frame)[0])
    clf = artifacts["classification"]
    high_risk = int(clf.predict(frame)[0])
    probability = None
    if hasattr(clf.named_steps["model"], "predict_proba"):
        proba = clf.predict_proba(frame)[0]
        classes = list(clf.named_steps["model"].classes_)
        if 1 in classes:
            probability = float(proba[classes.index(1)])
    cluster = int(artifacts["clustering"].predict(frame)[0])

    return {
        "yield_tons": round(max(yield_tons, 0.5), 4),
        "high_risk": high_risk,
        "risk_probability": round(probability, 4) if probability is not None else None,
        "cluster": cluster,
        "verification_code": config.VERIFICATION_CODE,
        "seed": config.SEED,
        "models": {
            "regression": artifacts["metadata"].get("regression_model", "unknown"),
            "classification": artifacts["metadata"].get("classification_model", "unknown"),
            "clustering": f"KMeans K={artifacts['metadata'].get('selected_k', '?')}",
        },
    }
