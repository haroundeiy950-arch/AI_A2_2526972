"""Inference, CLI and API tests including failure paths."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import config
from src.inference import RecordValidationError, predict_record

ROOT = Path(__file__).resolve().parents[1]


def test_valid_inference(artifacts, valid_record) -> None:
    result = predict_record(valid_record, artifacts)
    assert result["verification_code"] == config.VERIFICATION_CODE
    assert result["yield_tons"] >= 0.5
    assert result["high_risk"] in (0, 1)
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["cluster"] >= 0


def test_missing_field_is_rejected(artifacts, valid_record) -> None:
    broken = dict(valid_record)
    broken.pop("fertilizer_kg")
    with pytest.raises(RecordValidationError, match="fertilizer_kg"):
        predict_record(broken, artifacts)


def test_wrong_type_is_rejected(artifacts, valid_record) -> None:
    broken = dict(valid_record, rainfall_mm="a lot of rain")
    with pytest.raises(RecordValidationError, match="rainfall_mm"):
        predict_record(broken, artifacts)


def test_range_error_is_rejected(artifacts, valid_record) -> None:
    with pytest.raises(RecordValidationError, match="600.0 and 1400.0"):
        predict_record(dict(valid_record, rainfall_mm=1401.0), artifacts)
    with pytest.raises(RecordValidationError, match="season"):
        predict_record(dict(valid_record, season="C"), artifacts)
    with pytest.raises(RecordValidationError, match="cooperative_support"):
        predict_record(dict(valid_record, cooperative_support=7), artifacts)


def test_cli_accepts_valid_and_rejects_invalid() -> None:
    good = subprocess.run(
        [sys.executable, "predict.py", "--json", "samples/valid.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert good.returncode == 0, good.stderr
    payload = json.loads(good.stdout)
    assert payload["verification_code"] == config.VERIFICATION_CODE

    bad = subprocess.run(
        [sys.executable, "predict.py", "--json", "samples/invalid.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1
    assert "ERROR" in bad.stderr


def test_health_endpoint() -> None:
    import app as app_module

    client = TestClient(app_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["verification_code"] == config.VERIFICATION_CODE
    assert body["model_loaded"] is True


def test_model_info_endpoint() -> None:
    import app as app_module

    client = TestClient(app_module.app)
    body = client.get("/model-info").json()
    assert body["verification_code"] == config.VERIFICATION_CODE
    assert body["dataset"]["synthetic"] is True
    assert body["features"] == config.FEATURE_COLUMNS
    assert "selection_reason" in body["regression"]


def test_api_predict_valid_and_invalid(valid_record) -> None:
    import app as app_module

    client = TestClient(app_module.app)
    ok = client.post("/predict", json=valid_record)
    assert ok.status_code == 200, ok.text
    assert ok.json()["verification_code"] == config.VERIFICATION_CODE

    bad_range = client.post("/predict", json={**valid_record, "rainfall_mm": 5000})
    assert bad_range.status_code == 422
    assert "detail" in bad_range.json()

    missing = client.post("/predict", json={k: v for k, v in valid_record.items() if k != "season"})
    assert missing.status_code == 422

    unknown = client.post("/predict", json={**valid_record, "sneaky_target": 1})
    assert unknown.status_code == 422


def test_classification_excludes_target_columns() -> None:
    classification_metrics = json.loads(
        config.CLASSIFICATION_METRICS_PATH.read_text(encoding="utf-8")
    )
    features = set(classification_metrics["features"])
    assert not features & set(config.TARGET_COLUMNS)
    assert all("yield" not in f for f in features)
