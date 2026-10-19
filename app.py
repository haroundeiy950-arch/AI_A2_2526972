"""FastAPI service for the Musanze harvest decision system.

Run locally:
    uvicorn app:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health      liveness and identity check
    GET  /model-info  selected models, metrics and verification code
    POST /predict     validated single-farm prediction
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src import config
from src.inference import RecordValidationError, load_artifacts, predict_record


class FarmRecord(BaseModel):
    """One farm record accepted by POST /predict."""

    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(ge=600.0, le=1400.0, description="Annual rainfall in mm")
    avg_temp_c: float = Field(ge=12.0, le=26.0, description="Average temperature in Celsius")
    soil_moisture_pct: float = Field(ge=20.0, le=80.0, description="Soil moisture percentage")
    fertilizer_kg: float = Field(ge=0.0, le=250.0, description="Fertilizer applied in kg")
    field_size_ha: float = Field(ge=0.2, le=4.0, description="Field size in hectares")
    season: Literal["A", "B"] = Field(description="Growing season A or B")
    cooperative_support: Literal[0, 1] = Field(
        description="1 if the cooperative provided support, else 0"
    )


class PredictionResponse(BaseModel):
    yield_tons: float
    high_risk: int
    risk_probability: float | None
    cluster: int
    verification_code: str
    seed: int
    models: dict


app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    version="1.0.0",
    description=(
        "Classroom decision-support prototype for a Musanze potato cooperative. "
        f"Verification code {config.VERIFICATION_CODE}. Synthetic training data; "
        "must not replace an agronomist or determine access to finance or insurance."
    ),
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Turn pydantic errors into short, readable 4xx messages."""
    details = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err["loc"] if loc != "body")
        details.append(f"{field}: {err['msg']}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "invalid input",
            "verification_code": config.VERIFICATION_CODE,
            "detail": details or ["request body is invalid"],
        },
    )


@lru_cache(maxsize=1)
def _artifacts() -> dict:
    return load_artifacts(config.MODELS_DIR)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@app.get("/health")
def health() -> dict:
    """Liveness probe: confirms identity and that models can be loaded."""
    try:
        _artifacts()
        loaded = True
        error = None
    except FileNotFoundError as exc:
        loaded = False
        error = str(exc)
    return {
        "status": "ok" if loaded else "models missing",
        "verification_code": config.VERIFICATION_CODE,
        "seed": config.SEED,
        "student": config.FULL_NAME,
        "registration_number": config.REGISTRATION_NUMBER,
        "model_loaded": loaded,
        "detail": error,
    }


@app.get("/model-info")
def model_info() -> dict:
    """Model card style information plus the measured metrics."""
    metadata = _read_json(config.MODEL_METADATA_PATH)
    regression = _read_json(config.REGRESSION_METRICS_PATH)
    classification = _read_json(config.CLASSIFICATION_METRICS_PATH)
    clustering = _read_json(config.CLUSTERING_REPORT_PATH)
    return {
        "verification_code": config.VERIFICATION_CODE,
        "seed": config.SEED,
        "student": {
            "name": config.FULL_NAME,
            "registration_number": config.REGISTRATION_NUMBER,
            "username": config.INSTITUTIONAL_USERNAME,
        },
        "dataset": {
            "name": config.DATASET_NAME,
            "synthetic": True,
            "sha256": metadata.get("dataset_sha256"),
            "rows_train": metadata.get("n_train"),
            "rows_test": metadata.get("n_test"),
        },
        "features": config.FEATURE_COLUMNS,
        "regression": {
            "selected_model": regression.get("selected_model"),
            "scores": regression.get("all_models", {}).get(
                regression.get("selected_model", ""), {}
            ),
            "selection_reason": regression.get("selection_reason"),
        },
        "classification": {
            "selected_model": classification.get("selected_model"),
            "scores": classification.get("all_models", {}).get(
                classification.get("selected_model", ""), {}
            ),
            "selection_reason": classification.get("selection_reason"),
        },
        "clustering": {
            "selected_k": clustering.get("selected_k"),
            "k_scores": clustering.get("k_scores"),
            "selection_reason": clustering.get("selection_reason"),
        },
        "intended_use": (
            "Classroom decision-support prototype for estimating potato harvest, "
            "flagging production risk and grouping similar farms. Not a replacement "
            "for an agronomist; must not be used to decide access to finance, "
            "insurance or public services. Human review required."
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(record: FarmRecord):
    """Predict yield estimate, high-risk flag, probability and cluster."""
    try:
        result = predict_record(record.model_dump(), _artifacts())
    except RecordValidationError as exc:
        return JSONResponse(  # pragma: no cover - pydantic already blocks these
            status_code=422,
            content={"error": "invalid input", "detail": str(exc)},
        )
    return PredictionResponse(**result)


@app.get("/")
def root() -> dict:
    return {
        "service": "Musanze Harvest Intelligence Service",
        "verification_code": config.VERIFICATION_CODE,
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/predict"],
    }
