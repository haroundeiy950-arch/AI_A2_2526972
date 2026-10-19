"""Dataset and seed validation.

Every check returns a structured report; fatal problems raise
``DataValidationError`` with an actionable message.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


class DataValidationError(Exception):
    """Raised when the dataset or seed violates the assignment contract."""


def validate_seed(seed: int) -> None:
    """Reject any seed that is not LLLL (6972)."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed != config.SEED:
        raise DataValidationError(
            f"Invalid seed {seed!r}. This submission is signed with LLLL = "
            f"{config.SEED}; re-run with --seed {config.SEED}."
        )


def validate_student_id(student_id: str) -> None:
    """Reject a student id that does not belong to this submission."""
    expected = config.INSTITUTIONAL_USERNAME
    if student_id != expected and config.REGDIGITS not in student_id:
        raise DataValidationError(
            f"Invalid student id {student_id!r}. This submission belongs to "
            f"{expected} (registration {config.REGISTRATION_NUMBER})."
        )


def validate_schema(frame: pd.DataFrame) -> list[str]:
    """Check required columns, value ranges, categories and leakage."""
    problems: list[str] = []

    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise DataValidationError(f"missing required columns: {missing_cols}")

    extra = [c for c in frame.columns if c not in config.REQUIRED_COLUMNS]
    if extra:
        problems.append(f"unexpected columns: {extra}")

    for column, (low, high) in config.RANGES.items():
        values = frame[column].dropna()
        if values.empty:
            problems.append(f"{column}: column is entirely empty")
            continue
        bad = values[(values < low) | (values > high)]
        if len(bad):
            problems.append(
                f"{column}: {len(bad)} value(s) outside [{low}, {high}] "
                f"(e.g. {float(bad.iloc[0]):.4f})"
            )
        if values.isna().any():
            problems.append(f"{column}: unexpected missing values")

    for column, allowed in config.ALLOWED_VALUES.items():
        values = frame[column].dropna()
        bad = sorted(set(values.unique()) - set(allowed), key=str)
        if bad:
            problems.append(f"{column}: forbidden value(s) {bad}")

    for column in frame.columns:
        if column in config.NULLABLE_COLUMNS:
            continue
        nulls = int(frame[column].isna().sum())
        if nulls:
            problems.append(f"{column}: {nulls} missing value(s) but column must be complete")

    leakage = [c for c in config.FEATURE_COLUMNS if c in config.TARGET_COLUMNS]
    leakage += [
        c
        for c in frame.columns
        if "yield_per" in c or c.endswith("_per_ha") or c == "target"
    ]
    if leakage:
        problems.append(f"target leakage in feature set: {leakage}")

    if problems:
        raise DataValidationError("dataset validation failed -> " + "; ".join(problems))
    return problems


def missing_report(frame: pd.DataFrame) -> dict:
    """Count missing values per column."""
    counts = {c: int(frame[c].isna().sum()) for c in frame.columns}
    return {k: v for k, v in counts.items() if v} or {"none": 0}


def duplicate_report(frame: pd.DataFrame) -> dict:
    """Count exact duplicate rows."""
    mask = frame.duplicated(keep="first")
    return {"duplicate_rows": int(mask.sum()), "row_positions": [int(i) for i in frame.index[mask]]}


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (keep first occurrence). Missing values are
    intentionally left in place so that imputation happens on train data only."""
    return frame.drop_duplicates(keep="first").copy()


def validate_dataset(
    frame: pd.DataFrame, seed: int = config.SEED
) -> tuple[pd.DataFrame, dict]:
    """Validate the raw CSV, then clean it. Returns (clean frame, report)."""
    validate_seed(seed)

    report: dict = {
        "seed": seed,
        "verification_code": config.VERIFICATION_CODE,
        "rows_raw": int(len(frame)),
        "columns": list(frame.columns),
    }
    validate_schema(frame)

    report["missing_values"] = missing_report(frame)
    report["duplicates"] = duplicate_report(frame)
    report["expected_missing"] = {
        "soil_moisture_pct": config.MISSING_PER_COLUMN,
        "fertilizer_kg": config.MISSING_PER_COLUMN,
    }
    report["expected_duplicates"] = config.N_DUPLICATE_ROWS
    report["range_limits"] = {k: list(v) for k, v in config.RANGES.items()}
    report["allowed_values"] = {k: list(v) for k, v in config.ALLOWED_VALUES.items()}

    cleaned = clean(frame)
    report["rows_clean"] = int(len(cleaned))
    report["target_leakage_check"] = {
        "classification_features": config.FEATURE_COLUMNS,
        "excluded_target_columns": config.TARGET_COLUMNS,
        "passed": True,
    }
    report["status"] = "passed"
    return cleaned, report


def load_and_validate(path: Path | str = config.DATA_PATH, seed: int = config.SEED):
    """Read a CSV from disk and run the full validation."""
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"dataset not found: {path} (run data/generate_data.py first)")
    frame = pd.read_csv(path)
    return validate_dataset(frame, seed)
