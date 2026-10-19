"""Data generation, validation and reproducibility tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, generate_data
from src.validate import (
    DataValidationError,
    clean,
    validate_dataset,
    validate_seed,
    validate_student_id,
)


def test_seed_is_rejected_when_not_llll() -> None:
    with pytest.raises(DataValidationError):
        validate_seed(1234)
    validate_seed(config.SEED)


def test_student_id_guard() -> None:
    validate_student_id(config.INSTITUTIONAL_USERNAME)
    with pytest.raises(DataValidationError):
        validate_student_id("someone.else@ines.ac.rw")


def test_row_counts_and_defects() -> None:
    frame, defects = generate_data.build_frame(config.SEED)
    assert len(frame) == config.BASE_ROWS + config.N_DUPLICATE_ROWS
    assert defects["base_rows"] == config.BASE_ROWS
    assert int(frame["soil_moisture_pct"].isna().sum()) == config.MISSING_PER_COLUMN
    assert int(frame["fertilizer_kg"].isna().sum()) == config.MISSING_PER_COLUMN
    assert int(frame.duplicated().sum()) == config.N_DUPLICATE_ROWS


def test_targets_follow_the_formula() -> None:
    base = generate_data.build_frame(config.SEED)[0].iloc[: config.BASE_ROWS]
    assert (base["yield_tons"] >= generate_data.YIELD_FLOOR).all()
    median = (base["yield_tons"] / base["field_size_ha"]).median()
    expected = ((base["yield_tons"] / base["field_size_ha"]) < median).astype(int)
    assert (expected == base["high_risk"]).all()
    assert set(base["high_risk"].unique()) == {0, 1}


def test_validation_passes_and_cleaning_removes_duplicates() -> None:
    frame, _ = generate_data.build_frame(config.SEED)
    cleaned, report = validate_dataset(frame, config.SEED)
    assert report["status"] == "passed"
    assert len(cleaned) == config.BASE_ROWS
    assert int(cleaned.duplicated().sum()) == 0
    assert report["rows_raw"] == config.BASE_ROWS + config.N_DUPLICATE_ROWS


def test_range_violation_is_detected() -> None:
    frame, _ = generate_data.build_frame(config.SEED)
    frame.loc[0, "rainfall_mm"] = 5000.0
    with pytest.raises(DataValidationError, match="rainfall_mm"):
        validate_dataset(frame, config.SEED)


def test_datasets_are_reproducible(tmp_path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    generate_data.generate(config.SEED, first)
    generate_data.generate(config.SEED, second)
    assert generate_data.sha256_of(first) == generate_data.sha256_of(second)

    # and the committed dataset matches the same fingerprint
    assert generate_data.sha256_of(first) == generate_data.sha256_of(config.DATA_PATH)


def test_csv_uses_four_decimals() -> None:
    text = config.DATA_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert text.split(",")[0] == "rainfall_mm"
    sample_line = config.DATA_PATH.read_text(encoding="utf-8").splitlines()[1]
    rainfall = sample_line.split(",")[0]
    assert len(rainfall.split(".")[-1]) == 4
