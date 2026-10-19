"""Synthetic dataset generation for the Musanze harvest decision system.

The dataset is fully reproducible: every random draw comes from a single
``numpy.random.default_rng`` stream seeded with the student's LLLL value.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

SEASON_EFFECT = {"A": 0.50, "B": -0.20}
NOISE_MEAN = 0.0
NOISE_STD = 0.80
YIELD_FLOOR = 0.50


def build_frame(seed: int = config.SEED) -> tuple[pd.DataFrame, dict]:
    """Create the raw dataset including deliberate missing values and duplicates.

    Returns the full frame (245 rows = 242 base + 3 duplicates) together with
    a dictionary describing the injected defects.
    """
    if seed != config.SEED:
        raise ValueError(
            f"Seed {seed} is not allowed. This project requires seed "
            f"{config.SEED} (LLLL of the registration number)."
        )

    n = config.BASE_ROWS
    rng = np.random.default_rng(seed)

    rainfall_mm = rng.uniform(600.0, 1400.0, n)
    avg_temp_c = rng.uniform(12.0, 26.0, n)
    soil_moisture_pct = rng.uniform(20.0, 80.0, n)
    fertilizer_kg = rng.uniform(0.0, 250.0, n)
    field_size_ha = rng.uniform(0.2, 4.0, n)
    season = rng.choice(np.array(["A", "B"]), n)
    cooperative_support = rng.integers(0, 2, n).astype(int)
    noise = rng.normal(NOISE_MEAN, NOISE_STD, n)

    yield_tons = (
        0.004 * rainfall_mm
        + 0.06 * soil_moisture_pct
        + 0.02 * fertilizer_kg
        + 1.30 * field_size_ha
        - 0.25 * np.abs(avg_temp_c - 18.0)
        + 0.70 * cooperative_support
        + np.where(season == "A", SEASON_EFFECT["A"], SEASON_EFFECT["B"])
        + noise
    )
    yield_tons = np.maximum(yield_tons, YIELD_FLOOR)

    yield_per_ha = yield_tons / field_size_ha
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)

    base = pd.DataFrame(
        {
            "rainfall_mm": rainfall_mm,
            "avg_temp_c": avg_temp_c,
            "soil_moisture_pct": soil_moisture_pct,
            "fertilizer_kg": fertilizer_kg,
            "field_size_ha": field_size_ha,
            "season": season,
            "cooperative_support": cooperative_support,
            "yield_tons": yield_tons,
            "high_risk": high_risk,
        }
    )

    m = config.MISSING_PER_COLUMN
    soil_positions = rng.choice(n, size=m, replace=False)
    remaining = np.setdiff1d(np.arange(n), soil_positions)
    fertilizer_positions = rng.choice(remaining, size=m, replace=False)

    defects = {
        "base_rows": int(n),
        "missing_positions": {
            "soil_moisture_pct": sorted(int(i) for i in soil_positions),
            "fertilizer_kg": sorted(int(i) for i in fertilizer_positions),
        },
        "missing_counts": {"soil_moisture_pct": m, "fertilizer_kg": m},
        "duplicates_appended": config.N_DUPLICATE_ROWS,
    }

    base.iloc[soil_positions, base.columns.get_loc("soil_moisture_pct")] = np.nan
    base.iloc[fertilizer_positions, base.columns.get_loc("fertilizer_kg")] = np.nan

    duplicates = base.iloc[: config.N_DUPLICATE_ROWS].copy()
    return pd.concat([base, duplicates], ignore_index=True), defects


def write_csv(frame: pd.DataFrame, path: Path | str = config.DATA_PATH) -> Path:
    """Write the dataset with four decimal places as required."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.4f")
    return path


def sha256_of(path: Path | str) -> str:
    """Return the SHA-256 hex digest of the raw bytes of ``path``."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(seed: int = config.SEED, path: Path | str = config.DATA_PATH) -> dict:
    """Generate, write and fingerprint the dataset. Returns defect metadata."""
    frame, defects = build_frame(seed)
    path = write_csv(frame, path)
    defects["csv_path"] = str(path)
    defects["rows_written"] = len(frame)
    defects["sha256"] = sha256_of(path)
    return defects
