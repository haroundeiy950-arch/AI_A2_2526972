"""Project configuration: student identity, seed and dataset rules."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- identity ---
FULL_NAME = "Haroun Mohammed Arbab Yousef"
REGISTRATION_NUMBER = "25/26972"
REGDIGITS = REGISTRATION_NUMBER.replace("/", "")
INSTITUTIONAL_USERNAME = f"ug{REGDIGITS}@ines.ac.rw"

SEED = 6972
VERIFICATION_CODE = f"AI-A2-{SEED}"
ZIP_NAME = f"AI_A2_{REGDIGITS}.zip"
PACKAGE_NAME = f"AI_A2_{REGDIGITS}"

# ------------------------------------------------------------ data rules -----
BASE_ROWS = 240 + (SEED % 41)          # 6972 % 41 = 2  -> 242 base rows
MISSING_PER_COLUMN = round(0.05 * BASE_ROWS)  # round(12.1) = 12
N_DUPLICATE_ROWS = 3
TEST_SIZE = 0.2

DATASET_NAME = f"{PACKAGE_NAME}.csv"
DATA_PATH = PROJECT_ROOT / "data" / DATASET_NAME
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
SAMPLES_DIR = PROJECT_ROOT / "samples"
DOCS_DIR = PROJECT_ROOT / "docs"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"

REPORT_PATH = DOCS_DIR / f"{PACKAGE_NAME}_REPORT.pdf"

# ------------------------------------------------------------- schema --------
FEATURE_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]
TARGET_REGRESSION = "yield_tons"
TARGET_CLASSIFICATION = "high_risk"
TARGET_COLUMNS = [TARGET_REGRESSION, TARGET_CLASSIFICATION]
REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]
CATEGORICAL_FEATURES = ["season"]

# column -> (minimum, maximum) allowed by the assignment specification
RANGES: dict[str, tuple[float, float]] = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
ALLOWED_VALUES: dict[str, tuple] = {
    "season": ("A", "B"),
    "cooperative_support": (0, 1),
    "high_risk": (0, 1),
}
# columns that may legitimately contain missing values (deliberate defects)
NULLABLE_COLUMNS = ["soil_moisture_pct", "fertilizer_kg"]

# --------------------------------------------------------------- paths -------
GENERATE_SCRIPT = PROJECT_ROOT / "data" / "generate_data.py"
FINGERPRINT_PATH = ARTIFACTS_DIR / "data_fingerprint.txt"
VALIDATION_PATH = ARTIFACTS_DIR / "data_validation.json"
REGRESSION_METRICS_PATH = ARTIFACTS_DIR / "regression_metrics.json"
CLASSIFICATION_METRICS_PATH = ARTIFACTS_DIR / "classification_metrics.json"
CLUSTERING_REPORT_PATH = ARTIFACTS_DIR / "clustering_report.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
REGRESSION_MODEL_PATH = MODELS_DIR / "regression_pipeline.joblib"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_pipeline.joblib"
CLUSTERING_MODEL_PATH = MODELS_DIR / "clustering_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

ARTIFACT_LIST = [
    "data/AI_A2_2526972.csv",
    "artifacts/data_fingerprint.txt",
    "artifacts/data_validation.json",
    "artifacts/regression_metrics.json",
    "artifacts/classification_metrics.json",
    "artifacts/clustering_report.json",
    "artifacts/test_predictions.csv",
    "artifacts/plots/residuals.png",
    "artifacts/plots/confusion_matrix.png",
    "artifacts/plots/clusters.png",
    "models/preprocessor.joblib",
    "models/regression_pipeline.joblib",
    "models/classification_pipeline.joblib",
    "models/clustering_pipeline.joblib",
    "models/model_metadata.json",
]
