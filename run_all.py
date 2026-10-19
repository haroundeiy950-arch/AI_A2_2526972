"""Single reproducible entry point for the Musanze harvest decision system.

Usage
-----
    python run_all.py --student-id ug2526972@ines.ac.rw --seed 6972

This recreates the dataset, validates it, trains all three tasks and
regenerates every artifact and model file.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, generate_data, modeling, plots
from src.preprocess import build_preprocessor, split_features_targets
from src.validate import DataValidationError, validate_dataset, validate_seed, validate_student_id


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_fingerprint(path: Path, sha: str, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"dataset: {config.DATASET_NAME}",
                f"sha256: {sha}",
                f"rows_raw: {rows}",
                f"rows_clean: {config.BASE_ROWS}",
                f"seed: {config.SEED}",
                f"student_id: {config.INSTITUTIONAL_USERNAME}",
                f"verification_code: {config.VERIFICATION_CODE}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run(student_id: str, seed: int) -> dict:
    """Execute the whole pipeline and return a summary dictionary."""
    validate_seed(seed)
    validate_student_id(student_id)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Verification code : {config.VERIFICATION_CODE}")
    print(f"Student           : {config.FULL_NAME} ({config.REGISTRATION_NUMBER})")
    print(f"Student id        : {student_id}")
    print(f"Seed (LLLL)       : {seed}")

    # 1. recreate the dataset --------------------------------------------
    defects = generate_data.generate(seed=seed, path=config.DATA_PATH)
    raw = pd.read_csv(config.DATA_PATH)
    print(f"Dataset           : {config.DATA_PATH}")
    print(f"Rows              : {len(raw)} raw ({config.BASE_ROWS} base + 3 duplicates)")

    # 2. validate and clean ----------------------------------------------
    cleaned, validation = validate_dataset(raw, seed)
    validation["defects_injected"] = defects["missing_positions"] | {
        "duplicates_appended": defects["duplicates_appended"]
    }
    validation["csv_sha256"] = defects["sha256"]
    _write_json(config.VALIDATION_PATH, validation)
    _write_fingerprint(config.FINGERPRINT_PATH, defects["sha256"], len(raw))
    print(f"Missing values    : {validation['missing_values']}")
    print(f"Duplicates found  : {validation['duplicates']['duplicate_rows']}")
    print(f"SHA-256           : {defects['sha256']}")

    # 3. fixed train/test split ------------------------------------------
    (
        x_train,
        x_test,
        yreg_train,
        yreg_test,
        yclf_train,
        yclf_test,
        idx_train,
        idx_test,
    ) = split_features_targets(cleaned)
    print(f"Split             : {len(x_train)} train / {len(x_test)} test (stratified)")

    # 4. regression --------------------------------------------------------
    reg_metrics, reg_pipelines, reg_selected = modeling.train_regression(
        x_train, yreg_train, x_test, yreg_test
    )
    reg_metrics["dataset_sha256"] = defects["sha256"]
    _write_json(config.REGRESSION_METRICS_PATH, reg_metrics)
    joblib.dump(reg_pipelines[reg_selected], config.REGRESSION_MODEL_PATH)
    reg_scores = reg_metrics["all_models"][reg_selected]

    # 5. classification ----------------------------------------------------
    clf_metrics, clf_pipelines, clf_selected = modeling.train_classification(
        x_train, yclf_train, x_test, yclf_test
    )
    clf_metrics["dataset_sha256"] = defects["sha256"]
    _write_json(config.CLASSIFICATION_METRICS_PATH, clf_metrics)
    joblib.dump(clf_pipelines[clf_selected], config.CLASSIFICATION_MODEL_PATH)
    clf_scores = clf_metrics["all_models"][clf_selected]

    # 6. clustering --------------------------------------------------------
    x_all = cleaned[config.FEATURE_COLUMNS]
    clu_report, clu_pipeline, labels_all = modeling.train_clustering(x_train, x_all)
    clu_report["dataset_sha256"] = defects["sha256"]
    _write_json(config.CLUSTERING_REPORT_PATH, clu_report)
    joblib.dump(clu_pipeline, config.CLUSTERING_MODEL_PATH)

    # 7. shared preprocessor ----------------------------------------------
    preprocessor = build_preprocessor()
    preprocessor.fit(x_train)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    # 8. per-row test predictions ------------------------------------------
    reg_pred = reg_pipelines[reg_selected].predict(x_test)
    clf_pred = clf_pipelines[clf_selected].predict(x_test)
    clf_model = clf_pipelines[clf_selected].named_steps["model"]
    if hasattr(clf_model, "predict_proba"):
        proba = clf_pipelines[clf_selected].predict_proba(x_test)
        classes = list(clf_model.classes_)
        risk_prob = proba[:, classes.index(1)] if 1 in classes else np.full(len(x_test), np.nan)
    else:
        risk_prob = np.full(len(x_test), np.nan)

    test_clusters = clu_pipeline.predict(x_test)
    predictions = pd.DataFrame(
        {
            "row_id": np.asarray(idx_test, dtype=int),
            "actual_yield_tons": np.round(np.asarray(yreg_test, dtype=float), 4),
            "predicted_yield_tons": np.round(np.asarray(reg_pred, dtype=float), 4),
            "yield_error_tons": np.round(
                np.asarray(yreg_test, dtype=float) - np.asarray(reg_pred, dtype=float), 4
            ),
            "actual_high_risk": np.asarray(yclf_test, dtype=int),
            "predicted_high_risk": np.asarray(clf_pred, dtype=int),
            "risk_probability": np.round(np.asarray(risk_prob, dtype=float), 4),
            "cluster": np.asarray(test_clusters, dtype=int),
        }
    ).sort_values("row_id")
    config.PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(config.PREDICTIONS_PATH, index=False, float_format="%.4f")

    # 9. plots --------------------------------------------------------------
    plots.plot_residuals(yreg_test, reg_pred)
    plots.plot_confusion_matrix(
        clf_scores["confusion_matrix"], model_name=clf_selected
    )
    plots.plot_clusters(
        x_all,
        labels_all,
        clu_pipeline.named_steps["prep"],
        selected_k=clu_report["selected_k"],
    )

    # 10. model metadata ----------------------------------------------------
    metadata = {
        "verification_code": config.VERIFICATION_CODE,
        "seed": config.SEED,
        "full_name": config.FULL_NAME,
        "registration_number": config.REGISTRATION_NUMBER,
        "student_id": config.INSTITUTIONAL_USERNAME,
        "regression_model": reg_selected,
        "classification_model": clf_selected,
        "selected_k": clu_report["selected_k"],
        "silhouette": clu_report["k_scores"][str(clu_report["selected_k"])],
        "feature_columns": config.FEATURE_COLUMNS,
        "dataset_sha256": defects["sha256"],
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(config.MODEL_METADATA_PATH, metadata)

    summary = {
        "verification_code": config.VERIFICATION_CODE,
        "seed": seed,
        "sha256": defects["sha256"],
        "rows_raw": len(raw),
        "rows_clean": len(cleaned),
        "regression_selected": reg_selected,
        "regression_scores": reg_scores,
        "classification_selected": clf_selected,
        "classification_scores": clf_scores,
        "selected_k": clu_report["selected_k"],
        "silhouette": clu_report["k_scores"][str(clu_report["selected_k"])],
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recreate data, train all models and regenerate every artifact."
    )
    parser.add_argument(
        "--student-id",
        default=config.INSTITUTIONAL_USERNAME,
        help="institutional username, e.g. ug2526972@ines.ac.rw",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help=f"random seed LLLL (must be {config.SEED})",
    )
    args = parser.parse_args(argv)

    try:
        summary = run(args.student_id, args.seed)
    except DataValidationError as exc:
        print(f"VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2

    print("\n==================== FINAL SUMMARY ====================")
    print(f"Verification code   : {summary['verification_code']}")
    print(f"Dataset SHA-256     : {summary['sha256']}")
    print(f"Rows                : {summary['rows_raw']} raw / {summary['rows_clean']} clean")
    print(
        f"Regression          : {summary['regression_selected']} "
        f"(MAE {summary['regression_scores']['mae']}, "
        f"RMSE {summary['regression_scores']['rmse']}, "
        f"R2 {summary['regression_scores']['r2']})"
    )
    print(
        f"Classification      : {summary['classification_selected']} "
        f"(F1 {summary['classification_scores']['f1']}, "
        f"ROC-AUC {summary['classification_scores']['roc_auc']})"
    )
    print(
        f"Clustering          : K={summary['selected_k']} "
        f"(silhouette {summary['silhouette']})"
    )
    print("Artifacts written   :")
    for item in config.ARTIFACT_LIST:
        status = "ok" if (config.PROJECT_ROOT / item).exists() else "MISSING"
        print(f"  [{status}] {item}")
    print("=======================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
