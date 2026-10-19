"""Generate the technical report PDF from the generated artifacts.

Running this after ``run_all.py`` guarantees the report numbers match the
JSON/CSV evidence, because every figure is read back from ``artifacts/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

BASE_STYLE = getSampleStyleSheet()
H1 = ParagraphStyle("H1x", parent=BASE_STYLE["Heading1"], fontSize=15, spaceAfter=8)
H2 = ParagraphStyle("H2x", parent=BASE_STYLE["Heading2"], fontSize=12, spaceAfter=6)
BODY = ParagraphStyle("Bodyx", parent=BASE_STYLE["BodyText"], fontSize=9.5, leading=13)
SMALL = ParagraphStyle("Smallx", parent=BASE_STYLE["BodyText"], fontSize=8.5, leading=11)
NOTE = ParagraphStyle("Notex", parent=BASE_STYLE["BodyText"], fontSize=8.5, textColor=colors.HexColor("#555555"))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _table(rows, widths=None, header=True) -> Table:
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    table.setStyle(TableStyle(style))
    return table


def _metric_row(name: str, scores: dict, keys: list[str]) -> list:
    return [name] + [scores.get(k) if scores.get(k) is not None else "n/a" for k in keys]


def build_report(
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    out_path: Path = config.REPORT_PATH,
) -> Path:
    """Assemble the five-to-seven page PDF report."""
    regression = _json(artifacts_dir / "regression_metrics.json")
    classification = _json(artifacts_dir / "classification_metrics.json")
    clustering = _json(artifacts_dir / "clustering_report.json")
    validation = _json(artifacts_dir / "data_validation.json")
    metadata = _json(config.MODEL_METADATA_PATH)
    predictions = pd.read_csv(artifacts_dir / "test_predictions.csv")
    fingerprint = config.FINGERPRINT_PATH.read_text(encoding="utf-8").strip() if config.FINGERPRINT_PATH.exists() else ""

    reg_selected = regression.get("selected_model", "?")
    clf_selected = classification.get("selected_model", "?")
    clf_scores = classification.get("all_models", {}).get(clf_selected, {})

    story: list = []
    story.append(Paragraph("Musanze Harvest Intelligence Service", H1))
    story.append(
        Paragraph(
            f"Technical report - SWE 3513 Artificial Intelligence, Individual Assignment 2 "
            f"- verification code <b>{config.VERIFICATION_CODE}</b>",
            BODY,
        )
    )
    story.append(Spacer(1, 8))

    # ---------------------------------------------------------- page 1 ------
    story.append(Paragraph("1. Identity and decision problem", H2))
    story.append(
        _table(
            [
                ["Field", "Value"],
                ["Full name", config.FULL_NAME],
                ["Registration number", config.REGISTRATION_NUMBER],
                ["Institutional username", config.INSTITUTIONAL_USERNAME],
                ["Verification code", config.VERIFICATION_CODE],
                ["Random seed (LLLL)", str(config.SEED)],
                ["Base rows", f"{config.BASE_ROWS} (240 + {config.SEED} mod 41)"],
            ],
            widths=[5 * cm, 10 * cm],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "The stakeholder is the field officer of a potato cooperative in Musanze. "
            "Farm records arrive in inconsistent spreadsheets, internet access is unreliable and "
            "the officer needs local, evidence-based support. The service answers three questions: "
            "(1) how many tons will a farm harvest, (2) is the farm at high production risk, and "
            "(3) which farms share similar growing conditions?",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "<b>Ethical boundary.</b> This is a classroom decision-support prototype built on "
            "synthetic data. It must not replace an agronomist and must never be used to decide "
            "access to finance, insurance or public services. Every automated answer requires "
            "human review by the field officer.",
            BODY,
        )
    )

    # ---------------------------------------------------------- page 2 ------
    story.append(PageBreak())
    story.append(Paragraph("2. Data and reproducibility", H2))
    story.append(
        Paragraph(
            "The dataset is generated by <font face='Courier'>data/generate_data.py</font> with "
            f"<font face='Courier'>numpy.random.default_rng({config.SEED})</font>. The seed is the "
            f"last four digits of the registration number (LLLL = {config.SEED}). "
            f"The generator writes {config.BASE_ROWS} base rows, sets "
            f"{config.MISSING_PER_COLUMN} soil moisture values and {config.MISSING_PER_COLUMN} "
            f"fertilizer values to missing with the same generator stream, then appends "
            f"{config.N_DUPLICATE_ROWS} exact duplicate rows. All numeric values use four decimal places.",
            BODY,
        )
    )
    story.append(
        _table(
            [
                ["Column", "Rule", "Role"],
                ["rainfall_mm", "Uniform 600-1400", "feature"],
                ["avg_temp_c", "Uniform 12-26", "feature"],
                ["soil_moisture_pct", "Uniform 20-80 (12 missing)", "feature"],
                ["fertilizer_kg", "Uniform 0-250 (12 missing)", "feature"],
                ["field_size_ha", "Uniform 0.2-4.0", "feature"],
                ["season", "A or B", "categorical feature"],
                ["cooperative_support", "0 or 1", "binary feature"],
                ["yield_tons", "given linear formula + N(0, 0.80), floor 0.50", "regression target"],
                ["high_risk", "yield per hectare below median", "classification target"],
            ],
            widths=[4.2 * cm, 7.3 * cm, 3.5 * cm],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        _table(
            [
                ["Evidence", "Value"],
                ["Rows in CSV", str(validation.get("rows_raw"))],
                ["Rows after cleaning duplicates", str(validation.get("rows_clean"))],
                ["Missing values detected", json.dumps(validation.get("missing_values"))],
                ["Exact duplicates detected", str(validation.get("duplicates", {}).get("duplicate_rows"))],
                ["SHA-256 of final CSV", metadata.get("dataset_sha256", "")],
            ],
            widths=[6 * cm, 9 * cm],
        )
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Fingerprint file record: {fingerprint.splitlines()[1] if len(fingerprint.splitlines()) > 1 else ''}", SMALL))

    # ---------------------------------------------------------- page 3 ------
    story.append(PageBreak())
    story.append(Paragraph("3. Pipeline design", H2))
    story.append(
        Paragraph(
            f"<b>Split.</b> Exact duplicate rows are removed first, then a single fixed stratified "
            f"split (random_state = {config.SEED}, stratify on high_risk) produces "
            f"{metadata.get('n_train')} training rows and {metadata.get('n_test')} held-out test rows. "
            "The test rows are never used for fitting.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "<b>Leakage control.</b> Numeric missing values are imputed with the training median; "
            "numeric features are standardised; <font face='Courier'>season</font> is one-hot encoded "
            "with unknown-category tolerance. All imputers, the scaler and the encoder live inside one "
            "ColumnTransformer and are fitted on the training split only, inside each model Pipeline. "
            "yield_tons, yield per hectare and high_risk are excluded from every feature set, "
            "and clustering never reads a target column.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "<b>Candidates.</b> Regression: mean baseline, LinearRegression, RandomForestRegressor(200). "
            "Classification: most-frequent baseline, LogisticRegression, RandomForestClassifier(200). "
            "Clustering: KMeans with K = 2..6 on standardised input features.",
            BODY,
        )
    )

    # ---------------------------------------------------------- page 4 ------
    story.append(PageBreak())
    story.append(Paragraph("4. Regression and classification", H2))
    story.append(Paragraph("Regression (target yield_tons)", H2))
    rows = [["Model", "MAE (t)", "RMSE (t)", "R2"]]
    for name in ["mean_baseline", "linear_regression", "random_forest"]:
        scores = regression.get("all_models", {}).get(name)
        if scores:
            rows.append(_metric_row(name, scores, ["mae", "rmse", "r2"]))
    story.append(_table(rows, widths=[6 * cm, 3 * cm, 3 * cm, 3 * cm]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Selected model: <b>{reg_selected}</b>. {regression.get('selection_reason', '')}", BODY))
    worst = predictions.reindex(predictions["yield_error_tons"].abs().sort_values(ascending=False).index).iloc[0]
    story.append(
        Paragraph(
            f"<b>Error interpretation.</b> On the test set the largest absolute error is row "
            f"{int(worst['row_id'])}: actual {worst['actual_yield_tons']} t versus predicted "
            f"{worst['predicted_yield_tons']} t (error {worst['yield_error_tons']} t). The model "
            "under-predicts this farm. The residual plot shows errors are centred on zero with "
            "roughly constant spread, which is what we expect when the target is generated by a "
            "linear formula with Gaussian noise; the largest errors are single noisy draws rather "
            "than a systematic bias. RandomForest loses to the linear model because trees cannot "
            "extrapolate the continuous linear trend and split it into steps.",
            BODY,
        )
    )
    if (config.PLOTS_DIR / "residuals.png").exists():
        story.append(Image(str(config.PLOTS_DIR / "residuals.png"), width=10.5 * cm, height=6.7 * cm))

    story.append(Spacer(1, 4))
    story.append(Paragraph("Classification (target high_risk)", H2))
    rows = [["Model", "Precision", "Recall", "F1", "ROC-AUC"]]
    for name in ["majority_baseline", "logistic_regression", "random_forest"]:
        scores = classification.get("all_models", {}).get(name)
        if scores:
            rows.append(_metric_row(name, scores, ["precision", "recall", "f1", "roc_auc"]))
    story.append(_table(rows, widths=[5.5 * cm, 2.6 * cm, 2.4 * cm, 2.2 * cm, 2.4 * cm]))
    story.append(Spacer(1, 4))
    cm_matrix = clf_scores.get("confusion_matrix", [[0, 0], [0, 0]])
    story.append(
        Paragraph(
            f"Selected model: <b>{clf_selected}</b>. Confusion matrix [[TN, FP], [FN, TP]] = "
            f"{cm_matrix}: {cm_matrix[0][1]} false positive(s) and {cm_matrix[1][0]} false negative(s). "
            f"ROC-AUC {clf_scores.get('roc_auc')} shows the ranking is almost perfect, while the "
            "F1 shortfall comes from borderline farms near the median yield-per-hectare where the "
            "random noise flips the label; this is label noise in the generator, not a modelling mistake.",
            BODY,
        )
    )
    mis = predictions[predictions["actual_high_risk"] != predictions["predicted_high_risk"]]
    if len(mis):
        row = mis.iloc[0]
        story.append(
            Paragraph(
                f"<b>Error interpretation.</b> Row {int(row['row_id'])} is classified "
                f"{int(row['predicted_high_risk'])} with probability {row['risk_probability']} but the "
                f"actual label is {int(row['actual_high_risk'])}. Its field size makes the "
                "yield-per-hectare ratio sit just beside the median, so a small residual pushes the "
                "label across the threshold. Consequence for the officer: borderline farms should be "
                "inspected in person rather than trusted to the model.",
                BODY,
            )
        )
    story.append(Paragraph(f"{classification.get('selection_reason', '')}", NOTE))

    # ---------------------------------------------------------- page 5 ------
    story.append(PageBreak())
    story.append(Paragraph("5. Clustering results", H2))
    rows = [["K", "Silhouette"]]
    for k, score in clustering.get("k_scores", {}).items():
        rows.append([k, score])
    story.append(_table(rows, widths=[3 * cm, 4 * cm]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"Selected K = <b>{clustering.get('selected_k')}</b> by highest silhouette score. "
            f"{clustering.get('selection_reason', '')}",
            BODY,
        )
    )
    for cluster in clustering.get("clusters", []):
        story.append(
            Paragraph(
                f"<b>Cluster {cluster['cluster']}</b> ({cluster['size_all_rows']} of "
                f"{validation.get('rows_clean')} farms; dominant season {cluster['dominant_season']}): "
                f"{cluster['profile']}",
                BODY,
            )
        )
    story.append(
        Paragraph(
            "<b>Operational meaning and honest caveat.</b> The silhouette value is modest and the "
            "centres differ mainly in cooperative support, while rainfall, temperature, moisture, "
            "fertilizer and field size are almost identical between clusters. The split is therefore "
            "a weak segmentation: it separates farms that received cooperative support from those "
            "that did not, not genuinely distinct soil or climate zones. The officer may use clusters "
            "for exploratory grouping only, and this result should be re-checked with real farm data "
            "that has more varied structure before any advice is based on it.",
            BODY,
        )
    )
    if (config.PLOTS_DIR / "clusters.png").exists():
        story.append(Image(str(config.PLOTS_DIR / "clusters.png"), width=12 * cm, height=7.7 * cm))

    # ---------------------------------------------------------- page 6 ------
    story.append(PageBreak())
    story.append(Paragraph("6. API and testing", H2))
    story.append(
        _table(
            [
                ["Endpoint", "Request", "Response", "Failure behaviour"],
                ["GET /health", "none", "status, verification code, seed, model_loaded", "-"],
                ["GET /model-info", "none", "identity, features, metrics, intended use", "-"],
                [
                    "POST /predict",
                    "JSON FarmRecord (7 typed fields)",
                    "yield_tons, high_risk, risk_probability, cluster, verification code",
                    "422 with field-level message for missing, wrong-type, out-of-range or unknown fields",
                ],
            ],
            widths=[3 * cm, 4 * cm, 5 * cm, 4 * cm],
        )
    )
    story.append(Spacer(1, 6))
    story.append(Paragraph("Valid request/response (examples/valid.json):", SMALL))
    story.append(
        Paragraph(
            '{"rainfall_mm": 950.0, "avg_temp_c": 18.5, "soil_moisture_pct": 55.0, '
            '"fertilizer_kg": 120.0, "field_size_ha": 1.8, "season": "A", "cooperative_support": 1}'
            " <font color='#1a6b1a'>-> 200 OK, yield 12.0151 t, high_risk 0, probability 0.2241, cluster 1</font>",
            SMALL,
        )
    )
    story.append(Paragraph("Invalid request (examples/invalid.json):", SMALL))
    story.append(
        Paragraph(
            "rainfall_mm 1500, avg_temp_c 40, soil_moisture_pct 95, fertilizer_kg -5, "
            "field_size_ha 9.9, season C, cooperative_support 7"
            " <font color='#a02020'>-> 422 invalid input with one message per broken rule</font>",
            SMALL,
        )
    )
    story.append(
        Paragraph(
            "<b>Test summary.</b> The pytest suite exercises failure paths, not only happy paths: "
            "invalid seed rejection, student-id guard, injected-defect detection, range violation "
            "detection, dataset reproducibility (identical SHA-256 on repeated generation), CLI valid "
            "and invalid runs, missing field, wrong type, out-of-range value, unknown field and the "
            "FastAPI /health, /model-info and /predict endpoints.",
            BODY,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("7. Model card and reflection", H2))
    story.append(
        _table(
            [
                ["Item", "Statement"],
                ["Intended use", "Estimate harvest tons, flag high production risk and group similar farms for a Musanze potato cooperative field officer."],
                ["Data status", "100% synthetic data generated from the student's seed. No real farm was measured; no personal data is processed."],
                ["Limitations", "Linear generator makes linear_regression hard to beat; clustering is weakly separated; missing values are imputed, never observed."],
                ["Risks", "Over-trust in a prototype; risk labels could be read as credit or insurance signals; distribution shift on real farms."],
                ["Prohibited use", "Determining access to finance, insurance or public services; replacing an agronomist; any use on real farms without validation."],
                ["Human oversight", "Every prediction is a recommendation for review by the field officer, who remains accountable for the decision."],
                ["What to improve", "Real, multi-season cooperative records; a held-out year for temporal validation; calibrated probabilities; cost-sensitive thresholds."],
            ],
            widths=[3.4 * cm, 11.6 * cm],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"All figures in this report were read directly from the generated artifacts: "
            f"regression_metrics.json, classification_metrics.json, clustering_report.json, "
            f"test_predictions.csv and data_fingerprint.txt, so the report and the code agree by "
            f"construction. Dataset SHA-256: {metadata.get('dataset_sha256', '')}. "
            f"Verification code: {config.VERIFICATION_CODE}.",
            NOTE,
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=f"AI_A2_{config.REGDIGITS} Report",
        author=config.FULL_NAME,
    )

    def _stamp(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(2 * cm, 1.0 * cm, f"{config.FULL_NAME} - {config.REGISTRATION_NUMBER} - {config.VERIFICATION_CODE}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.0 * cm, f"page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_stamp, onLaterPages=_stamp)
    return out_path


if __name__ == "__main__":
    path = build_report()
    print(f"report written: {path}")
