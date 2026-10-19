# Musanze Harvest Intelligence Service

**Individual Assignment 2 — SWE 3513 Artificial Intelligence**

| | |
|---|---|
| Full name | Haroun Mohammed Arbab Yousef |
| Registration number | 25/26972 |
| Institutional username | ug2526972@ines.ac.rw |
| Verification code | **AI-A2-6972** |
| Random seed (LLLL) | **6972** |
| Python | 3.11.9 |
| Tested on | Windows 11 (PowerShell), Python 3.11.9 |

## Problem

A potato cooperative in Musanze receives farm records in inconsistent spreadsheets. The field officer
needs a local, evidence-based tool that estimates harvest quantity (**regression**), identifies farms
at high production risk (**classification**) and groups farms with similar conditions (**clustering**).
Internet access may be unreliable, so training and prediction work fully offline; a small FastAPI
service is provided for later mobile integration.

> **The dataset is 100 % synthetic.** It is generated from my registration digits and is not a
> measurement of any real farm. This service is a classroom decision-support prototype and must never
> be used to decide access to finance, insurance or public services; every result requires human review.

## Setup

```powershell
python -m venv .venv
# activate:  .venv\Scripts\Activate.ps1   (Windows)  /  source .venv/bin/activate  (Linux/macOS)
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used to produce the submitted artifacts:
numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0,
fastapi 0.148.0, uvicorn 0.54.0, pydantic 2.13.5, httpx 0.28.1, pytest 9.1.1, reportlab 5.0.1.
`reportlab` is used only to regenerate the technical report and evidence PDFs; it is justified here
and can be removed if only the model pipeline is required.

## Run commands

```powershell
python run_all.py --student-id ug2526972@ines.ac.rw --seed 6972   # recreate data, train, regenerate artifacts
python predict.py --json samples/valid.json                       # valid CLI prediction
python predict.py --json samples/invalid.json                     # invalid record -> error, exit code 1
pytest -q                                                         # automated tests
uvicorn app:app --host 127.0.0.1 --port 8000                      # FastAPI service
python -m src.report                                              # regenerate the report PDF from the artifacts
python -m src.evidence                                            # regenerate test log + declaration PDFs
```

`run_all.py` rejects any seed other than 6972 and any student id that is not mine.

## Reproducibility

* Generator: `data/generate_data.py` uses `numpy.random.default_rng(6972)`.
* Base rows: `240 + (6972 mod 41)` = **242**; then 12 missing soil-moisture values, 12 missing
  fertilizer values and 3 exact duplicate rows are injected. All numeric values use 4 decimal places.
* Dataset SHA-256: **a05f4763ec64a6967d3bd49a679c6ae351d143bb1ffe5827fe99e685a66d4474**
  (recorded in `artifacts/data_fingerprint.txt`). Re-running `run_all.py` reproduces the fingerprint
  and metrics within floating-point tolerance.

## Artifacts produced

```
data/AI_A2_2526972.csv                      synthetic dataset (245 raw / 242 clean rows)
artifacts/data_fingerprint.txt              SHA-256 of the final CSV
artifacts/data_validation.json              schema, range, missing, duplicate and leakage checks
artifacts/regression_metrics.json           MAE / RMSE / R2 for baseline + 2 candidates, selection reason
artifacts/classification_metrics.json       precision / recall / F1 / confusion matrix / ROC-AUC + reason
artifacts/clustering_report.json            silhouette scores K=2..6, selected K, centres, profiles
artifacts/test_predictions.csv              row id, actuals, predictions, probabilities, errors, cluster
artifacts/plots/residuals.png               regression residuals (test set)
artifacts/plots/confusion_matrix.png        selected classifier confusion matrix
artifacts/plots/clusters.png                cluster map (input features only)
models/                                     saved preprocessor + regression, classification, clustering pipelines
docs/AI_A2_2526972_REPORT.pdf               five-to-seven page technical report
evidence/                                   demo video, test log, declaration, AI-use disclosure
```

## Results (seed 6972)

| Task | Baseline | Selected model | Key metrics |
|---|---|---|---|
| Regression (`yield_tons`) | mean, MAE 2.6024 | **linear_regression** | MAE 0.895, RMSE 1.1486, R2 0.8645 |
| Classification (`high_risk`) | majority, F1 0.0 | **logistic_regression** | precision 0.9259, recall 1.0, F1 0.9615, ROC-AUC 0.995 |
| Clustering | — | **KMeans K=2** | silhouette 0.1573 |

Linear/logistic models win because the generator is an explicit linear formula plus small Gaussian
noise; the random forests cannot extrapolate that continuous trend and split it into steps.

## API

Start `uvicorn app:app --host 127.0.0.1 --port 8000`, then:

* `GET /health` → identity, seed and `model_loaded`.
* `GET /model-info` → selected models, metrics, features and intended-use statement.
* `POST /predict` → typed body with the seven features; returns yield, risk flag, probability and cluster.

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"rainfall_mm\":950,\"avg_temp_c\":18.5,\"soil_moisture_pct\":55,\"fertilizer_kg\":120,\"field_size_ha\":1.8,\"season\":\"A\",\"cooperative_support\":1}"
```

Valid response: `200 OK` with `{"yield_tons": 12.0151, "high_risk": 0, "risk_probability": 0.2241, "cluster": 1, "verification_code": "AI-A2-6972", ...}`.
Invalid record: `422` with one readable message per broken rule (missing field, wrong type,
out-of-range value, unknown field).

## Known limitations

* Synthetic linear generator: the task is easier than real agronomy and shuffles only 7 features.
* Missing values are imputed, never observed; imputation is fitted on training rows only.
* Clustering separation is weak (silhouette 0.157) and mainly reflects cooperative support.
* No temporal validation, no probability calibration, no cost-sensitive threshold tuning.
* The model must not be used for real farms without collecting real, multi-season records.

## Repository

* URL: `TODO_FILL_REPOSITORY_URL`
* Commit history:
  * `4bb42e6da4d231c41b3f9cb4952978ddad03af2e` — full system: code, models, artifacts, report
  * `74518e70e064325ba993facaa3779f4732ef7498` — documents the system commit in this README
  * later commits — run `git log --oneline` (a commit cannot contain its own hash)

## Package layout

```
AI_A2_2526972/
  README.md  requirements.txt  run_all.py  predict.py  app.py
  src/                reusable modules (config, generation, validation, preprocessing, modeling, inference, plots, report, evidence)
  data/generate_data.py  data/AI_A2_2526972.csv
  models/             saved pipelines loaded by predict.py and app.py
  artifacts/          JSON, CSV, TXT and PNG evidence
  samples/            valid.json and invalid.json
  tests/              pytest suite (17 tests)
  docs/AI_A2_2526972_REPORT.pdf
  evidence/           AI_A2_2526972_DEMO.mp4, AI_A2_2526972_TEST_LOG.pdf, AI_A2_2526972_DECLARATION.pdf, AI_USE.md
```
