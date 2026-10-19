# AI use disclosure - AI-A2-6972

Student: Haroun Mohammed Arbab Yousef (25/26972) - ug2526972@ines.ac.rw

## Tools used

| Tool | How it was used |
|---|---|
| AI coding assistant (OpenCode / big-pickle) | Drafted and refactored the Python modules (`src/generate_data.py`, `src/validate.py`, `src/preprocess.py`, `src/modeling.py`, `src/plots.py`, `src/inference.py`, `run_all.py`, `predict.py`, `app.py`, `tests/`, report generator). |
| Assignment brief PDF | Source of every rule: seed, row count, formula, missing/duplicate defects, artifact list, API contract. |

## What the assistant did NOT do

- It did not choose the registration number, seed (6972) or verification code (AI-A2-6972) - those are mine.
- It did not present the results as real-world evidence: the dataset is synthetic by construction.
- It did not replace my understanding: I ran `run_all.py`, `predict.py`, `uvicorn app:app` and `pytest -q`
  myself, read the metrics JSON files, inspected the three plots and wrote the interpretation of the
  regression error, the classification error and the cluster meaning that appear in the report.

## My responsibility

I can explain and modify every function in this repository (seed handling, validation ranges,
preprocessing pipeline, model selection criteria, API schema and tests). If the assessor asks for a
live change (for example a new range rule or an extra response field), I will make it directly in the
code and re-run `run_all.py` to confirm the artifacts stay reproducible.

## Honesty statement

All numbers in `docs/AI_A2_2526972_REPORT.pdf` were read back from the generated artifacts at build
time, not copied by hand. The dataset fingerprint `a05f4763ec64a6967d3bd49a679c6ae351d143bb1ffe5827fe99e685a66d4474`
is produced by `data/generate_data.py` on any clean computer.
