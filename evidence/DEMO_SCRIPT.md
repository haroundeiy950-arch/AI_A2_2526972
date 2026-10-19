# Demo video script — AI_A2_2526972 (2 minutes, unedited, voice required)

Say your name, **Haroun Mohammed Arbab Yousef**, registration **25/26972** and code
**AI-A2-6972** in the recording. Keep the terminal clock visible at the start or end.

| # | What to show | Command / file | ~Time |
|---|---|---|---|
| 1 | Handwritten card: full name, registration, AI-A2-6972 | hold to camera | 0:00-0:15 |
| 2 | Final commit hash | `git log -1 --format="%H %s"` (show `b5591f7` or whatever is last) | 0:15-0:25 |
| 3 | Full pipeline runs, fingerprint + artifact list appear | `python run_all.py --student-id ug2526972@ines.ac.rw --seed 6972` | 0:25-0:50 |
| 4 | Open one metrics JSON and one plot, explain ONE result without reading | `artifacts/classification_metrics.json` + `artifacts/plots/confusion_matrix.png` | 0:50-1:15 |
| 5 | CLI: valid record then the deliberately invalid one | `python predict.py --json samples/valid.json` then `... invalid.json` (expect ERROR, exit 1) | 1:15-1:30 |
| 6 | FastAPI docs, call /health and /predict, show one 422 | `uvicorn app:app --port 8000` then `http://127.0.0.1:8000/docs` | 1:30-1:50 |
| 7 | Tests pass, then report + declaration, end on clock | `pytest -q` then open `docs/…_REPORT.pdf`, `evidence/…_DECLARATION.pdf` | 1:50-2:00 |

One result you can explain in your own words (from this machine):

> "The logistic regression flags 25 of the 25 truly high-risk farms and gets 22 of 24 normal
> farms right — F1 0.9615, ROC-AUC 0.995. The two false positives are farms sitting right on
> the median yield-per-hectare line, so their label flips with the random noise."
