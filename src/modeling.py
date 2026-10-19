"""Model training and evaluation for regression, classification and clustering.

Every candidate is a full sklearn Pipeline (preprocessing + estimator) so the
exact same transformations are applied at training and inference time.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src import config
from src.preprocess import build_preprocessor

CLASS_LABELS = [0, 1]
POSITIVE_CLASS = 1


def _regression_scores(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def _classification_scores(y_true, y_pred, y_prob=None) -> dict:
    scores = {
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
    }
    if y_prob is not None:
        try:
            scores["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except ValueError:
            scores["roc_auc"] = None
    else:
        scores["roc_auc"] = None
    return scores


def _pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


# ---------------------------------------------------------------- regression --
def train_regression(x_train, y_train, x_test, y_test) -> tuple[dict, dict, str]:
    """Compare a mean baseline with two regression candidates on the test set.

    Returns (metrics dict, fitted pipelines keyed by name, selected name).
    """
    candidates = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=config.SEED, n_jobs=-1
        ),
    }

    fitted: dict[str, Pipeline] = {}
    scores: dict[str, dict] = {}
    for name, estimator in candidates.items():
        pipe = _pipeline(estimator)
        pipe.fit(x_train, y_train)
        fitted[name] = pipe
        scores[name] = _regression_scores(y_test, pipe.predict(x_test))

    baseline_name = "mean_baseline"
    selected = min(scores, key=lambda n: scores[n]["mae"])
    winner = scores[selected]
    runner = sorted(scores, key=lambda n: scores[n]["mae"])[1]
    reason = (
        f"Selected by lowest test MAE among all evaluated models: "
        f"{selected} MAE={winner['mae']} vs {runner} MAE={scores[runner]['mae']} "
        f"(baseline MAE={scores[baseline_name]['mae']}). Test set has "
        f"{len(y_test)} rows, split with random_state={config.SEED}."
    )

    metrics = {
        "task": "regression",
        "target": config.TARGET_REGRESSION,
        "verification_code": config.VERIFICATION_CODE,
        "seed": config.SEED,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "baseline": {**scores[baseline_name], "model": "DummyRegressor(mean)"},
        "candidates": {
            name: {**scores[name], "model": name}
            for name in candidates
            if name != baseline_name
        },
        "all_models": scores,
        "selected_model": selected,
        "selection_reason": reason,
        "metric_notes": {
            "mae": "mean absolute error in tons, lower is better",
            "rmse": "root mean squared error in tons, lower is better",
            "r2": "coefficient of determination on the held-out test set",
        },
    }
    return metrics, fitted, selected


# ------------------------------------------------------------ classification --
def train_classification(x_train, y_train, x_test, y_test) -> tuple[dict, dict, str]:
    """Compare a majority baseline with two classifiers on the test set."""
    candidates = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=config.SEED, n_jobs=-1
        ),
    }

    fitted: dict[str, Pipeline] = {}
    scores: dict[str, dict] = {}
    for name, estimator in candidates.items():
        pipe = _pipeline(estimator)
        pipe.fit(x_train, y_train)
        fitted[name] = pipe
        y_pred = pipe.predict(x_test)
        y_prob = None
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            proba = pipe.predict_proba(x_test)
            classes = list(pipe.named_steps["model"].classes_)
            if POSITIVE_CLASS in classes:
                y_prob = proba[:, classes.index(POSITIVE_CLASS)]
        scores[name] = _classification_scores(y_test, y_pred, y_prob)

    baseline_name = "majority_baseline"
    selected = max(
        scores,
        key=lambda n: (scores[n]["f1"], scores[n]["roc_auc"] or 0.0),
    )
    reason = (
        f"Selected by highest test F1 (tie-break ROC-AUC): {selected} "
        f"F1={scores[selected]['f1']}, ROC-AUC={scores[selected]['roc_auc']} vs "
        f"baseline F1={scores[baseline_name]['f1']}, "
        f"ROC-AUC={scores[baseline_name]['roc_auc']}."
    )

    metrics = {
        "task": "classification",
        "target": config.TARGET_CLASSIFICATION,
        "positive_class": POSITIVE_CLASS,
        "class_labels": CLASS_LABELS,
        "features": config.FEATURE_COLUMNS,
        "leakage_control": (
            "yield_tons, yield per hectare and high_risk are excluded from the "
            "feature set; preprocessing is fitted on train rows only"
        ),
        "verification_code": config.VERIFICATION_CODE,
        "seed": config.SEED,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "baseline": {**scores[baseline_name], "model": "DummyClassifier(most_frequent)"},
        "candidates": {
            name: {**scores[name], "model": name}
            for name in candidates
            if name != baseline_name
        },
        "all_models": scores,
        "selected_model": selected,
        "selection_reason": reason,
        "metric_notes": {
            "precision": "of farms flagged high_risk, share that really were",
            "recall": "of truly high-risk farms, share that were flagged",
            "f1": "harmonic mean of precision and recall for class 1",
            "confusion_matrix": "[[tn, fp], [fn, tp]] with labels [0, 1]",
            "roc_auc": "area under ROC from positive-class probabilities",
        },
    }
    return metrics, fitted, selected


# ---------------------------------------------------------------- clustering --
def train_clustering(
    x_train: pd.DataFrame, x_all: pd.DataFrame
) -> tuple[dict, Pipeline, np.ndarray]:
    """Fit K=2..6, choose K by silhouette score, profile the selected clusters.

    Only the seven input features are used; no target column is involved.
    Returns (report dict, fitted pipeline, cluster labels for x_all).
    """
    forbidden = set(config.TARGET_COLUMNS)
    if forbidden & set(x_all.columns):
        raise ValueError(f"clustering must not use target columns: {forbidden & set(x_all.columns)}")

    probe = _pipeline(KMeans(n_clusters=2, random_state=config.SEED, n_init=10))
    scaled_train = probe.named_steps["prep"].fit_transform(x_train)

    scores: dict[str, float] = {}
    models: dict[int, KMeans] = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
        km.fit(scaled_train)
        models[k] = km
        scores[str(k)] = round(float(silhouette_score(scaled_train, km.labels_)), 4)

    selected_k = max(scores, key=lambda k: scores[k])
    k = int(selected_k)
    pipeline = Pipeline(
        [("prep", build_preprocessor()), ("kmeans", models[k])]
    )
    pipeline.fit(x_train)
    labels_all = pipeline.predict(x_all)

    train_labels = models[k].labels_
    centers = (
        x_all.assign(_cluster=labels_all)
        .groupby("_cluster")[config.NUMERIC_FEATURES]
        .mean()
        .reindex(range(k))
    )
    center_units = "mean of original feature values per cluster (all rows)"
    center_frame = centers

    sizes = {int(c): int((train_labels == c).sum()) for c in range(k)}
    all_sizes = {int(c): int((labels_all == c).sum()) for c in range(k)}
    season_modes = {
        int(c): (
            x_all.loc[labels_all == c, "season"].mode().iloc[0]
            if not x_all.loc[labels_all == c, "season"].mode().empty
            else "n/a"
        )
        for c in range(k)
    }

    clusters = []
    for c in range(k):
        row = center_frame.loc[c]
        clusters.append(
            {
                "cluster": c,
                "size_train": sizes[c],
                "size_all_rows": all_sizes[c],
                "centers": {col: round(float(row[col]), 4) for col in center_frame.columns},
                "dominant_season": str(season_modes[c]),
                "profile": _plain_profile(row, c, center_frame, season_modes[c]),
            }
        )

    report = {
        "task": "clustering",
        "verification_code": config.VERIFICATION_CODE,
        "seed": config.SEED,
        "features_used": config.FEATURE_COLUMNS,
        "targets_used": "none - unsupervised grouping of input features only",
        "scaling": "StandardScaler fitted on training rows inside the pipeline",
        "k_scores": scores,
        "selected_k": k,
        "selection_reason": (
            f"K={k} gives the highest silhouette score on the training set "
            f"({scores[selected_k]}) among K=2..6. Silhouette ranges from -1 to 1; "
            f"higher means better separated, denser clusters."
        ),
        "center_units": center_units,
        "clusters": clusters,
        "interpretation": _report_interpretation(scores, str(k), clusters),
    }
    return report, pipeline, labels_all


def _plain_profile(row: pd.Series, cluster_id: int, centers: pd.DataFrame, season: str) -> str:
    """Build a plain-language farm profile from the cluster centre."""
    ranked: dict[str, list] = {}
    for col in config.NUMERIC_FEATURES:
        ordered = centers[col].sort_values()
        ranked[col] = list(ordered.index)

    highs, lows = [], []
    for col, order in ranked.items():
        value = float(row[col])
        if order[-1] == cluster_id:
            highs.append(f"{col} ({value:.2f})")
        elif order[0] == cluster_id:
            lows.append(f"{col} ({value:.2f})")

    spread = centers[config.NUMERIC_FEATURES].std().replace(0.0, 1e-9)
    z = ((centers.loc[cluster_id, config.NUMERIC_FEATURES] - centers[config.NUMERIC_FEATURES].mean()) / spread)
    driver = z.abs().idxmax()
    driver_z = float(z[driver])
    driver_value = float(row[driver])

    parts = [f"Cluster {cluster_id} ({season}-leaning farms):"]
    if highs:
        parts.append("the highest " + ", ".join(highs) + " among all clusters")
    if lows:
        parts.append("the lowest " + ", ".join(lows) + " among all clusters")
    if not highs and not lows:
        parts.append("near-average conditions on every measured feature")
    parts.append(
        f"its strongest difference is {driver}={driver_value:.2f} "
        f"(about {abs(driver_z):.1f} standard deviations from the overall mean)"
    )
    return "; ".join(parts) + "."


def _report_interpretation(scores: dict, best_k: str, clusters: list[dict]) -> str:
    strengths = ", ".join(f"K={k} ({v})" for k, v in scores.items())
    summaries = " | ".join(f"C{c['cluster']}: {c['profile']}" for c in clusters)
    return (
        f"Silhouette scores -> {strengths}. Selected K={best_k}. {summaries} "
        "Clusters describe growing conditions only; they are not yield or risk "
        "predictions and must be reviewed by the field officer before any action."
    )
