"""Plot generation: residuals, confusion matrix and cluster map."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src import config

PLOT_STYLE = {"dpi": 130, "bbox_inches": "tight"}


def _footer(title: str) -> str:
    return f"{title}  |  {config.VERIFICATION_CODE}  |  seed {config.SEED}"


def plot_residuals(y_true, y_pred, path=config.PLOTS_DIR / "residuals.png") -> str:
    """Residual vs predicted scatter for the selected regression model."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_pred, residuals, alpha=0.75, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted yield_tons")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(_footer("Regression residuals on the held-out test set"))
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **PLOT_STYLE)
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(
    matrix, path=config.PLOTS_DIR / "confusion_matrix.png", model_name: str = ""
) -> str:
    """Confusion matrix heatmap for the selected classifier."""
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["normal (0)", "high_risk (1)"])
    ax.set_yticks([0, 1], labels=["normal (0)", "high_risk (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    thresh = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=13,
            )
    ax.set_title(_footer(f"Confusion matrix - {model_name or 'selected classifier'}"))
    fig.colorbar(im, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **PLOT_STYLE)
    plt.close(fig)
    return str(path)


def plot_clusters(
    features: pd.DataFrame,
    labels,
    preprocessor,
    path=config.PLOTS_DIR / "clusters.png",
    selected_k: int | None = None,
) -> str:
    """Two-dimensional PCA map of the selected clusters (no target column used)."""
    scaled = preprocessor.transform(features)
    coords = PCA(n_components=2, random_state=config.SEED).fit_transform(scaled)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.8,
        edgecolor="k",
        linewidth=0.3,
    )
    ax.set_xlabel("PCA component 1 (input features only)")
    ax.set_ylabel("PCA component 2")
    title = f"Farm clusters (K={selected_k})" if selected_k else "Farm clusters"
    ax.set_title(_footer(title))
    ax.grid(alpha=0.3)
    handles, _ = scatter.legend_elements()
    ax.legend(
        handles,
        [f"cluster {c}" for c in sorted(set(labels.tolist()))],
        title="cluster",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **PLOT_STYLE)
    plt.close(fig)
    return str(path)
