"""Train/test separation and preprocessing.

Imputers, scalers and encoders are fitted on training data only, inside a
single ColumnTransformer that is reused verbatim at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def split_features_targets(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    np.ndarray,
    np.ndarray,
]:
    """Split into train/test features, both targets and original row indices.

    The split is fixed with ``random_state = LLLL`` and stratified on
    ``high_risk`` so every run produces identical train/test rows.
    """
    features = frame[config.FEATURE_COLUMNS]
    y_reg = frame[config.TARGET_REGRESSION]
    y_clf = frame[config.TARGET_CLASSIFICATION]
    indices = frame.index.to_numpy()

    x_train, x_test, yreg_train, yreg_test, yclf_train, yclf_test, idx_train, idx_test = (
        train_test_split(
            features,
            y_reg,
            y_clf,
            indices,
            test_size=config.TEST_SIZE,
            random_state=config.SEED,
            stratify=y_clf,
        )
    )
    return x_train, x_test, yreg_train, yreg_test, yclf_train, yclf_test, idx_train, idx_test


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median imputation + scaling. Categorical: most-frequent + one-hot."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
