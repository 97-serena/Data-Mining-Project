"""Compare preprocessing choices on shared training-partition validation folds."""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.data import RANDOM_STATE

CV_FOLDS = 5
CV_REPEATS = 2
SCORING = {"Average precision": "average_precision", "ROC AUC": "roc_auc"}


def make_shared_folds(X, y):
    """Return the same repeated stratified splits for the same ordered rows."""
    if not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError("Predictors and target must have matching, unique row indices")
    counts = y.value_counts(dropna=False)
    if set(counts.index) != {0, 1} or counts.min() < CV_FOLDS:
        raise ValueError(f"Need at least {CV_FOLDS} examples of each binary churn label")
    splitter = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE,
    )
    return list(splitter.split(X, y))


def make_model_pipeline(estimator, preprocessor):
    """Join preprocessing and a classifier so they are fitted together."""
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def compare_preprocessing(estimator, preprocessors, X, y, *, n_jobs=1):
    """Report the mean and sample standard deviation of each validation metric.

    Pass a mapping of labels to unfitted preprocessors, and the ordered training
    predictors and churn labels from load_training_data(). Every option uses the
    same classifier settings and fold assignments. Cross-validation fits fresh
    copies of both the preprocessor and the classifier for every fold.
    """
    if not preprocessors:
        raise ValueError("Provide at least one preprocessing option")
    folds = make_shared_folds(X, y)
    rows = []
    for name, preprocessor in preprocessors.items():
        pipeline = make_model_pipeline(estimator, preprocessor)
        scores = cross_validate(
            pipeline, X, y, scoring=SCORING, cv=folds, n_jobs=n_jobs, error_score="raise",
        )
        row = {"Preprocessing": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric} (mean)"] = np.mean(values)
            row[f"{metric} (SD)"] = np.std(values, ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Preprocessing")


def tree_feature_importance(fitted_pipeline):
    """Return a fitted tree's built-in importance for each transformed column."""
    check_is_fitted(fitted_pipeline)
    model = fitted_pipeline.named_steps["model"]
    columns = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    return pd.Series(model.feature_importances_, index=columns, name="Importance").sort_values(
        ascending=False,
    )
