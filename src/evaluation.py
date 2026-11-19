"""Compare models and preprocessing on shared training-partition validation folds."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.data import RANDOM_STATE

CV_FOLDS = 5
CV_REPEATS = 2
SCORING = {"Average precision": "average_precision", "ROC AUC": "roc_auc"}


def make_shared_folds(
    X: pd.DataFrame, y: pd.Series,
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Return pairs of training and validation row positions.

    The same ordered inputs and aligned churn labels produce identical splits.
    """
    if not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError("Predictors and target must have matching, unique row indices")
    counts = y.value_counts(dropna=False)
    if set(counts.index) != {0, 1} or counts.min() < CV_FOLDS:
        raise ValueError(f"Need at least {CV_FOLDS} examples of each binary churn label")
    splitter = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    return list(splitter.split(X, y))


def make_model_pipeline(estimator: BaseEstimator, preprocessor: BaseEstimator) -> Pipeline:
    """Build an unfitted pipeline with independent copies of both components."""
    return Pipeline([("preprocess", clone(preprocessor)), ("model", clone(estimator))])


def compare_preprocessing(
    estimator: BaseEstimator,
    preprocessors: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Return one row per option, with mean and sample SD for AP and ROC AUC.

    Pass named preprocessors, the output of prepare_predictors(), and the
    aligned churn labels. Each option uses the same model settings and
    validation folds. Its preprocessing and model are fitted together on
    each fold's training rows.
    """
    if not preprocessors:
        raise ValueError("Provide at least one preprocessing option")

    pipelines = {
        name: make_model_pipeline(estimator, preprocessor)
        for name, preprocessor in preprocessors.items()
    }
    return compare_models(pipelines, X, y, n_jobs=n_jobs).rename_axis("Preprocessing")


def compare_models(
    models: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Score named classifiers on shared folds and summarize AP and ROC AUC."""
    if not models:
        raise ValueError("Provide at least one model")

    folds = make_shared_folds(X, y)
    summaries = {}
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            scoring=SCORING,
            cv=folds,
            n_jobs=n_jobs,
            error_score="raise",
        )
        summary = {}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            summary[f"{metric} (mean)"] = np.mean(values)
            summary[f"{metric} (SD)"] = np.std(values, ddof=1)
        summaries[name] = summary
    return pd.DataFrame.from_dict(summaries, orient="index").rename_axis("Model")


def tree_feature_importance(fitted_pipeline: Pipeline) -> pd.Series:
    """Rank encoded columns by a fitted decision tree or forest's built-in importance."""
    check_is_fitted(fitted_pipeline)
    model = fitted_pipeline.named_steps["model"]
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        raise TypeError(f"{type(model).__name__} does not expose feature_importances_")

    columns = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    return pd.Series(importance, index=columns, name="Importance").sort_values(ascending=False)
