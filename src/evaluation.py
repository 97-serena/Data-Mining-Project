"""shared five-fold evaluation for baselines and model pipelines

use the original training row order and reuse the same folds for every configuration
the test partition is not loaded here
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import RANDOM_STATE
from src.preprocessing import make_preprocessor


def make_shared_folds(X, y):
    """create reproducible training and validation positions for five folds"""
    if not X.index.equals(y.index):
        raise ValueError('Predictors and target must have the same row index and order')
    if set(y.unique()) != {0, 1}:
        raise ValueError('The target must contain at least five examples of both 0 and 1')
    class_counts = y.value_counts()
    if class_counts.min() < 5:
        raise ValueError('The target must contain at least five examples of both 0 and 1')
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(X, y))


def make_model_pipeline(estimator, groups, model_kind='linear', **preprocessing_options):
    """attach unfitted preprocessing to an estimator

    evaluation clones and fits the full pipeline within each fold to avoid leakage
    """
    preprocessor = make_preprocessor(
        groups, model_kind=model_kind, **preprocessing_options,
    )
    return Pipeline([('preprocess', preprocessor), ('model', estimator)])


def evaluate_classifier(estimator, X, y, folds, name):
    """return fold scores and out-of-fold churn scores in the original row order

    the estimator must provide predict_proba
    learned models need the full preprocessing and model pipeline
    """
    if not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError('Predictors and target must have matching, unique row indices')
    if set(y.unique()) != {0, 1}:
        raise ValueError('Use the binary churn target: 0 = stay, 1 = churn')
    if len(folds) != 5:
        raise ValueError('Use all five shared folds')

    # check the assignments before fitting: each row is validated exactly once
    validation_counts = np.zeros(len(X), dtype=int)
    for training_rows, validation_rows in folds:
        combined_rows = np.concatenate([training_rows, validation_rows])
        sorted_rows = np.sort(combined_rows)
        expected_rows = np.arange(len(X))
        if not np.array_equal(sorted_rows, expected_rows):
            raise ValueError('Each fold must partition all rows without overlap or duplicates')
        validation_counts[validation_rows] += 1
    if not np.all(validation_counts == 1):
        raise ValueError('Every row must appear in exactly one validation fold')

    # store scores by validation position to preserve the original row order
    score_rows = []
    churn_scores = np.full(len(X), np.nan)
    fold_numbers = np.zeros(len(X), dtype=int)

    for fold_number, (training_rows, validation_rows) in enumerate(folds, start=1):
        # a fresh copy prevents fitted state from carrying over between folds
        fitted = clone(estimator)
        training_features = X.iloc[training_rows]
        training_target = y.iloc[training_rows]
        fitted.fit(training_features, training_target)
        positive_column = list(fitted.classes_).index(1)
        validation_features = X.iloc[validation_rows]
        probabilities = fitted.predict_proba(validation_features)
        scores = probabilities[:, positive_column]
        if not np.isfinite(scores).all():
            raise ValueError('Churn scores must be finite')

        validation_target = y.iloc[validation_rows]
        prevalence = validation_target.mean()
        ap = average_precision_score(validation_target, scores)
        score_rows.append({
            'model': name,
            'fold': fold_number,
            'training_rows': len(training_rows),
            'validation_rows': len(validation_rows),
            'churn_prevalence': prevalence,
            'average_precision': ap,
            'ap_gain_over_prevalence': ap - prevalence,
        })
        churn_scores[validation_rows] = scores
        fold_numbers[validation_rows] = fold_number

    predictions = pd.DataFrame({
        'model': name,
        'fold': fold_numbers,
        'y_true': y.to_numpy(),
        'churn_score': churn_scores,
    }, index=X.index)
    return pd.DataFrame(score_rows), predictions


def summarize_scores(fold_results):
    """summarize average precision by fold rather than pooling predictions"""
    grouped_results = fold_results.groupby('model', sort=False)
    return grouped_results.agg(
        folds=('fold', 'count'),
        mean_ap=('average_precision', 'mean'),
        std_ap=('average_precision', 'std'),
        mean_churn_prevalence=('churn_prevalence', 'mean'),
        mean_ap_gain=('ap_gain_over_prevalence', 'mean'),
    )
