"""Keep early-stopping data separate from fitted preprocessing."""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from src.data import RANDOM_STATE
from src.preprocessing import make_preprocessor


def _dense(values):
    return values.toarray() if sparse.issparse(values) else values


class HoldoutEarlyStoppingClassifier(ClassifierMixin, BaseEstimator):
    """Split raw rows before learning preprocessing or fitting HistGradientBoosting.

    The estimator must accept X_val and y_val in fit, as HistGradientBoosting does.
    Training uses fit_transform for cross-fitted target encoding; stopping and future
    rows use transform. The fitted model keeps the internal fitting rows only.
    """

    def __init__(self, preprocessor, estimator, validation_fraction=0.1,
                 random_state=RANDOM_STATE):
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X, y):
        if not 0 < self.validation_fraction < 1:
            raise ValueError('validation_fraction must be between zero and one')
        if not X.index.is_unique or not X.index.equals(y.index):
            raise ValueError('Predictors and target must have matching unique indices')
        fitting_rows, stopping_rows = train_test_split(
            np.arange(len(X)), test_size=self.validation_fraction,
            stratify=y, random_state=self.random_state,
        )
        self.fitting_rows_ = fitting_rows
        self.stopping_rows_ = stopping_rows
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.preprocessor_ = clone(self.preprocessor)
        fitting_values = self.preprocessor_.fit_transform(
            X.iloc[fitting_rows], y.iloc[fitting_rows],
        )
        stopping_values = self.preprocessor_.transform(X.iloc[stopping_rows])
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(
            _dense(fitting_values), y.iloc[fitting_rows],
            X_val=_dense(stopping_values), y_val=y.iloc[stopping_rows],
        )
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict_proba(_dense(self.preprocessor_.transform(X)))

    def predict(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(_dense(self.preprocessor_.transform(X)))


def make_early_stopping_pipeline(estimator, groups, model_kind='tree',
                                 validation_fraction=0.1, **preprocessing_options):
    """Build an unfitted classifier with a separate internal stopping sample."""
    return HoldoutEarlyStoppingClassifier(
        make_preprocessor(groups, model_kind=model_kind, **preprocessing_options),
        estimator, validation_fraction=validation_fraction,
    )
