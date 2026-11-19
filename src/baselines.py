"""Reference scores that require no learned relationship with predictors."""

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted

from src.data import RANDOM_STATE


def constant_churn_baseline() -> DummyClassifier:
    """Score every customer with the churn rate in the training labels."""
    return DummyClassifier(strategy="prior")


def stratified_random_baseline() -> DummyClassifier:
    """Draw random labels in proportion to the training labels."""
    return DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)


class TenureRule(ClassifierMixin, BaseEstimator):
    """Give higher priority to customers with 11–12 months of service."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Register the binary classes for scikit-learn without learning parameters."""
        if "MonthsInService" not in X.columns:
            raise ValueError("The tenure rule requires MonthsInService")
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Return fixed ranking scores, with missing tenure assigned zero."""
        check_is_fitted(self, "classes_")
        in_window = X["MonthsInService"].between(11, 12)
        return in_window.fillna(False).to_numpy(dtype=float)

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        """Flag customers in the tenure window."""
        return self.decision_function(X).astype(int)
