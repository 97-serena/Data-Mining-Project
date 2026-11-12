"""fixed tenure baseline, with the dummy classifier supplied by scikit-learn"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class TenureRule(ClassifierMixin, BaseEstimator):
    """give higher priority to customers in months 11–12

    scores are fixed priorities, not probabilities, and missing tenure gets a score of zero
    """

    def fit(self, X, y=None):
        if 'MonthsInService' not in X.columns:
            raise ValueError('The tenure rule requires MonthsInService')
        # only record the interface expected by the shared evaluator
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X):
        """return fixed priorities in scikit-learn's two-column format"""
        check_is_fitted(self, 'classes_')
        in_window = X['MonthsInService'].between(11, 12)
        in_window = in_window.fillna(False)
        scores = in_window.to_numpy(dtype=float)
        return np.column_stack([1 - scores, scores])

    def predict(self, X):
        """flag customers whose tenure falls in the fixed window"""
        probabilities = self.predict_proba(X)
        positive_scores = probabilities[:, 1]
        return positive_scores.astype(int)
