"""Shared preprocessing steps for the churn models.

Fixed rules that learn nothing from the data live in `clean_raw()` and can run before the
cross-validation split. Anything that learns a value from the data (such as a median) is a
transformer, so a pipeline fits it on the training folds only.

Order in the final pipeline: `clean_raw()` -> engineered features -> imputation.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MISSING_CATEGORY = "__MISSING__"

# nonbinary categorical predictors, encoded as categories rather than numbers
CATEGORICAL_COLUMNS = [
    "ServiceArea",
    "HandsetPrice",
    "IncomeGroup",
    "CreditRating",
    "PrizmCode",
    "Occupation",
    "MaritalStatus",
    "Homeownership",
]

# age columns and the flags that mark a recorded zero
AGE_ZERO_FLAGS = {"AgeHH1": "age1_zero", "AgeHH2": "age2_zero"}

# indicator names that differ from the default "<column>_missing"
INDICATOR_NAMES = {"PercChangeMinutes": "ChangeMissing"}

# text values that would mean "missing" if they ever appeared in a column
MISSING_MARKERS = ["NA", "", "empty"]


def indicator_name(column):
    """Name of the missingness indicator for a column."""
    return INDICATOR_NAMES.get(column, f"{column}_missing")


def clean_raw(df):
    """Apply the fixed cleaning rules and return a cleaned copy.

    - `NA` and empty cells are already read as missing by `pandas.read_csv`; stop if any
      other missing marker is left in a text column.
    - Negative `CurrentEquipmentDays` become missing (an equipment age cannot be negative).
    - Categorical columns keep recorded values such as "Unknown", "Other" and
      `IncomeGroup` = 0 as their own categories; genuine missing values become `__MISSING__`.
    """
    out = df.copy()

    # stop if a missing marker survived as text
    for column in out.select_dtypes(exclude="number").columns:
        leftover = out[column].isin(MISSING_MARKERS)
        if leftover.any():
            raise ValueError(f"{column} contains {int(leftover.sum())} unparsed missing markers")

    # negative equipment ages are invalid
    is_negative_age = out["CurrentEquipmentDays"] < 0
    out.loc[is_negative_age, "CurrentEquipmentDays"] = np.nan

    # store categories as text and mark genuine missing values
    for column in CATEGORICAL_COLUMNS:
        if column not in out.columns:
            continue
        is_missing = out[column].isna()
        values = out[column].astype(str)
        values[is_missing] = MISSING_CATEGORY
        out[column] = values.astype(object)

    return out


class MedianImputer(BaseEstimator, TransformerMixin):
    """Median imputation with missingness indicators, learned on the fitted rows only.

    An indicator is added for each column that has missing values in the fitted data.
    Use `AgeImputer` for the age columns, because their zeros need separate handling.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.medians_ = X.median()
        self.indicator_columns_ = [c for c in X.columns if X[c].isna().any()]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)[list(self.feature_names_in_)]
        indicators = {indicator_name(c): X[c].isna().astype(float)
                      for c in self.indicator_columns_}
        filled = X.astype(float).fillna(self.medians_)
        return pd.concat([filled, pd.DataFrame(indicators, index=X.index)], axis=1)

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        names += [indicator_name(c) for c in self.indicator_columns_]
        return np.asarray(names, dtype=object)


class AgeImputer(BaseEstimator, TransformerMixin):
    """Handle recorded zeros and genuine missing values in `AgeHH1` and `AgeHH2`.

    A recorded zero is a placeholder (real ages start at 18), so medians use nonzero ages only.

    zero_strategy="keep": keep the zeros and add `age1_zero` / `age2_zero`.
    zero_strategy="median": replace the zeros with the median, without zero flags.

    In both cases genuine missing ages get the median plus a missingness indicator.
    """

    def __init__(self, zero_strategy="keep"):
        self.zero_strategy = zero_strategy

    def fit(self, X, y=None):
        if self.zero_strategy not in ("keep", "median"):
            raise ValueError("zero_strategy must be 'keep' or 'median'")
        X = pd.DataFrame(X)[list(AGE_ZERO_FLAGS)]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.medians_ = {c: X.loc[X[c] > 0, c].median() for c in X.columns}
        self.indicator_columns_ = [c for c in X.columns if X[c].isna().any()]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)[list(self.feature_names_in_)].astype(float)
        out = X.copy()
        for column in self.feature_names_in_:
            is_zero = X[column] == 0
            is_missing = X[column].isna()
            if self.zero_strategy == "keep":
                out[AGE_ZERO_FLAGS[column]] = is_zero.astype(float)
            else:
                out.loc[is_zero, column] = self.medians_[column]
            out.loc[is_missing, column] = self.medians_[column]
        for column in self.indicator_columns_:
            out[indicator_name(column)] = X[column].isna().astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.zero_strategy == "keep":
            names += [AGE_ZERO_FLAGS[c] for c in self.feature_names_in_]
        names += [indicator_name(c) for c in self.indicator_columns_]
        return np.asarray(names, dtype=object)
