"""Prepare prediction inputs and define shared feature processing for Cell2Cell."""

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.utils.validation import check_is_fitted, validate_data

from src.data import RANDOM_STATE
from src.features import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PREDICTOR_COLUMNS,
    SERVICE_AREA,
)

AGE_COLUMNS = ("AgeHH1", "AgeHH2")
# Billing values and usage changes can be signed, so keep their original units.
SIGNED_COLUMNS = (
    "MonthlyRevenue",
    "TotalRecurringCharge",
    "PercChangeMinutes",
    "PercChangeRevenues",
)
MISSING_CATEGORY = "__MISSING__"


def prepare_predictors(customers: pd.DataFrame) -> pd.DataFrame:
    """Select and clean the raw columns available before retention outreach.

    This keeps the original row index and excludes churn labels, identifiers,
    and fields with uncertain timing. It applies fixed rules to placeholders
    and yes/no values. Call it on raw records before model fitting or scoring.
    """
    if not isinstance(customers, pd.DataFrame) or not customers.columns.is_unique:
        raise ValueError("Pass a DataFrame with unique column names")
    missing = sorted(set(PREDICTOR_COLUMNS).difference(customers.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(missing)}")

    cleaned = customers.loc[:, list(PREDICTOR_COLUMNS)].copy()
    # Zero household ages are placeholders. Negative equipment ages are invalid.
    for column in AGE_COLUMNS:
        cleaned[column] = cleaned[column].where(cleaned[column] > 0)
    days = cleaned["CurrentEquipmentDays"]
    cleaned["CurrentEquipmentDays"] = days.where(days >= 0)

    for column in (*CATEGORICAL_COLUMNS, SERVICE_AREA):
        values = cleaned[column].astype("string")
        if column == "IncomeGroup":
            # CSV readers may represent integer codes as floats when values are missing.
            values = values.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
        cleaned[column] = values.fillna(MISSING_CATEGORY).astype(object)

    for column in BINARY_COLUMNS:
        values = cleaned[column]
        encoded = values.map({"Yes": 1.0, "No": 0.0})
        if (values.notna() & encoded.isna()).any():
            raise ValueError(f"Expected Yes/No values in {column}")
        cleaned[column] = encoded
    return cleaned


class LogSkewedNumeric(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Compress right-skewed nonnegative measurements, chosen on training data."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        validate_data(self, X, skip_check_array=True)
        skew = X.skew()
        self.log_columns_ = [
            column
            for column in X.columns
            if column not in SIGNED_COLUMNS
            and (X[column].dropna() >= 0).all()
            and skew[column] > 1
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "log_columns_")
        validate_data(self, X, reset=False, skip_check_array=True)
        result = X.copy()
        for column in self.log_columns_:
            values = result[column].where(result[column] >= 0).astype(float)
            result[column] = np.log1p(values)
        return result


class ServiceAreaFrequencyEncoder(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Represent each service area by its share of training rows."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        validate_data(self, X, skip_check_array=True)
        self.frequencies_ = X[SERVICE_AREA].value_counts(normalize=True)
        return self

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        check_is_fitted(self, "frequencies_")
        validate_data(self, X, reset=False, skip_check_array=True)
        frequencies = X[SERVICE_AREA].map(self.frequencies_).fillna(0.0)
        return frequencies.to_numpy().reshape(-1, 1)


def numeric_median_imputer() -> SimpleImputer:
    """Fill numeric blanks with training medians and record missingness."""
    return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)


def target_encode_service_area() -> TargetEncoder:
    """Estimate smoothed churn rates by area using five internal training splits."""
    return TargetEncoder(
        target_type="binary",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )


def combine_feature_steps(
    *,
    numeric: BaseEstimator,
    service_area: BaseEstimator,
) -> ColumnTransformer:
    """Attach explicit numeric and ServiceArea steps to the prepared inputs.

    Use mode imputation for yes/no columns and one-hot encoding for other
    categories. The caller can set the column transformer's sparse_threshold
    to 0.0 for models that use dense input.
    """
    binary_imputer = SimpleImputer(
        strategy="most_frequent", add_indicator=True, keep_empty_features=True,
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_COLUMNS)),
            ("binary", binary_imputer, list(BINARY_COLUMNS)),
            ("service_area", service_area, [SERVICE_AREA]),
            ("category", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_COLUMNS)),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )
