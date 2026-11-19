"""Build unfitted preprocessing for prediction-time inputs."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.utils.validation import check_is_fitted

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
SIGNED_COLUMNS = ("MonthlyRevenue", "TotalRecurringCharge", "PercChangeMinutes", "PercChangeRevenues")
MISSING_CATEGORY = "__MISSING__"


def _clean_inputs(X):
    """Select approved inputs and normalize recorded missing values and yes/no labels."""
    cleaned = X.loc[:, list(PREDICTOR_COLUMNS)].copy()
    for column in AGE_COLUMNS:
        cleaned[column] = cleaned[column].where(cleaned[column] > 0)
    days = cleaned["CurrentEquipmentDays"]
    cleaned["CurrentEquipmentDays"] = days.where(days >= 0)

    for column in (*CATEGORICAL_COLUMNS, SERVICE_AREA):
        values = cleaned[column].astype("string")
        if column == "IncomeGroup":
            values = values.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
        cleaned[column] = values.fillna(MISSING_CATEGORY).astype(object)

    for column in BINARY_COLUMNS:
        values = cleaned[column]
        encoded = values.map({"Yes": 1.0, "No": 0.0})
        if (values.notna() & encoded.isna()).any():
            raise ValueError(f"Expected Yes/No values in {column}")
        cleaned[column] = encoded
    return cleaned


def _input_feature_names(transformer, input_features):
    """Use the approved input names and their order after cleaning."""
    return np.asarray(PREDICTOR_COLUMNS, dtype=object)


class _SkewedLog(TransformerMixin, BaseEstimator):
    """Compress right-skewed nonnegative measurements, chosen on training data."""

    def fit(self, X, y=None):
        skew = X.skew()
        self.columns_ = [
            column for column in X.columns
            if column not in SIGNED_COLUMNS
            and (X[column].dropna() >= 0).all()
            and skew[column] > 1
        ]
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        result = X.copy()
        for column in self.columns_:
            values = result[column].where(result[column] >= 0).astype(float)
            result[column] = np.log1p(values)
        return result

    def get_feature_names_out(self, input_features):
        return np.asarray(input_features, dtype=object)


class _ServiceAreaFrequency(TransformerMixin, BaseEstimator):
    """Represent each service area by its share of training rows."""

    def fit(self, X, y=None):
        self.frequencies_ = X[SERVICE_AREA].value_counts(normalize=True)
        return self

    def transform(self, X):
        check_is_fitted(self, "frequencies_")
        frequencies = X[SERVICE_AREA].map(self.frequencies_).fillna(0.0)
        return frequencies.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features):
        return np.asarray(input_features, dtype=object)


def make_preprocessor(*, encoding="target", scale_numeric=False, log_numeric=False, dense=False):
    """Return a fresh scikit-learn pipeline for the Cell2Cell predictors.

    Choose how ServiceArea is encoded: "target" uses smoothed churn rates with
    internal stratified cross-fitting, "onehot" uses an indicator per area,
    and "frequency" uses the share of training rows in each area. Other
    categories always use one-hot encoding.

    Numeric blanks use training medians and binary blanks use training modes.
    Optional log compression applies to skewed, nonnegative measurements.
    Optional scaling centers and scales numeric inputs and their missing
    indicators. Set dense=True for estimators that require a dense matrix.
    """
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    area_encoders = {
        "target": TargetEncoder(
            target_type="binary",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        ),
        "onehot": onehot,
        "frequency": _ServiceAreaFrequency(),
    }
    if encoding not in area_encoders:
        raise ValueError("encoding must be 'target', 'onehot', or 'frequency'")

    numeric_steps = []
    if log_numeric:
        numeric_steps.append(("log", _SkewedLog()))
    numeric_steps.append(("impute", SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True,
    )))
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    columns = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_COLUMNS)),
            ("binary", SimpleImputer(
                strategy="most_frequent", add_indicator=True, keep_empty_features=True,
            ), list(BINARY_COLUMNS)),
            ("service_area", area_encoders[encoding], [SERVICE_AREA]),
            ("category", onehot, list(CATEGORICAL_COLUMNS)),
        ],
        sparse_threshold=0.0 if dense else 1.0,
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("clean", FunctionTransformer(_clean_inputs, feature_names_out=_input_feature_names)),
        ("encode", columns),
    ])
