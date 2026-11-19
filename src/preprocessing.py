"""Build unfitted preprocessing for prediction-time inputs."""

from typing import Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder
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


def _clean_inputs(X: pd.DataFrame) -> pd.DataFrame:
    """Select eligible inputs and normalize missing values and yes/no labels."""
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError("Pass a DataFrame with unique column names")
    missing = sorted(set(PREDICTOR_COLUMNS).difference(X.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(missing)}")

    cleaned = X.loc[:, list(PREDICTOR_COLUMNS)].copy()
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


def _input_feature_names(_transformer, _input_features) -> NDArray[np.object_]:
    """Use the approved input names and their order after cleaning."""
    return np.asarray(PREDICTOR_COLUMNS, dtype=object)


class _SkewedLog(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
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

class _ServiceAreaFrequency(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
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


def make_preprocessor(
    *,
    encoding: Literal["target", "onehot", "frequency"] = "target",
    scale_numeric: bool = False,
    log_numeric: bool = False,
    dense: bool = False,
) -> Pipeline:
    """Build an unfitted pipeline for raw Cell2Cell inputs.

    `encoding` controls ServiceArea. Target encoding estimates smoothed
    churn rates in five internal training splits. Frequency encoding uses
    each area's share of training rows. One-hot encoding uses an indicator
    per area. Other categorical fields always use one-hot encoding.

    Numeric blanks use training medians and yes/no blanks use the most
    frequent training label. `log_numeric` compresses right-skewed
    nonnegative measurements. `scale_numeric` standardizes numeric values
    and their missing indicators. Set `dense=True` for models that require
    dense input.

    Fit this as part of a model pipeline inside each validation split.
    Preprocessing expects a DataFrame with the columns in PREDICTOR_COLUMNS.
    """
    if encoding == "target":
        area_encoder = TargetEncoder(
            target_type="binary",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        )
    elif encoding == "onehot":
        area_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    elif encoding == "frequency":
        area_encoder = _ServiceAreaFrequency()
    else:
        raise ValueError("encoding must be 'target', 'onehot', or 'frequency'")

    numeric_imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    binary_imputer = SimpleImputer(
        strategy="most_frequent", add_indicator=True, keep_empty_features=True,
    )
    numeric_steps = []
    if log_numeric:
        numeric_steps.append(("log", _SkewedLog()))
    numeric_steps.append(("impute", numeric_imputer))
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    columns = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_COLUMNS)),
            ("binary", binary_imputer, list(BINARY_COLUMNS)),
            ("service_area", area_encoder, [SERVICE_AREA]),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        sparse_threshold=0.0 if dense else 1.0,
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("clean", FunctionTransformer(_clean_inputs, feature_names_out=_input_feature_names)),
        ("encode", columns),
    ])
