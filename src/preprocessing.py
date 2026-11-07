"""Reusable preprocessing for the churn models.

Select raw groups, clean fixed invalid values, calculate derivatives, then fit
imputation, encoding and numerical transformations on training folds only.
Call fit_transform(X_train, y_train) for internally cross-fitted target encoding.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.utils.validation import check_is_fitted

from src.features import EngineerFeatures, get_feature_spec

MISSING_CATEGORY = "__MISSING__"
CATEGORICAL_COLUMNS = [
    "ServiceArea", "HandsetPrice", "IncomeGroup", "CreditRating",
    "PrizmCode", "Occupation", "MaritalStatus", "Homeownership",
]
AGE_ZERO_FLAGS = {"AgeHH1": "age1_zero", "AgeHH2": "age2_zero"}
INDICATOR_NAMES = {"PercChangeMinutes": "ChangeMissing"}
MISSING_MARKERS = ["NA", "", "empty"]
BINARY_COLUMNS = [
    "ChildrenInHH", "NewCellphoneUser", "NotNewCellphoneUser",
    "HandsetRefurbished", "HandsetWebCapable", "TruckOwner", "RVOwner",
    "OwnsMotorcycle", "OwnsComputer", "HasCreditCard", "BuysViaMailOrder",
    "RespondsToMailOffers", "OptOutMailings", "NonUSTravel", "MadeCallToRetentionTeam",
]

# Billing amounts, changes and ratios can be signed and never receive log1p.
# Only these nonnegative measurements can be selected by training-fold skewness.
LOG_CANDIDATES = [
    "OverageMinutes", "MonthsInService", "CurrentEquipmentDays", "Handsets",
    "HandsetModels", "UniqueSubs", "ActiveSubs", "AdjustmentsToCreditRating",
    "MonthlyMinutes", "DirectorAssistedCalls", "RoamingCalls", "ThreewayCalls",
    "ReceivedCalls", "OutboundCalls", "InboundCalls", "PeakCallsInOut",
    "OffPeakCallsInOut", "CallForwardingCalls", "CallWaitingCalls",
    "DroppedCalls", "BlockedCalls", "UnansweredCalls", "DroppedBlockedCalls",
    "CustomerCareCalls", "ReferralsMadeBySubscriber", "RetentionCalls",
    "RetentionOffersAccepted", "inactive_subs",
]


def indicator_name(column):
    """Return the agreed name of a numeric missingness indicator."""
    return INDICATOR_NAMES.get(column, f"{column}_missing")


def clean_raw(df):
    """Apply fixed cleaning to the available columns and return a copy.

The loader parses NA and empty cells. Reject any unparsed marker; preserve
recorded Unknown/Other states. No parameters are learned here.
"""
    out = df.copy()
    for column in out.select_dtypes(exclude="number").columns:
        leftover = out[column].isin(MISSING_MARKERS)
        if leftover.any():
            raise ValueError(f"{column} contains {int(leftover.sum())} unparsed missing markers")

    if "CurrentEquipmentDays" in out.columns:
        is_negative_age = out["CurrentEquipmentDays"] < 0
        out.loc[is_negative_age, "CurrentEquipmentDays"] = np.nan

    for column in CATEGORICAL_COLUMNS:
        if column not in out.columns:
            continue
        is_missing = out[column].isna()
        values = out[column].astype(str)
        if column == "IncomeGroup":
            # A missing numeric code can change the input dtype to float.
            # Keep observed integer codes consistent between training and validation.
            values = values.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
        values[is_missing] = MISSING_CATEGORY
        out[column] = values.astype(object)
    return out


class SelectRawFeatures(BaseEstimator, TransformerMixin):
    """Select an explicit list, keeping identifiers and outcomes outside the pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.transform(X)
        return self

    def transform(self, X):
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing raw predictors: {sorted(missing)}")
        return X[list(self.columns)].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns, dtype=object)


class MedianImputer(BaseEstimator, TransformerMixin):
    """Fit medians and missingness indicators on training rows only.

An entirely missing training column has no median and raises a clear error.
Use AgeImputer for household ages, whose medians exclude zero codes.
"""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.medians_ = X.median()
        empty_columns = list(self.medians_[self.medians_.isna()].index)
        if empty_columns:
            raise ValueError(f"No observed training values for: {empty_columns}")
        self.indicator_columns_ = [c for c in X.columns if X[c].isna().any()]
        return self

    def transform(self, X):
        check_is_fitted(self, "medians_")
        X = pd.DataFrame(X)[list(self.feature_names_in_)]
        indicators = {}
        for column in self.indicator_columns_:
            indicators[indicator_name(column)] = X[column].isna().astype(float)
        filled = X.astype(float).fillna(self.medians_)
        return pd.concat([filled, pd.DataFrame(indicators, index=X.index)], axis=1)

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        names += [indicator_name(c) for c in self.indicator_columns_]
        return np.asarray(names, dtype=object)


class AgeImputer(BaseEstimator, TransformerMixin):
    """Compare keeping recorded age zeros with replacing them by nonzero medians."""

    def __init__(self, zero_strategy="keep"):
        self.zero_strategy = zero_strategy

    def fit(self, X, y=None):
        if self.zero_strategy not in ("keep", "median"):
            raise ValueError("zero_strategy must be 'keep' or 'median'")
        X = pd.DataFrame(X)[list(AGE_ZERO_FLAGS)]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.medians_ = {}
        for column in X.columns:
            observed_ages = X.loc[X[column] > 0, column]
            if observed_ages.empty:
                raise ValueError(f"No positive training ages for {column}")
            self.medians_[column] = float(observed_ages.median())
        self.indicator_columns_ = [c for c in X.columns if X[c].isna().any()]
        return self

    def transform(self, X):
        check_is_fitted(self, "medians_")
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


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Encode Yes/No, using training modes and indicators for genuine missingness."""

    def _numeric(self, X):
        out = pd.DataFrame(index=X.index)
        for column in X.columns:
            unexpected = X[column].notna() & ~X[column].isin(["Yes", "No"])
            if unexpected.any():
                raise ValueError(f"Unexpected yes/no labels in {column}: {list(X.loc[unexpected, column].unique())}")
            out[column] = X[column].map({"Yes": 1.0, "No": 0.0})
        return out

    def fit(self, X, y=None):
        numeric = self._numeric(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.modes_ = {}
        for column in numeric.columns:
            modes = numeric[column].mode()
            if modes.empty:
                raise ValueError(f"No observed training labels for {column}")
            self.modes_[column] = float(modes.iloc[0])
        self.indicator_columns_ = [c for c in numeric.columns if numeric[c].isna().any()]
        return self

    def transform(self, X):
        check_is_fitted(self, "modes_")
        numeric = self._numeric(X[list(self.feature_names_in_)])
        out = numeric.fillna(self.modes_)
        for column in self.indicator_columns_:
            out[indicator_name(column)] = numeric[column].isna().astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        names += [indicator_name(c) for c in self.indicator_columns_]
        return np.asarray(names, dtype=object)


class ContinuousScaler(BaseEstimator, TransformerMixin):
    """Scale only named continuous columns; leave indicators unchanged."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.scaler_ = StandardScaler().fit(X[list(self.columns)])
        return self

    def transform(self, X):
        check_is_fitted(self, "scaler_")
        out = X[list(self.feature_names_in_)].astype(float).copy()
        out.loc[:, list(self.columns)] = self.scaler_.transform(out[list(self.columns)])
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()


class NumericPreparation(BaseEstimator, TransformerMixin):
    """Impute, optionally log skewed nonnegative columns, and scale continuous values.

Log eligibility uses only the training fold: an approved nonnegative measurement
with observed skewness greater than 1. Binary ContractWindow and all indicators
remain unscaled. Signed billing/change fields and ratios are never logged.
"""

    def __init__(self, scale=True, log_transform=True):
        self.scale = scale
        self.log_transform = log_transform

    def _logged(self, X):
        out = X.copy()
        for column in self.log_columns_:
            if (out[column] < 0).any():
                raise ValueError(f"Negative value in nonnegative log column {column}")
            out[column] = np.log1p(out[column])
        return out

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.imputer_ = MedianImputer().fit(X)
        self.log_columns_ = []
        if self.log_transform:
            for column in X.columns:
                observed = X[column].dropna()
                if column in LOG_CANDIDATES and (observed >= 0).all() and observed.skew() > 1:
                    self.log_columns_.append(column)
        values = self._logged(self.imputer_.transform(X))
        self.continuous_columns_ = [c for c in X.columns if c != "ContractWindow"]
        if self.scale and self.continuous_columns_:
            self.scaler_ = ContinuousScaler(self.continuous_columns_).fit(values)
        return self

    def transform(self, X):
        check_is_fitted(self, "imputer_")
        values = self._logged(self.imputer_.transform(X))
        if self.scale and self.continuous_columns_:
            values = self.scaler_.transform(values)
        return values

    def get_feature_names_out(self, input_features=None):
        return self.imputer_.get_feature_names_out()


def make_preprocessor(groups, model_kind="linear", age_strategy="keep",
                      service_area_encoding="target", log_transform=True,
                      include_retention=False):
    """Build an unfitted pipeline for a chosen combination of feature groups.

The modeling notebook chooses the baseline and addition order. This factory
supports each combination without inspecting outcomes or fitting parameters.
The returned matrix is sparse where one-hot encoding makes that useful.
"""
    if model_kind not in ("linear", "tree"):
        raise ValueError("model_kind must be 'linear' or 'tree'")
    if service_area_encoding not in ("target", "onehot"):
        raise ValueError("service_area_encoding must be 'target' or 'onehot'")
    spec = get_feature_spec(groups, include_retention, age_strategy)
    columns = spec["raw_features"] + spec["calculated_features"]
    numeric_columns = []
    binary_columns = []
    categorical_columns = []
    for column in columns:
        if column in BINARY_COLUMNS:
            binary_columns.append(column)
        elif column in CATEGORICAL_COLUMNS:
            categorical_columns.append(column)
        elif column not in AGE_ZERO_FLAGS:
            numeric_columns.append(column)

    is_linear = model_kind == "linear"
    transformations = []
    if numeric_columns:
        numeric = NumericPreparation(scale=is_linear, log_transform=is_linear and log_transform)
        transformations.append(("numeric", numeric, numeric_columns))

    age_columns = [c for c in AGE_ZERO_FLAGS if c in columns]
    if age_columns:
        age_steps = [("impute", AgeImputer(zero_strategy=age_strategy))]
        if is_linear:
            age_steps.append(("scale", ContinuousScaler(age_columns)))
        transformations.append(("ages", Pipeline(age_steps), age_columns))

    if binary_columns:
        transformations.append(("binary", BinaryEncoder(), binary_columns))

    # Keep ServiceArea separate from the other nominal features for both options.
    if "ServiceArea" in categorical_columns:
        categorical_columns.remove("ServiceArea")
        if service_area_encoding == "target":
            inner_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            area_encoder = TargetEncoder(target_type="binary", smooth="auto", cv=inner_folds)
        else:
            area_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        transformations.append(("service_area", area_encoder, ["ServiceArea"]))

    if categorical_columns:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        transformations.append(("categories", onehot, categorical_columns))

    prepared_columns = ColumnTransformer(
        transformations, remainder="drop", sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("select", SelectRawFeatures(spec["raw_features"])),
        ("clean", FunctionTransformer(clean_raw, feature_names_out="one-to-one")),
        ("engineer", EngineerFeatures(groups=spec["groups"])),
        ("columns", prepared_columns),
    ])
