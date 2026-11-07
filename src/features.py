"""Feature groups and the engineered values selected during exploration.

These definitions describe which information is available, before encoding and imputation.
The modeling notebook chooses its baseline and group-addition order explicitly.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.eda_utils import FEATURE_GROUPS


RETENTION_COLUMNS = [
    'RetentionCalls',
    'RetentionOffersAccepted',
    'MadeCallToRetentionTeam',
]

# reuse the EDA membership without adding retention information to the main comparison
MAIN_FEATURE_GROUPS = {}
for group_name, group_columns in FEATURE_GROUPS.items():
    main_columns = []
    for column in group_columns:
        if column not in RETENTION_COLUMNS:
            main_columns.append(column)
    MAIN_FEATURE_GROUPS[group_name] = main_columns

# indicators are registered here, but their existing imputers produce them only once
ENGINEERED_FEATURES = {
    'ContractWindow': {
        'group': 'account_billing',
        'parents': ['MonthsInService'],
        'parent_groups': ['account_billing'],
        'created_by': 'EngineerFeatures',
    },
    'inactive_subs': {
        'group': 'account_billing',
        'parents': ['UniqueSubs', 'ActiveSubs'],
        'parent_groups': ['account_billing'],
        'created_by': 'EngineerFeatures',
    },
    'RevenuePerMinute': {
        'group': 'account_billing',
        'parents': ['MonthlyRevenue', 'MonthlyMinutes'],
        'parent_groups': ['account_billing', 'usage'],
        'created_by': 'EngineerFeatures',
    },
    'RelChangeMinutes': {
        'group': 'usage_change_retention',
        'parents': ['PercChangeMinutes', 'MonthlyMinutes'],
        'parent_groups': ['usage_change_retention', 'usage'],
        'created_by': 'EngineerFeatures',
    },
    'ChangeMissing': {
        'group': 'usage_change_retention',
        'parents': ['PercChangeMinutes'],
        'parent_groups': ['usage_change_retention'],
        'created_by': 'MedianImputer',
    },
    'age1_zero': {
        'group': 'demographics_region',
        'parents': ['AgeHH1'],
        'parent_groups': ['demographics_region'],
        'created_by': 'AgeImputer',
    },
    'age2_zero': {
        'group': 'demographics_region',
        'parents': ['AgeHH2'],
        'parent_groups': ['demographics_region'],
        'created_by': 'AgeImputer',
    },
}


def _checked_groups(groups, allow_empty=False):
    """Check an ordered group list without changing its order."""
    if not isinstance(groups, (list, tuple)):
        raise ValueError('Feature groups must be an ordered list or tuple')
    selected_groups = list(groups)
    if not selected_groups and not allow_empty:
        raise ValueError('Select at least one feature group')
    if len(selected_groups) != len(set(selected_groups)):
        raise ValueError('A feature group cannot be selected more than once')
    for group_name in selected_groups:
        if group_name not in MAIN_FEATURE_GROUPS:
            raise ValueError(f'Unknown feature group: {group_name}')
    return selected_groups


def get_feature_spec(groups, include_retention=False, age_strategy='keep'):
    """Describe raw inputs and eligible derivatives for selected groups.

    This is a feature specification, not the final encoded column list. Missingness
    indicators depend on the fitted training fold; zero flags depend on the age option.
    Retention fields are a separate, explicit sensitivity option.
    """
    selected_groups = _checked_groups(groups)
    if age_strategy not in ('keep', 'median'):
        raise ValueError("age_strategy must be 'keep' or 'median'")

    raw_features = []
    for group_name in selected_groups:
        raw_features.extend(MAIN_FEATURE_GROUPS[group_name])
    if include_retention:
        raw_features.extend(RETENTION_COLUMNS)

    engineered_features = []
    calculated_features = []
    for feature, details in ENGINEERED_FEATURES.items():
        parents_available = set(details['parent_groups']).issubset(selected_groups)
        is_omitted_age_flag = details['created_by'] == 'AgeImputer' and age_strategy == 'median'
        if parents_available and not is_omitted_age_flag:
            engineered_features.append(feature)
            if details['created_by'] == 'EngineerFeatures':
                calculated_features.append(feature)

    return {
        'groups': selected_groups,
        'raw_features': raw_features,
        'calculated_features': calculated_features,
        'engineered_features': engineered_features,
    }


def build_feature_configurations(baseline_groups, addition_order):
    """Build nested configurations from an order supplied by the modeling notebook.

    Every group must appear exactly once across the baseline and additions. This function
    deliberately supplies no default baseline or ranking of the groups.
    """
    baseline = _checked_groups(baseline_groups)
    additions = _checked_groups(addition_order, allow_empty=True)
    all_groups = _checked_groups(baseline + additions)
    if set(all_groups) != set(MAIN_FEATURE_GROUPS):
        raise ValueError('The baseline and additions must cover all six feature groups')
    if not additions:
        return {'full': baseline.copy()}

    configurations = {'baseline': baseline.copy()}
    selected_groups = baseline.copy()
    for position, group_name in enumerate(additions):
        selected_groups.append(group_name)
        configuration_name = 'add_' + group_name
        if position == len(additions) - 1:
            configuration_name = 'full'
        configurations[configuration_name] = selected_groups.copy()
    return configurations


class EngineerFeatures(BaseEstimator, TransformerMixin):
    """Calculate the four selected derivatives before fold-fitted imputation.

    Input must contain the selected groups' cleaned raw predictors. Retention columns may
    also be supplied for sensitivity analysis. The imputers create ChangeMissing and the
    age-zero flags separately. No statistics or target values are used in these formulas.
    """

    def __init__(self, groups):
        self.groups = groups

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('EngineerFeatures requires a DataFrame with named raw columns')
        if not X.columns.is_unique:
            raise ValueError('Raw feature columns must be unique')

        specification = get_feature_spec(self.groups)
        required_columns = specification['raw_features']
        missing_columns = [column for column in required_columns if column not in X.columns]
        if missing_columns:
            raise ValueError(f'Missing raw predictors: {missing_columns}')
        allowed_columns = required_columns + RETENTION_COLUMNS
        unexpected_columns = [column for column in X.columns if column not in allowed_columns]
        if unexpected_columns:
            raise ValueError(f'Unexpected raw predictors: {unexpected_columns}')

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.calculated_features_ = specification['calculated_features']
        return self

    def transform(self, X):
        check_is_fitted(self, 'calculated_features_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError('EngineerFeatures requires a DataFrame with named raw columns')
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Raw columns and their order must match the fitted input')
        out = X.copy()

        for feature in self.calculated_features_:
            if feature == 'ContractWindow':
                tenure = out['MonthsInService'].astype(float)
                values = tenure.between(11, 12).astype(float)
                values = values.where(tenure.notna())
            elif feature == 'inactive_subs':
                values = out['UniqueSubs'].astype(float) - out['ActiveSubs'].astype(float)
            elif feature == 'RevenuePerMinute':
                denominator = out['MonthlyMinutes'].astype(float) + 1
                denominator = denominator.where(denominator != 0)
                values = out['MonthlyRevenue'].astype(float) / denominator
            elif feature == 'RelChangeMinutes':
                denominator = out['MonthlyMinutes'].astype(float)
                denominator = denominator.where(denominator > 0)
                values = out['PercChangeMinutes'].astype(float) / denominator

            # undefined results stay missing for the later imputer
            out[feature] = values.replace([np.inf, -np.inf], np.nan)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'calculated_features_')
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError('input_features must match the fitted raw columns')
        names = list(self.feature_names_in_) + self.calculated_features_
        return np.asarray(names, dtype=object)
