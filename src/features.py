"""feature groups and values selected during exploration

these definitions describe the available inputs before encoding and imputation
the modeling notebook chooses the baseline and group order
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

# age-zero option selected in notebooks/modelling.py section 9; used as the default everywhere
DEFAULT_AGE_STRATEGY = 'median'

# main comparison: exploration groups without retention fields
MAIN_FEATURE_GROUPS = {}
for group_name, group_columns in FEATURE_GROUPS.items():
    main_columns = []
    for column in group_columns:
        if column not in RETENTION_COLUMNS:
            main_columns.append(column)
    MAIN_FEATURE_GROUPS[group_name] = main_columns

# indicators are listed here but created by the imputers
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
    """check an ordered group list without changing its order"""
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


def get_feature_spec(groups, include_retention=False, age_strategy=DEFAULT_AGE_STRATEGY):
    """describe raw inputs and eligible derived features for the selected groups

    missing indicators depend on the training fold and age flags depend on the strategy
    retention fields are an optional sensitivity check
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
        parent_groups = set(details['parent_groups'])
        parents_available = parent_groups.issubset(selected_groups)
        is_age_flag = details['created_by'] == 'AgeImputer'
        omit_age_flag = is_age_flag and age_strategy == 'median'
        if not parents_available or omit_age_flag:
            continue
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
    """build nested configurations from the order supplied by the modeling notebook

    each group must appear once across the baseline and additions
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
    """calculate selected features before fitting the imputer

    inputs must contain the cleaned raw predictors for the selected groups
    retention columns are optional and imputers create missing and age flags separately
    these formulas do not use learned statistics or target values
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
        """add eligible features and leave undefined values missing for imputation"""
        check_is_fitted(self, 'calculated_features_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError('EngineerFeatures requires a DataFrame with named raw columns')
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Raw columns and their order must match the fitted input')
        out = X.copy()

        for feature in self.calculated_features_:
            if feature == 'ContractWindow':
                tenure = out['MonthsInService'].astype(float)
                in_window = tenure.between(11, 12)
                values = in_window.astype(float)
                values = values.where(tenure.notna())
            elif feature == 'inactive_subs':
                total_subscribers = out['UniqueSubs'].astype(float)
                active_subscribers = out['ActiveSubs'].astype(float)
                values = total_subscribers - active_subscribers
            elif feature == 'RevenuePerMinute':
                # monthly revenue / (monthly minutes + 1)
                denominator = out['MonthlyMinutes'].astype(float) + 1
                denominator = denominator.where(denominator != 0)
                revenue = out['MonthlyRevenue'].astype(float)
                values = revenue / denominator
            elif feature == 'RelChangeMinutes':
                # relative change only when the monthly minutes are positive
                denominator = out['MonthlyMinutes'].astype(float)
                denominator = denominator.where(denominator > 0)
                change = out['PercChangeMinutes'].astype(float)
                values = change / denominator

            # non-finite results become missing values for the later imputer
            out[feature] = values.replace([np.inf, -np.inf], np.nan)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'calculated_features_')
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError('input_features must match the fitted raw columns')
        names = list(self.feature_names_in_) + self.calculated_features_
        return np.asarray(names, dtype=object)
