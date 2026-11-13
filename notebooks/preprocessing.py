# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocessing
#
# This chapter prepares the telecom customer data for churn modeling. Building on the
# exploratory analysis, we handle missing and invalid values, encode categories, construct
# engineered features, and define reusable feature sets and preprocessing pipelines.
#
# We use the predefined training partition throughout. Transformations that learn from data
# are fitted within training folds, preserving the independence of validation observations.
# The final checks examine the resulting representations across five stratified folds.

# %%
# locate the repository from its root or the notebooks folder
from pathlib import Path
import sys

project_root = Path.cwd()
shared_loader_file = project_root / 'src' / 'data.py'
if not shared_loader_file.is_file():
    project_root = project_root.parent
    shared_loader_file = project_root / 'src' / 'data.py'

# stop with a clear message when the notebook is opened outside the project
if not shared_loader_file.is_file():
    raise FileNotFoundError('Open this notebook from the repository root or notebooks folder')

# make the shared project modules available
sys.path.insert(0, str(project_root))

# import the shared data loader and table display tools
import pandas as pd
from IPython.display import display
from src.data import load_train

# show complete feature lists in the summary tables
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 200)

# %% [markdown]
# ## 1. Dataset and target variable
#
# The predefined split separates customers into training and test partitions. We verify that
# the customer identifiers are unique and the partitions do not overlap, then inspect the
# training target distribution. The test outcomes remain reserved for final evaluation.

# %%
# load the predefined training partition
train = load_train()

# read the committed split assignments
split_file = project_root / 'dataset' / 'split.csv'
split_info = pd.read_csv(split_file)

# select the customer ids assigned to training
is_training_customer = split_info['split'] == 'train'
training_customer_values = split_info.loc[is_training_customer, 'CustomerID']
training_ids = set(training_customer_values)

# select the customer ids assigned to testing without inspecting their outcomes
is_test_customer = split_info['split'] == 'test'
test_customer_values = split_info.loc[is_test_customer, 'CustomerID']
test_ids = set(test_customer_values)

# check that the assignments use unique ids and the two expected partition labels
observed_split_labels = set(split_info['split'])
assert split_info['CustomerID'].is_unique
assert observed_split_labels == {'train', 'test'}

# verify that the two partitions do not overlap and the loader returns all training ids
actual_training_ids = set(train['CustomerID'])
assert training_ids.isdisjoint(test_ids)
assert actual_training_ids == training_ids

# show the partition sizes and training churn distribution
training_churn_distribution = train['churn'].value_counts(normalize=True)
print('Assigned train/test customers:', len(training_ids), len(test_ids))
print('Training churn distribution:')
print(training_churn_distribution)

# %% [markdown]
# ## 2. Leakage prevention
#
# We exclude the identifier and target from predictors. The three retention fields remain
# outside the main set because their prediction-time availability is uncertain; we preserve
# them in a separate sensitivity representation.

# %%
# list the retention fields kept outside the main representation
retention_columns = [
    'RetentionCalls',
    'RetentionOffersAccepted',
    'MadeCallToRetentionTeam',
]

# collect the predictor columns without the identifier or targets
excluded_columns = ['CustomerID', 'Churn', 'churn']
feature_columns = []
for column in train.columns:
    if column not in excluded_columns:
        feature_columns.append(column)

# define the main representation without retention information
main_features = []
for column in feature_columns:
    if column not in retention_columns:
        main_features.append(column)

# preserve the complete predictor list for the sensitivity comparison
with_retention_features = feature_columns.copy()

# check the original feature-set sizes
assert len(main_features) == 53
assert len(with_retention_features) == 56

# summarize the two representations
feature_summary = pd.DataFrame({
    'feature_set': ['main', 'with retention'],
    'raw_predictors': [len(main_features), len(with_retention_features)],
})
display(feature_summary)

# %% [markdown]
# ## 3. Missing values and invalid entries
#
# Negative `CurrentEquipmentDays` values are treated as missing because an equipment age
# cannot be negative. Numeric gaps are filled with training-fold medians and accompanied by
# per-column missingness indicators. Genuine nonbinary categorical gaps use `__MISSING__`,
# preserving recorded Unknown/Other values as distinct categories.
#
# The code lives in `src/preprocessing.py` so the pipelines and the modeling notebook can reuse it:
#
# - `clean_raw()` applies the fixed rules. It learns nothing from the data, so it can run before the
#   cross-validation split.
# - `MedianImputer` learns medians, so it must be fitted on training folds only. In the final
#   pipeline it runs after the engineered features (section 6), because derivatives are computed
#   before imputation.
#
# After `clean_raw()` the numeric columns therefore still contain missing values; only the imputer
# fills them.
#
# In the data, missing values are stored as `NA` or empty cells, which `read_csv` already reads as
# missing; the marker `empty` does not occur. The missingness indicator for `PercChangeMinutes` is
# named `ChangeMissing` and belongs to the usage-change feature group.

# %%
# apply the fixed cleaning rules
from src.preprocessing import (
    AGE_ZERO_FLAGS,
    CATEGORICAL_COLUMNS,
    MISSING_CATEGORY,
    AgeImputer,
    MedianImputer,
    clean_raw,
)

clean = clean_raw(train)

# compare missing values before and after cleaning; numeric gaps stay until the imputer runs
missing_summary = pd.DataFrame({
    'before_cleaning': train.isna().sum(),
    'after_cleaning': clean.isna().sum(),
})
has_missing = missing_summary.sum(axis=1) > 0
display(missing_summary[has_missing])

# check the cleaning rules
negative_equipment_days = int((train['CurrentEquipmentDays'] < 0).sum())
print('Negative CurrentEquipmentDays turned into missing:', negative_equipment_days)
assert (clean['CurrentEquipmentDays'] < 0).sum() == 0
assert clean[CATEGORICAL_COLUMNS].isna().sum().sum() == 0
print('ServiceArea marked as', MISSING_CATEGORY, ':', int((clean['ServiceArea'] == MISSING_CATEGORY).sum()))

# %%
# fit the median imputer on one training fold and apply it to the matching validation fold
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fit_rows, check_rows = next(folds.split(clean, clean['churn']))
fit_part = clean.iloc[fit_rows]
check_part = clean.iloc[check_rows]

# numeric predictors of the main feature set; the age columns are handled in section 4
numeric_columns = []
for column in main_features:
    is_numeric = pd.api.types.is_numeric_dtype(clean[column])
    if is_numeric and column not in AGE_ZERO_FLAGS:
        numeric_columns.append(column)

median_imputer = MedianImputer().fit(fit_part[numeric_columns])
imputed_check = median_imputer.transform(check_part[numeric_columns])

# the medians come from the training fold, and no missing values remain afterwards
assert median_imputer.medians_.equals(fit_part[numeric_columns].median())
assert imputed_check.isna().sum().sum() == 0
assert list(imputed_check.columns) == list(median_imputer.get_feature_names_out())

# show the missing values in the validation fold before and after imputation
indicator_columns = [c for c in imputed_check.columns if c not in numeric_columns]
print('Numeric columns:', len(numeric_columns), '| indicators added:', len(indicator_columns))
print('Missing values in the validation fold:',
      int(check_part[numeric_columns].isna().sum().sum()), 'before imputation,',
      int(imputed_check.isna().sum().sum()), 'after')

# list each indicator with the column it describes and its training-fold median
indicator_table = pd.DataFrame({
    'column': median_imputer.indicator_columns_,
    'indicator': indicator_columns,
    'training_fold_median': median_imputer.medians_[median_imputer.indicator_columns_].values,
})
display(indicator_table)

# %% [markdown]
# ## 4. Special values and categories
#
# `IncomeGroup` = 0 remains a separate category, and `HandsetPrice` is categorical.
# Recorded zeros in `AgeHH1`/`AgeHH2` have two representations: (a) retain the zeros and add
# `age1_zero`/`age2_zero`, or (b) replace them with training-fold medians of nonzero ages,
# without zero indicators. Option (a) is the default and the tie-break choice for model
# comparisons. Genuine missing ages use nonzero training-fold medians and retain their
# missingness indicators in both representations.
#
# `clean_raw()` already stores `IncomeGroup` and `HandsetPrice` as categories, keeping
# `IncomeGroup` = 0 and "Unknown" as their own values. `AgeImputer` handles the ages: its
# `zero_strategy` switches between option (a) `"keep"` and option (b) `"median"`.

# %%
# check the categorical special values
print('IncomeGroup categories:', sorted(clean['IncomeGroup'].unique()))
print('HandsetPrice = Unknown:', int((clean['HandsetPrice'] == 'Unknown').sum()))
assert (clean['IncomeGroup'] == '0').sum() == (train['IncomeGroup'] == 0).sum()

# %%
# fit both age options on the same training fold and apply them to the validation fold
age_columns = list(AGE_ZERO_FLAGS)
option_labels = {'keep': '(a) keep', 'median': '(b) median'}
option_rows = []
option_outputs = {}
for strategy, label in option_labels.items():
    age_imputer = AgeImputer(zero_strategy=strategy).fit(fit_part[age_columns])
    ages = age_imputer.transform(check_part[age_columns])

    # medians use nonzero ages of the training fold only
    for column in age_columns:
        nonzero = fit_part.loc[fit_part[column] > 0, column]
        assert age_imputer.medians_[column] == nonzero.median()
    assert ages.isna().sum().sum() == 0
    assert list(ages.columns) == list(age_imputer.get_feature_names_out())

    # in option (a), the zero flags match the recorded zeros
    if strategy == 'keep':
        for column, flag in AGE_ZERO_FLAGS.items():
            assert ages[flag].sum() == (check_part[column] == 0).sum()

    # collect the results for the summary tables
    option_rows.append({
        'option': label,
        'AgeHH1 median': age_imputer.medians_['AgeHH1'],
        'AgeHH2 median': age_imputer.medians_['AgeHH2'],
        'zeros left in validation fold': int((ages[age_columns] == 0).sum().sum()),
        'output columns': ages.shape[1],
    })
    option_outputs[label] = list(ages.columns)

# compare the two options: training-fold medians and zeros left after imputation
display(pd.DataFrame(option_rows).set_index('option'))

# show which output columns each option produces
all_output_columns = list(dict.fromkeys(option_outputs['(a) keep'] + option_outputs['(b) median']))
column_table = pd.DataFrame(
    {label: ['yes' if c in columns else '-' for c in all_output_columns]
     for label, columns in option_outputs.items()},
    index=all_output_columns,
)
column_table.index.name = 'output column'
display(column_table)

# %% [markdown]
# ## 5. Categorical encoding
#
# Yes/no features are encoded as 0/1, with missing values imputed using
# training-fold modes. Other categorical features, including `IncomeGroup`
# and `CreditRating`, use one-hot encoding.
#
# `ServiceArea` uses smoothed target encoding with internal 5-fold cross-fitting
# to prevent target leakage. One-hot encoding is retained for comparison.
# All encoders are fitted on training data only.

# %%
# encode yes/no values using training-fold statistics
from src.preprocessing import BINARY_COLUMNS, BinaryEncoder
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

# binary encoding: Yes=1, No=0
binary_columns = [c for c in main_features if c in BINARY_COLUMNS]
binary_encoder = BinaryEncoder()
encoded_training_binary = binary_encoder.fit_transform(fit_part[binary_columns])
encoded_binary = binary_encoder.transform(check_part[binary_columns])

# one-hot encoding for nominal features
nominal_columns = [
    c for c in main_features
    if c in CATEGORICAL_COLUMNS and c != "ServiceArea"
]
nominal_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
encoded_training_nominal = nominal_encoder.fit_transform(fit_part[nominal_columns])
encoded_nominal = nominal_encoder.transform(check_part[nominal_columns])

# serviceArea: cross-fitted target encoding to prevent leakage
inner_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
area_encoder = TargetEncoder(target_type="binary", smooth="auto", cv=inner_folds)
encoded_training_areas = area_encoder.fit_transform(
    fit_part[["ServiceArea"]], fit_part["churn"]
)
encoded_validation_areas = area_encoder.transform(check_part[["ServiceArea"]])

# serviceArea: one-hot alternative
area_onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
onehot_training_areas = area_onehot.fit_transform(fit_part[["ServiceArea"]])
onehot_validation_areas = area_onehot.transform(check_part[["ServiceArea"]])

# encoding summary
encoding_rows = [
    {"representation": "yes/no", "input_columns": len(binary_columns),
     "output_columns": encoded_binary.shape[1]},
    {"representation": "nominal one-hot", "input_columns": len(nominal_columns),
     "output_columns": encoded_nominal.shape[1]},
    {"representation": "ServiceArea target", "input_columns": 1,
     "output_columns": encoded_validation_areas.shape[1]},
    {"representation": "ServiceArea one-hot", "input_columns": 1,
     "output_columns": onehot_validation_areas.shape[1]},
]
display(pd.DataFrame(encoding_rows))

# %% [markdown]
# ## 6. Engineered features
#
# The exploratory analysis selected seven engineered features. Four are calculated from
# cleaned raw values before imputation; the remaining three are missingness or age-zero
# indicators produced during imputation. Each feature retains its raw-parent dependencies.

# %%
# calculate the selected derivatives from cleaned, unimputed raw inputs
from src.features import MAIN_FEATURE_GROUPS, EngineerFeatures

all_groups = list(MAIN_FEATURE_GROUPS)
feature_engineer = EngineerFeatures(groups=all_groups)
engineered = feature_engineer.fit_transform(clean[main_features])
calculated_features = feature_engineer.calculated_features_

# verify the formulas without changing the raw predictors
pd.testing.assert_frame_equal(engineered[main_features], clean[main_features])
assert engineered['inactive_subs'].equals(
    clean['UniqueSubs'].astype(float) - clean['ActiveSubs'].astype(float)
)
assert engineered['ContractWindow'].dropna().isin([0, 1]).all()
undefined_change = clean['MonthlyMinutes'].isna() | (clean['MonthlyMinutes'] <= 0)
assert engineered.loc[undefined_change, 'RelChangeMinutes'].isna().all()

# inspect the four calculated features; the imputers create the other three flags
engineered_rows = []
for column in calculated_features:
    engineered_rows.append({
        'feature': column,
        'missing_before_imputation': int(engineered[column].isna().sum()),
        'minimum': engineered[column].min(),
        'maximum': engineered[column].max(),
    })
display(pd.DataFrame(engineered_rows))

# %% [markdown]
# ## 7. Feature sets for modeling
#
# The six feature groups preserve the information categories established during exploration.
# The full representation combines all groups and their eligible engineered features. Group
# combinations also support incremental comparisons: a derivative is available only when all
# of its parent groups are present.

# %%
# import the shared definitions and the helper used later by modeling
from src.features import (
    DEFAULT_AGE_STRATEGY,
    ENGINEERED_FEATURES,
    RETENTION_COLUMNS,
    build_feature_configurations,
    get_feature_spec,
)

# %% [markdown]
# ### 7.1 Raw feature groups
#
# We reuse the six EDA groups. The three retention fields stay outside the main
# representation and are included only in the separate sensitivity option.

# %%
# show the group memberships without assigning an evaluation order
group_rows = []
group_feature_sets = {}
for group_name, group_columns in MAIN_FEATURE_GROUPS.items():
    group_feature_sets[group_name] = get_feature_spec([group_name])
    group_rows.append({
        'group': group_name,
        'raw_predictors': len(group_columns),
        'features': ', '.join(group_columns),
    })
display(pd.DataFrame(group_rows))

# %% [markdown]
# ### 7.2 Engineered features and parent groups
#
# A derivative enters only when all of its parent groups are present. Missingness
# indicators follow the same rule. `ChangeMissing` is produced by the numeric imputer
# when that training fold has missing minutes-change values. The age-zero flags are
# produced only with the keep-zeros option. These columns are not created twice.

# %%
# record the source of each engineered feature and its dependencies
engineered_group_rows = []
for feature, details in ENGINEERED_FEATURES.items():
    engineered_group_rows.append({
        'feature': feature,
        'group': details['group'],
        'raw_parents': ', '.join(details['parents']),
        'required_groups': ', '.join(details['parent_groups']),
        'created_by': details['created_by'],
    })
display(pd.DataFrame(engineered_group_rows))

# %% [markdown]
# ### 7.3 Full feature sets and incremental configurations
#
# The main set contains 53 raw predictors; the retention sensitivity set contains 56.
# The modelling chapter selected the median-age option, which is now the default: it omits
# the two age-zero flags, leaving five engineered features. The keep-zeros option adds the
# flags back, giving seven. Actual output columns also depend on fitted missingness
# indicators and categorical encoding.
#
# `build_feature_configurations(baseline_groups, addition_order)` constructs nested group
# lists ending in the full set. Each list can be passed to `get_feature_spec` and
# `make_preprocessor`. The baseline and addition order are supplied for each evaluation;
# feature preparation itself imposes no ranking on the groups.

# %%
# define the complete reference sets without fitting any transformation
full_feature_set = get_feature_spec(all_groups)
retention_feature_set = get_feature_spec(all_groups, include_retention=True)
keep_age_feature_set = get_feature_spec(all_groups, age_strategy='keep')
feature_sets = {
    'full': full_feature_set,
    'with_retention': retention_feature_set,
}

# verify coverage, exclusions and the optional age flags
assert set(full_feature_set['raw_features']) == set(main_features)
assert set(retention_feature_set['raw_features']) == set(with_retention_features)
assert len(full_feature_set['raw_features']) == 53
assert len(retention_feature_set['raw_features']) == 56
assert len(full_feature_set['engineered_features']) == 5
assert len(keep_age_feature_set['engineered_features']) == 7
assert set(RETENTION_COLUMNS).isdisjoint(full_feature_set['raw_features'])

feature_set_rows = []
for name, specification in feature_sets.items():
    names = specification['raw_features'] + specification['engineered_features']
    assert len(names) == len(set(names))
    assert set(['CustomerID', 'Churn', 'churn', 'split', 'fold']).isdisjoint(names)
    feature_set_rows.append({
        'feature_set': name,
        'raw_predictors': len(specification['raw_features']),
        'planned_engineered_features': ', '.join(specification['engineered_features']),
    })
display(pd.DataFrame(feature_set_rows))

# check that cross-group features require both groups
billing_only = get_feature_spec(['account_billing'])
billing_and_usage = get_feature_spec(['account_billing', 'usage'])
change_only = get_feature_spec(['usage_change_retention'])
change_and_usage = get_feature_spec(['usage_change_retention', 'usage'])
assert 'RevenuePerMinute' not in billing_only['engineered_features']
assert 'RevenuePerMinute' in billing_and_usage['engineered_features']
assert 'RelChangeMinutes' not in change_only['engineered_features']
assert 'RelChangeMinutes' in change_and_usage['engineered_features']

# %% [markdown]
# ## 8. Numerical transformations
#
# Logistic Regression uses training-fold scaling of continuous numerical columns, while
# tree-based models receive unscaled numerical values. Extreme observations are retained
# without clipping.
#
# For Logistic Regression, the default representation applies `log1p` to eligible, highly
# skewed nonnegative measurements. A second representation omits the log transformation,
# allowing its effect to be assessed during model evaluation.

# %%
# fit numerical alternatives using the same training and validation fold
from src.preprocessing import NumericPreparation, indicator_name

numeric_feature_columns = []
for column in engineered.columns:
    if pd.api.types.is_numeric_dtype(engineered[column]) and column not in AGE_ZERO_FLAGS:
        numeric_feature_columns.append(column)

numeric_fit = engineered.iloc[fit_rows][numeric_feature_columns]
numeric_check = engineered.iloc[check_rows][numeric_feature_columns]
numerical_options = [
    ('linear with log1p', True, True),
    ('linear without log1p', True, False),
    ('tree', False, False),
]
numerical_rows = []
for label, scale, use_log in numerical_options:
    preparation = NumericPreparation(scale=scale, log_transform=use_log)
    preparation.fit(numeric_fit)
    prepared = preparation.transform(numeric_check)
    assert prepared.isna().sum().sum() == 0
    assert prepared['ContractWindow'].isin([0, 1]).all()
    for column in preparation.imputer_.indicator_columns_:
        assert prepared[indicator_name(column)].isin([0, 1]).all()
    numerical_rows.append({
        'option': label,
        'output_columns': prepared.shape[1],
        'logged_columns': ', '.join(preparation.log_columns_) or 'none',
    })
display(pd.DataFrame(numerical_rows))

# %% [markdown]
# Log eligibility is learned within each training fold: an eligible nonnegative
# measurement must have skewness greater than 1. Signed billing amounts, changes
# and ratios are never logged. Only continuous values are scaled; binary features
# and missingness indicators keep their 0/1 values. No values are clipped.

# %% [markdown]
# ## 9. Preprocessing pipelines and validation
#
# Reusable pipelines combine feature selection, cleaning, feature engineering and numerical
# preparation. Imputation, encoding, scaling and other learned transformations are fitted
# only on training folds during cross-validation.

# %%
# create reusable, unfitted pipelines for the complete main and sensitivity sets
from src.preprocessing import make_preprocessor

linear_preprocessor = make_preprocessor(all_groups, model_kind='linear')
tree_preprocessor = make_preprocessor(all_groups, model_kind='tree')
retention_preprocessor = make_preprocessor(all_groups, include_retention=True)

# %% [markdown]
# ### 9.1 Training-fold validation
#
# The checks below run each preprocessing alternative through the same five training
# folds. They verify finite values, row counts, feature names and group membership.
# They also exercise each group on its own and the separate retention representation.
# These are transformation checks, not model-score comparisons or an addition schedule.
#
# Training uses `fit_transform(X_fit, y_fit)` so ServiceArea target encoding is
# cross-fitted internally. Validation uses only `transform(X_check)`. When modeling,
# place this preprocessor inside the model pipeline so each cross-validation fold fits
# its own transformations. Do not export one globally fitted matrix for model CV.

# %%
from scipy import sparse
import numpy as np

# enumerate alternatives without choosing them by validation outcomes
validation_options = []
for model_kind in ['linear', 'tree']:
    log_options = [True, False] if model_kind == 'linear' else [False]
    for age_strategy in ['keep', 'median']:
        for area_encoding in ['target', 'onehot']:
            for use_log in log_options:
                validation_options.append({
                    'name': f'full / {model_kind} / {age_strategy} / {area_encoding} / log={use_log}',
                    'groups': all_groups,
                    'model_kind': model_kind,
                    'age_strategy': age_strategy,
                    'service_area_encoding': area_encoding,
                    'log_transform': use_log,
                    'include_retention': False,
                })

# verify that individual groups and the retention option work independently
for model_kind in ['linear', 'tree']:
    for group_name in all_groups:
        validation_options.append({
            'name': f'{group_name} only / {model_kind}',
            'groups': [group_name],
            'model_kind': model_kind,
        })
    validation_options.append({
        'name': f'with retention / {model_kind}',
        'groups': all_groups,
        'model_kind': model_kind,
        'include_retention': True,
    })

# keep targets separate; the pipeline explicitly selects only its allowed raw inputs
X_train = train.drop(columns=['Churn', 'churn'])
y_train = train['churn']
shared_folds = list(folds.split(X_train, y_train))
validation_rows = []

for option in validation_options:
    settings = option.copy()
    option_name = settings.pop('name')
    for fold_number, (training_rows, validation_rows_in_fold) in enumerate(shared_folds, start=1):
        preprocessor = make_preprocessor(**settings)
        fit_values = preprocessor.fit_transform(
            X_train.iloc[training_rows], y_train.iloc[training_rows],
        )
        check_values = preprocessor.transform(X_train.iloc[validation_rows_in_fold])
        output_names = list(preprocessor.get_feature_names_out())

        # training and validation use the same fitted schema within each fold
        assert fit_values.shape == (len(training_rows), len(output_names))
        assert check_values.shape == (len(validation_rows_in_fold), len(output_names))
        assert len(output_names) == len(set(output_names))
        for values in [fit_values, check_values]:
            entries = values.data if sparse.issparse(values) else values
            assert np.isfinite(entries).all()

        # raw selection excludes outcomes, identifiers and unintended groups
        expected = get_feature_spec(settings['groups'], settings.get('include_retention', False))
        selected = list(preprocessor.named_steps['select'].get_feature_names_out())
        assert selected == expected['raw_features']
        assert set(['CustomerID', 'Churn', 'churn', 'split', 'fold']).isdisjoint(output_names)
        if not settings.get('include_retention', False):
            assert set(RETENTION_COLUMNS).isdisjoint(output_names)

        # no flag is generated twice, and median-age handling omits zero flags
        assert output_names.count('ChangeMissing') <= 1
        if settings.get('age_strategy', DEFAULT_AGE_STRATEGY) == 'median':
            assert set(AGE_ZERO_FLAGS.values()).isdisjoint(output_names)

        validation_rows.append({
            'configuration': option_name,
            'fold': fold_number,
            'training_rows': fit_values.shape[0],
            'validation_rows': check_values.shape[0],
            'output_columns': len(output_names),
            'passed': True,
        })

# summarize all fold checks; encoded column counts may differ between fitted folds
pipeline_checks = pd.DataFrame(validation_rows)
validation_summary = pipeline_checks.groupby('configuration', sort=False).agg(
    folds_checked=('fold', 'count'),
    minimum_columns=('output_columns', 'min'),
    maximum_columns=('output_columns', 'max'),
    all_passed=('passed', 'all'),
)
assert validation_summary['folds_checked'].eq(5).all()
assert validation_summary['all_passed'].all()
display(validation_summary)
print('Passed preprocessing checks:', len(pipeline_checks))

# %% [markdown]
# `make_preprocessor` provides the validated transformations for any selected combination
# of feature groups. Combined with an estimator, the pipeline fits preprocessing separately
# within each training fold and applies the learned transformations to held-out observations.
# The test partition remains reserved for final model evaluation.
