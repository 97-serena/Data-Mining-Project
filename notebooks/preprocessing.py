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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocessing
#
# The sections below follow the team's preprocessing task list, with the owner of each task.
# Please work in your own section. Sections 1 and 2 contain code; sections 3 to 9 are still to
# be implemented. Implementation details follow `data_exploration.py` section 8, except where a
# section says **Open question**.
#
# We use the repository's committed split and shared loader. The model baseline, group-addition
# order, training and evaluation belong to the later modeling notebook.

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

# %% [markdown]
# ## 1. Prepare dataset and target variable
#
# **Owner:** Shiqi
#
# We verify the predefined customer assignments and the training partition. We do not recreate
# the split or inspect test outcomes while preparing the workflow.

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
# ## 2. Handle potential data leakage
#
# **Owner:** Paola
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
# ## 3. Handle missing values and invalid entries
#
# **Owner:** Shiqi
#
# Standardize missing-value markers (`NA`, `empty`) and turn negative `CurrentEquipmentDays` into
# missing. Use training-fold median imputation with per-column missingness indicators for numeric
# features. Use `__MISSING__` for genuine nonbinary categorical missing values, and keep recorded
# Unknown/Other values separate.

# %%
# TODO (Shiqi)

# %% [markdown]
# ## 4. Handle special values and categories
#
# **Owner:** Shiqi
#
# Retain `IncomeGroup` = 0 as a separate category. Retain recorded zeros in `AgeHH1`/`AgeHH2` and
# add `age1_zero`/`age2_zero` before imputation; genuine missing ages stay distinct. Treat
# `HandsetPrice` as categorical.

# %%
# TODO (Shiqi)

# %% [markdown]
# ## 5. Encode categorical features
#
# **Owner:** Yan-Ru
#
# Encode yes/no features as 0/1 and apply one-hot encoding to other categorical features,
# including `IncomeGroup` and `CreditRating`.
#
# **Open question:** how to encode `ServiceArea` (one-hot or smoothed target encoding with internal
# cross-fitting). `data_exploration.py` section 8 currently says one-hot.

# %%
# TODO (Yan-Ru)

# %% [markdown]
# ## 6. Implement feature engineering
#
# **Owner:** Kaustubh
#
# Assign each engineered feature to its corresponding feature group and implement it. The seven
# selected features are defined in `data_exploration.py` section 8.

# %%
# TODO (Kaustubh)

# %% [markdown]
# ## 7. Prepare feature sets for modeling
#
# **Owner:** Yannis
#
# Implement the feature-group definitions established during EDA, including the planned
# engineered features. Define the full feature set and the configurations that add feature groups
# incrementally. A derivative is available only when all of its parent groups are present.

# %%
# TODO (Yannis)

# %% [markdown]
# ## 8. Implement numerical transformations
#
# **Owner:** All
#
# Implement training-fold scaling of continuous numerical columns for Logistic Regression. Keep
# numerical features unscaled for tree-based models. Do not delete rows or clip values.
#
# **Open question:** whether Logistic Regression uses `log1p` on highly skewed nonnegative columns.
# `data_exploration.py` section 8 currently says no log transformation.

# %%
# TODO (All)

# %% [markdown]
# ## 9. Build and validate preprocessing pipelines
#
# **Owner:** All
#
# Combine the transformations into reproducible pipelines. Ensure imputation, encoding, scaling
# and other learned transformations are fitted only on training folds during cross-validation.

# %%
# TODO (All)
