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
# # Random Forest
#
# We predict churn, when a customer leaves, using information available when we decide
# whom to contact. We will use churn risk and customer value to prioritize outreach.
# We use a random forest to estimate churn by averaging predictions from many trees.

# %%
from pathlib import Path
import sys

from IPython.display import display
from sklearn.ensemble import RandomForestClassifier

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.data import RANDOM_STATE, load_train
from src.evaluation import compare_preprocessing
from src.preprocessing import (
    combine_dense_feature_steps,
    frequency_encode_service_area,
    numeric_median_imputer,
    prepare_predictors,
    target_encode_service_area,
)

train = load_train()
X = prepare_predictors(train)
y = train["churn"]

# %% [markdown]
# ## Preprocessing options
#
# We use the same eligible predictors and model settings for every option.
# This lets us compare the effect of each set of preprocessing steps.
# We compare two ways to encode `ServiceArea`. Target encoding estimates churn rates
# by area. Frequency encoding measures how common each area is.

# %%
estimator = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

preprocessors = {
    "Target encoding": combine_dense_feature_steps(
        numeric=numeric_median_imputer(),
        service_area=target_encode_service_area(),
    ),
    "Frequency encoding": combine_dense_feature_steps(
        numeric=numeric_median_imputer(),
        service_area=frequency_encode_service_area(),
    ),
}

# %% [markdown]
# ## Performance before tuning
#
# We use repeated stratified cross-validation. We split the training data into five
# parts with similar churn rates, validate on each part in turn, and repeat with a new
# split. We use the same splits for every model and option. In each split, we
# fit preprocessing and the model on the training rows.
#
# Average precision summarizes how well we find churners while limiting false alarms.
# ROC AUC measures how often a churner ranks above a customer who stays.
# We report the mean and sample standard deviation, which describes how much
# the ten validation scores vary.

# %%
results = compare_preprocessing(estimator, preprocessors, X, y)
display(results.round(3))

# %% [markdown]
# ## Hyperparameter tuning
#
# Hyperparameters are settings we choose before fitting. We will tune `max_features` and `min_samples_leaf`, which control how many inputs
# each split can consider and the minimum number of customers at the end of a tree path.
# We will compare a small set of settings on the training partition.

# %% [markdown]
# ## Feature importance
#
# Once we fit our chosen pipeline, we will use `tree_feature_importance` from
# `src.evaluation` to see which inputs did the most to separate churners in tree splits.
# This score can favor inputs with many possible split points. Each column created
# by one-hot encoding gets its own score.
