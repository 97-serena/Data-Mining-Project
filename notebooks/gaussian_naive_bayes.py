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
# # Gaussian Naive Bayes
#
# We predict churn, when a customer leaves, using information available when we decide
# whom to contact. We will use churn risk and customer value to prioritize outreach.
# We use Gaussian Naive Bayes to estimate churn from how inputs are distributed among
# customers who churned and those who stayed.

# %%
from pathlib import Path
import sys

from IPython.display import display
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.data import load_train
from src.evaluation import compare_preprocessing
from src.preprocessing import (
    combine_dense_feature_steps,
    frequency_encode_service_area,
    log_skewed_numeric,
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
# We compare target encoding (an estimated churn rate by `ServiceArea`) and frequency
# encoding (how common each area is). We also try a log transform, which compresses large
# values and may help some inputs better match the model's bell-shaped distributions.

# %%
estimator = GaussianNB()

preprocessors = {
    "Target encoding": combine_dense_feature_steps(
        numeric=numeric_median_imputer(),
        service_area=target_encode_service_area(),
    ),
    "Frequency encoding": combine_dense_feature_steps(
        numeric=numeric_median_imputer(),
        service_area=frequency_encode_service_area(),
    ),
    "Target encoding + log transform": combine_dense_feature_steps(
        numeric=make_pipeline(log_skewed_numeric(), numeric_median_imputer()),
        service_area=target_encode_service_area(),
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
# Hyperparameters are settings we choose before fitting. We will tune `var_smoothing`, which stabilizes estimates for inputs with little variation.
# We will compare a small set of settings on the training partition.
