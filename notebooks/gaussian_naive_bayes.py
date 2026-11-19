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

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.data import load_training_data
from src.evaluation import compare_preprocessing
from src.preprocessing import make_preprocessor

X, y = load_training_data()

# %% [markdown]
# ## Preprocessing options
#
# Preprocessing prepares the inputs for the model. We keep model settings fixed so we
# can compare how the options below affect performance.
# We compare target encoding (an estimated churn rate by `ServiceArea`) and frequency
# encoding (how common each area is). We also try a log transform, which compresses large
# values and may help some inputs better match the model's bell-shaped distributions.

# %%
estimator = GaussianNB()

preprocessors = {
    "Target encoding": make_preprocessor(encoding="target", dense=True),
    "Frequency encoding": make_preprocessor(encoding="frequency", dense=True),
    "Target encoding + log transform": make_preprocessor(
        encoding="target", log_numeric=True, dense=True,
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
