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
# # Logistic Regression
#
# We predict churn, when a customer leaves, using information available when we decide
# whom to contact. We will use churn risk and customer value to prioritize outreach.
# We use logistic regression to combine inputs into a churn probability.

# %%
from pathlib import Path
import sys

from IPython.display import display
from sklearn.linear_model import LogisticRegression

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
# We compare two ways to encode `ServiceArea`. Target encoding estimates churn rates by area,
# while one-hot encoding represents areas with 0/1 columns. We scale numeric inputs
# so different units do not dominate fitting, and try a log transform to compress large values.

# %%
estimator = LogisticRegression(C=1, max_iter=2000)

preprocessors = {
    "Target encoding + scaling": make_preprocessor(encoding="target", scale_numeric=True),
    "One-hot encoding + scaling": make_preprocessor(encoding="onehot", scale_numeric=True),
    "Target encoding + scaling + log transform": make_preprocessor(
        encoding="target", scale_numeric=True, log_numeric=True,
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
# Hyperparameters are settings we choose before fitting. We will tune `C`, which controls how strongly we limit large model weights.
# We will compare a small set of settings on the training partition.
