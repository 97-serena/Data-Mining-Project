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
# # Decision Tree
#
# We predict churn from information available when we decide whom to contact.
# We estimate churn by asking a series of questions about the inputs.

# %%
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.data import RANDOM_STATE, load_train
from src.preprocessing import (
    ServiceAreaFrequencyEncoder,
    combine_feature_steps,
    numeric_median_imputer,
    numeric_zero_imputer,
    prepare_predictors,
    target_encode_service_area,
)

train = load_train()
X = prepare_predictors(train)
y = train["churn"]

splitter = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=RANDOM_STATE,
)

# %% [markdown]
# ## Preprocessing options
#
# We compare three ways to encode service area. Target encoding estimates churn
# rates, frequency encoding measures how common each area is, and one-hot
# encoding creates a 0/1 column per area. We compare filling missing numeric values
# with the training median or with zero.

# %%
model = DecisionTreeClassifier()

encoding_options = [
    ("Target", target_encode_service_area()),
    ("Frequency", ServiceAreaFrequencyEncoder()),
    ("One-hot", OneHotEncoder(handle_unknown="ignore")),
]

imputation_options = [
    ("Median", numeric_median_imputer()),
    ("Zero", numeric_zero_imputer()),
]

# %% [markdown]
# ## Performance before tuning
#
# We fit preprocessing and the model within each training fold. We report the
# mean and standard deviation across ten validation splits for average precision
# (finding churners while limiting false alarms) and ROC AUC (ranking churners
# above customers who stay).

# %%
rows = []

for encoding_name, encoder in encoding_options:
    for imputation_name, imputer in imputation_options:
        preprocessing = combine_feature_steps(
            numeric=imputer,
            service_area=encoder,
        )
        pipeline = make_pipeline(preprocessing, model)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=splitter,
            scoring=["average_precision", "roc_auc"],
            error_score="raise",
        )

        ap = scores["test_average_precision"]
        auc = scores["test_roc_auc"]
        rows.append(
            [
                encoding_name,
                imputation_name,
                ap.mean(),
                ap.std(ddof=1),
                auc.mean(),
                auc.std(ddof=1),
            ]
        )


results = pd.DataFrame(
    rows,
    columns=["Encoding", "Imputation", "AP mean", "AP SD", "ROC AUC mean", "ROC AUC SD"],
)
display(results.round(4))

# %% [markdown]
# Frequency encoding (needs checking).

# %% [markdown]
# ## Hyperparameter tuning
#
# Hyperparameters are settings we choose before fitting. We will tune `max_depth`
# and `min_samples_leaf` to control tree depth and the minimum size of its leaves.
# We will compare these settings on the training partition.

# %% [markdown]
# ## Feature importance
#
# Once we fit our chosen pipeline, we will read the tree model's built-in
# `feature_importances_` and label the scores using the fitted preprocessing's
# `get_feature_names_out()`. This measure can favor inputs with many split points.
