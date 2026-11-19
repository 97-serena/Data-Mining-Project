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
# # Histogram Gradient Boosting
#
# We predict churn from information available when we decide whom to contact.
# We use ranges of numeric inputs to build a sequence of trees that improve earlier predictions.

# %%
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.data import RANDOM_STATE, load_train
from src.preprocessing import (
    ServiceAreaFrequencyEncoder,
    combine_feature_steps,
    numeric_median_imputer,
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
# We compare target encoding (estimated churn rate by service area) with frequency
# encoding (how common each area is). We keep median imputation fixed.

# %%
model = HistGradientBoostingClassifier()

encoding_options = [
    ("Target", target_encode_service_area()),
    ("Frequency", ServiceAreaFrequencyEncoder()),
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
    # Keep the transformed features in a dense matrix.
    preprocessing = combine_feature_steps(
        numeric=numeric_median_imputer(),
        service_area=encoder,
    ).set_params(sparse_threshold=0.0)
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
            ap.mean(),
            ap.std(ddof=1),
            auc.mean(),
            auc.std(ddof=1),
        ]
    )

results = pd.DataFrame(
    rows,
    columns=["Encoding", "AP mean", "AP SD", "ROC AUC mean", "ROC AUC SD"],
)
display(results.round(4))

# %%
Target encoding (needs checking) add zero vs median

# %% [markdown]
# ## Hyperparameter tuning
#
# Hyperparameters are settings we choose before fitting. We will try combinations
# of `learning_rate`, `max_iter`, and `max_leaf_nodes` to control each tree's impact,
# the number of trees, and their size.
# We will compare these settings on the training partition.
