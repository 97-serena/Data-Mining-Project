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
# We predict churn from information available when we decide whom to contact.
# We use logistic regression to estimate each customer's probability of churning.

# %%
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display
from sklearn import linear_model as sk_linear
from sklearn import model_selection as sk_selection
from sklearn import pipeline as sk_pipeline
from sklearn import preprocessing as sk_preprocessing

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src import data, preprocessing

train = data.load_train()
X = preprocessing.prepare_predictors(train)
y = train["churn"]

splitter = sk_selection.RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=data.RANDOM_STATE,
)

# %% [markdown]
# ## Preprocessing options
#
# We compare target encoding (estimated churn rate by service area) with one-hot
# encoding (a 0/1 column per area). For each encoder, we try a log transform
# to compress large values. We fill missing numbers with training-fold
# medians or zero, then scale the numeric inputs.
#
# We give the default solver a fixed 2,000-iteration cap so it can reach
# convergence for every option. All other estimator settings use defaults.

# %%
model = sk_linear.LogisticRegression(max_iter=2000)

encoding_options = [
    ("Target", preprocessing.target_encode_service_area()),
    ("One-hot", sk_preprocessing.OneHotEncoder(handle_unknown="ignore")),
]

log_options = [
    ("No log", []),
    ("Log", [preprocessing.LogSkewedNumeric()]),
]

imputation_options = [
    ("Median", preprocessing.numeric_median_imputer()),
    ("Zero", preprocessing.numeric_zero_imputer()),
]

# %% [markdown]
# ## Performance before tuning
#
# We fit each pipeline inside the same ten training folds and use mean average
# precision (AP) to compare the preprocessing options. We also report ROC AUC
# and how much both scores vary across folds.

# %%
rows = []
ap_by_option = []

for encoding_name, encoder in encoding_options:
    for log_name, optional_log_steps in log_options:
        for imputation_name, imputer in imputation_options:
            numeric_steps = optional_log_steps + [
                imputer,
                sk_preprocessing.StandardScaler(),
            ]
            feature_steps = preprocessing.combine_feature_steps(
                numeric=sk_pipeline.make_pipeline(*numeric_steps),
                service_area=encoder,
            )
            pipeline = sk_pipeline.make_pipeline(feature_steps, model)

            scores = sk_selection.cross_validate(
                pipeline,
                X,
                y,
                cv=splitter,
                scoring=["average_precision", "roc_auc"],
                error_score="raise",
            )

            ap = scores["test_average_precision"]
            ap_by_option.append(ap)
            auc = scores["test_roc_auc"]
            rows.append(
                [
                    encoding_name,
                    log_name,
                    imputation_name,
                    ap.mean(),
                    ap.std(ddof=1),
                    auc.mean(),
                    auc.std(ddof=1),
                ]
            )

results = pd.DataFrame(
    rows,
    columns=["Encoding", "Log", "Imputation", "AP mean", "AP SD", "ROC AUC mean", "ROC AUC SD"],
)

# %% [markdown]
# ## Preprocessing decision
#
# **We choose target encoding, the log transform, median imputation, and scaling.**
# Mean AP was 0.39804 (SD 0.00667). With median held fixed, adding the log step
# raised AP by 0.01606 (paired SD 0.00320). Target encoding beat one-hot by
# 0.00997 (paired SD 0.00247) with log and median fixed. Each gain appeared
# on all 10 folds. Zero raised mean AP by 0.00004 (paired SD 0.00025) and
# led on 5 folds. We keep median as a typical observed value for a missing input.
#
# On each fold we subtract our choice's AP from the alternative's AP. We show
# the mean and SD of those differences and count the folds where the alternative led.

# %%
chosen_row = results.index[
    results["Encoding"].eq("Target")
    & results["Log"].eq("Log")
    & results["Imputation"].eq("Median")
][0]
paired_ap = pd.DataFrame(ap_by_option).sub(ap_by_option[chosen_row], axis="columns")
results["AP difference"] = paired_ap.mean(axis=1)
results["Difference SD"] = paired_ap.std(axis=1, ddof=1)
results["Folds ahead"] = paired_ap.gt(0).sum(axis=1)
display(results.round(5))

# %%
chosen_features = preprocessing.combine_feature_steps(
    numeric=sk_pipeline.make_pipeline(
        preprocessing.LogSkewedNumeric(),
        preprocessing.numeric_median_imputer(),
        sk_preprocessing.StandardScaler(),
    ),
    service_area=preprocessing.target_encode_service_area(),
)
chosen_pipeline = sk_pipeline.make_pipeline(chosen_features, model)

# %% [markdown]
# ## Hyperparameter tuning
#
# We will tune only the estimator in `chosen_pipeline`, using validation AP.
# The chosen preprocessing steps will fit again on each fold's training rows.
#
# Hyperparameters are settings we choose before fitting. We will tune `C`, which
# controls how strongly we limit large model weights.
# We will compare these settings on the training partition.
