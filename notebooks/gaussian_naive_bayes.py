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
# We predict churn from information available when we decide whom to contact.
# We estimate churn from how inputs are distributed among customers who churned and stayed.

# %%
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display
from sklearn import model_selection as sk_selection
from sklearn import naive_bayes as sk_naive_bayes
from sklearn import pipeline as sk_pipeline

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
# We compare target encoding (estimated churn rate by service area) with frequency
# encoding (how common each area is). For each encoder, we try a log transform
# to compress large values. We then fill missing numbers with training-fold
# medians or zero.
#
# We use the estimator defaults for every preprocessing option.

# %%
model = sk_naive_bayes.GaussianNB()

encoding_options = [
    ("Target", preprocessing.target_encode_service_area()),
    ("Frequency", preprocessing.ServiceAreaFrequencyEncoder()),
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
            numeric_steps = optional_log_steps + [imputer]
            # Keep the transformed features in a dense matrix.
            feature_steps = preprocessing.combine_feature_steps(
                numeric=sk_pipeline.make_pipeline(*numeric_steps),
                service_area=encoder,
            ).set_params(sparse_threshold=0.0)
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
# **We choose frequency encoding, the log transform, and median imputation.**
# Mean AP was 0.35150 (SD 0.00794). With frequency and median fixed, adding
# the log step raised AP by 0.01025 (paired SD 0.00555) and led on all 10
# folds. Switching to target encoding or to zero imputation changed mean AP
# by only 0.00009 or 0.00015. We keep the encoder that only needs area counts
# and fill numeric blanks with typical observed values.
#
# On each fold we subtract our choice's AP from the alternative's AP. We show
# the mean and SD of those differences and count the folds where the alternative led.

# %%
chosen_row = results.index[
    results["Encoding"].eq("Frequency")
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
    ),
    service_area=preprocessing.ServiceAreaFrequencyEncoder(),
).set_params(sparse_threshold=0.0)
chosen_pipeline = sk_pipeline.make_pipeline(chosen_features, model)

# %% [markdown]
# ## Hyperparameter tuning
#
# We will tune only the estimator in `chosen_pipeline`, using validation AP.
# The chosen preprocessing steps will fit again on each fold's training rows.
#
# Hyperparameters are settings we choose before fitting. We will tune
# `var_smoothing`, which stabilizes estimates for inputs with little variation.
# We will compare these settings on the training partition.
