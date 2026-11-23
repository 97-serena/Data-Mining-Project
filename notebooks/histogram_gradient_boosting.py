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
from joblib import parallel_config
from sklearn import base as sk_base
from sklearn import ensemble as sk_ensemble
from sklearn import model_selection as sk_selection
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
AP_TOLERANCE = 0.002
N_JOBS = 4

# %% [markdown]
# ## Preprocessing options
#
# We compare target encoding (estimated churn rate by service area) with frequency
# encoding (how common each area is). For numeric inputs, we compare median
# and zero imputation with letting the trees learn how to route missing values.
#
# We fix the seed for repeatable results and use the other estimator defaults.

# %%
model = sk_ensemble.HistGradientBoostingClassifier(random_state=data.RANDOM_STATE)

# List the simpler choices first within each step for the AP near-tie rule.
encoding_options = [
    ("Frequency", preprocessing.ServiceAreaFrequencyEncoder()),
    ("Target", preprocessing.target_encode_service_area()),
]

imputation_options = [
    ("Native missing values", "passthrough"),
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
pipelines = []

for encoding_name, encoder in encoding_options:
    for imputation_name, imputer in imputation_options:
        # Keep the transformed features in a dense matrix.
        feature_steps = preprocessing.combine_feature_steps(
            numeric=imputer,
            service_area=encoder,
        ).set_params(sparse_threshold=0.0)
        pipeline = sk_pipeline.make_pipeline(feature_steps, model)
        pipelines.append(pipeline)

        with parallel_config(backend="loky", inner_max_num_threads=1):
            scores = sk_selection.cross_validate(
                pipeline,
                X,
                y,
                cv=splitter,
                scoring=["average_precision", "roc_auc"],
                n_jobs=N_JOBS,
                error_score="raise",
            )

        ap = scores["test_average_precision"]
        ap_by_option.append(ap)
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
    columns=["Encoding", "Numeric missing values", "AP mean", "AP SD", "ROC AUC mean", "ROC AUC SD"],
)

# %% [markdown]
# ## Preprocessing decision
#
# Among options within 0.002 AP of the best, we prefer frequency encoding
# because it uses area counts alone. For numeric missing values, we prefer
# native handling, followed by median fill and zero fill.
#
# On each fold we subtract our choice's AP from the alternative's AP. We show
# the mean and SD of those differences and count the folds where the alternative led.

# %%
near_tie = results["AP mean"] >= results["AP mean"].max() - AP_TOLERANCE
chosen_row = results.index[near_tie][0]
paired_ap = pd.DataFrame(ap_by_option).sub(ap_by_option[chosen_row], axis="columns")
results["AP difference"] = paired_ap.mean(axis=1)
results["Difference SD"] = paired_ap.std(axis=1, ddof=1)
results["Folds ahead"] = paired_ap.gt(0).sum(axis=1)
print("Chosen preprocessing")
display(results.loc[[chosen_row]].round(5))
display(results.round(5))

# %%
chosen_pipeline = pipelines[chosen_row]

# %% [markdown]
# **We choose service-area frequency encoding and native numeric missing-value
# handling.** With the default estimator, this option reached 0.45201 mean AP.
# Frequency encoding with zero fill reached 0.45262, a difference of 0.00061
# within our 0.002 threshold. We keep the simpler pipeline and let the trees
# learn where to send rows with missing numeric values.

# %% [markdown]
# ## Hyperparameter tuning
#
# We refit the chosen preprocessing on each fold's training rows. We tune
# `learning_rate` (each tree's contribution), `max_iter` (the number of trees),
# and `max_leaf_nodes` (the size of each tree).
#
# Smaller learning rates get more rounds so they can accumulate comparable
# updates. Each of the six rate and round pairs is tested with three tree
# sizes. We disable early stopping for every candidate so each fit uses its
# specified number of rounds. The default-model comparison uses scikit-learn's
# automatic early stopping.
#
# We use the same ten validation folds. Among settings within 0.002 AP of
# the best, we prefer fewer leaves per tree, then fewer trees. We use mean
# AP if both complexity limits match.

# %%
model_step = chosen_pipeline.steps[-1][0]
fixed_round_pipeline = sk_base.clone(chosen_pipeline).set_params(
    **{f"{model_step}__early_stopping": False}
)
rate_and_rounds = [(0.05, [200, 400]), (0.10, [100, 200]), (0.20, [50, 100])]
param_grid = [
    {
        f"{model_step}__learning_rate": [learning_rate],
        f"{model_step}__max_iter": rounds,
        f"{model_step}__max_leaf_nodes": [15, 31, 63],
    }
    for learning_rate, rounds in rate_and_rounds
]

search = sk_selection.GridSearchCV(
    estimator=fixed_round_pipeline,
    param_grid=param_grid,
    scoring=["average_precision", "roc_auc"],
    cv=splitter,
    refit=False,
    n_jobs=N_JOBS,
    error_score="raise",
)
with parallel_config(backend="loky", inner_max_num_threads=1):
    _ = search.fit(X, y)

# %%
cv_results = pd.DataFrame(search.cv_results_)
n_folds = splitter.get_n_splits()
fold_ap = cv_results[[f"split{i}_test_average_precision" for i in range(n_folds)]]
tuning = pd.DataFrame(
    {
        "Learning rate": cv_results[f"param_{model_step}__learning_rate"],
        "Boosting rounds": cv_results[f"param_{model_step}__max_iter"],
        "Max leaves": cv_results[f"param_{model_step}__max_leaf_nodes"],
        "AP mean": cv_results["mean_test_average_precision"],
        "AP SD": fold_ap.std(axis=1, ddof=1),
        "ROC AUC mean": cv_results["mean_test_roc_auc"],
    }
)
tuning["Near tie"] = tuning["AP mean"] >= tuning["AP mean"].max() - AP_TOLERANCE
chosen_setting = tuning.loc[tuning["Near tie"]].sort_values(
    ["Max leaves", "Boosting rounds", "AP mean"], ascending=[True, True, False]
).index[0]

tuning["AP difference"] = fold_ap.sub(fold_ap.loc[chosen_setting], axis="columns").mean(axis=1)
print("Chosen hyperparameters")
display(tuning.loc[[chosen_setting]].round(5))
display(tuning.sort_values("AP mean", ascending=False).round(5))

# %%
metrics = ["AP mean", "AP SD", "ROC AUC mean"]
comparison = pd.DataFrame(
    [results.loc[chosen_row, metrics], tuning.loc[chosen_setting, metrics]],
    index=["Default boosting", "Tuned boosting"],
)
display(comparison.round(5))

tuned_pipeline = sk_base.clone(fixed_round_pipeline).set_params(
    **cv_results.loc[chosen_setting, "params"]
)
_ = tuned_pipeline.fit(X, y)

# %% [markdown]
# ## Tuning decision
#
# **We choose `learning_rate=0.10`, `max_iter=200`, and `max_leaf_nodes=15`.**
# The best grid result used 400 rounds at a learning rate of 0.05 and the same
# leaf limit. Its mean AP was 0.00149 higher, within our near-tie range, so we
# choose the 200-round model.
#
# With the chosen preprocessing, mean validation AP rose from 0.45201 at
# the defaults to 0.45703. Mean ROC AUC rose from 0.67600 to 0.67846. We fit
# `tuned_pipeline` on the full training partition and reserve the test split
# for final evaluation.
