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

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from joblib import parallel_config
from sklearn import base as sk_base
from sklearn import inspection as sk_inspection
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

N_SPLITS = 5
# Use four workers and limit each to one compute thread below.
N_JOBS = 4
SCORING = ["average_precision", "roc_auc"]

splitter = sk_selection.RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=2,
    random_state=data.RANDOM_STATE,
)
AP_TOLERANCE = 0.002

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
# We fit each pipeline on the same ten training folds. We compare mean
# validation average precision (AP) and report ROC AUC alongside it.

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

            with parallel_config(backend="loky", inner_max_num_threads=1):
                scores = sk_selection.cross_validate(
                    pipeline,
                    X,
                    y,
                    cv=splitter,
                    scoring=SCORING,
                    n_jobs=N_JOBS,
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
    columns=[
        "Encoding",
        "Log",
        "Imputation",
        "AP mean",
        "AP SD",
        "ROC AUC mean",
        "ROC AUC SD",
    ],
)

# %% [markdown]
# ## Preprocessing decision
#
# **We choose target encoding, the log transform, median imputation, and scaling.**
# Mean AP was 0.39804 (SD 0.00668). With median held fixed, adding the log step
# raised AP by 0.01606 (paired SD 0.00320). Target encoding beat one-hot by
# 0.00984 (paired SD 0.00244) with log and median fixed. Each gain appeared
# on all 10 folds. Zero raised mean AP by 0.00004 (paired SD 0.00026) and
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
# We keep the preprocessing choice fixed and fit it again on each fold's
# training rows. We tune `C` and `class_weight`. Smaller values of `C` apply
# stronger regularization. Balanced class weights give churners more weight
# during fitting.
#
# We use the same ten validation folds. Among settings within 0.002 AP of
# the best, we prefer no class weights, then the smallest `C`, because
# stronger regularization gives a simpler model. We use mean AP if both
# choices match.

# %%
model_step = chosen_pipeline.steps[-1][0]
param_grid = {
    f"{model_step}__C": [
        0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.03, 0.1, 0.3,
        1.0, 3.0, 10.0, 30.0, 100.0,
    ],
    f"{model_step}__class_weight": [None, "balanced"],
}

search = sk_selection.GridSearchCV(
    estimator=chosen_pipeline,
    param_grid=param_grid,
    scoring=SCORING,
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
        "C": cv_results[f"param_{model_step}__C"].astype(float),
        "Class weight": cv_results[f"param_{model_step}__class_weight"].fillna("Unweighted"),
        "AP mean": cv_results["mean_test_average_precision"],
        "AP SD": fold_ap.std(axis=1, ddof=1),
        "ROC AUC mean": cv_results["mean_test_roc_auc"],
    }
)
tuning["Near tie"] = tuning["AP mean"] >= tuning["AP mean"].max() - AP_TOLERANCE
chosen_setting = (
    tuning.loc[tuning["Near tie"]]
    .assign(weighted=lambda table: table["Class weight"].ne("Unweighted"))
    .sort_values(["weighted", "C", "AP mean"], ascending=[True, True, False])
    .index[0]
)

tuning["AP difference"] = fold_ap.sub(fold_ap.loc[chosen_setting], axis="columns").mean(axis=1)
print("Chosen hyperparameters")
display(tuning.loc[[chosen_setting]].round(5))
display(tuning.sort_values("AP mean", ascending=False).round(5))

# %% [markdown]
# ## Tuning decision
#
# **We choose `C=0.0015` with no class weights.** The best mean AP, 0.39931,
# came from `C=0.005` without class weights. Every unweighted setting from
# `C=0.0015` up to the default `C=1` fell within 0.002 AP of it, so we choose
# the strongest regularization in this range. Its mean AP was 0.39791
# (SD 0.00722), 0.00140 below the best and 0.00013 below the default, and it
# led the default on 4 of 10 folds. Tuning makes little difference for logistic
# regression: validation AP is almost the same for any `C` in this range.

# %%
tuned_pipeline = sk_base.clone(chosen_pipeline).set_params(
    **cv_results.loc[chosen_setting, "params"]
)
_ = tuned_pipeline.fit(X, y)

# %% [markdown]
# ## Model evaluation
#
# We compare the default and tuned models with the chosen preprocessing on
# the same ten validation folds used for tuning. See [Model comparison](model_comparison.py)
# for the final test evaluation plan.

# %%
metrics = ["AP mean", "AP SD", "ROC AUC mean"]
comparison = pd.DataFrame(
    [results.loc[chosen_row, metrics], tuning.loc[chosen_setting, metrics]],
    index=["Default", "Tuned"],
)
display(comparison.round(5))

# Compare AP on matching folds so each pair uses the same validation rows.
ap_gain = fold_ap.loc[chosen_setting].to_numpy() - ap_by_option[chosen_row]
print(f"The tuned model leads on {(ap_gain > 0).sum()} of {n_folds} folds by AP.")

# %% [markdown]
# ## Top 10 features by permutation importance
#
# We refit the tuned pipeline on each training fold in the first repeat.
# On its validation fold, we shuffle each input five times before encoding
# and measure the AP drop. This gives each categorical input one score.

# %%
first_repeat = list(splitter.split(X, y))[:N_SPLITS]
permutation_rows = []
with parallel_config(backend="loky", inner_max_num_threads=1):
    for fit_rows, valid_rows in first_repeat:
        fold_pipeline = sk_base.clone(tuned_pipeline).fit(X.iloc[fit_rows], y.iloc[fit_rows])
        permutation = sk_inspection.permutation_importance(
            fold_pipeline,
            X.iloc[valid_rows],
            y.iloc[valid_rows],
            scoring="average_precision",
            n_repeats=5,
            n_jobs=N_JOBS,
            random_state=data.RANDOM_STATE,
        )
        permutation_rows.append(permutation.importances_mean)

fold_permutation = pd.DataFrame(permutation_rows, columns=X.columns)
feature_importance = pd.DataFrame(
    {
        "Mean AP drop": fold_permutation.mean(),
        "AP drop SD": fold_permutation.std(ddof=1),
        "Folds with a drop": fold_permutation.gt(0).sum(),
    }
).sort_values("Mean AP drop", ascending=False)
feature_importance.index.name = "Feature"
top_features = feature_importance.head(10)
display(top_features.round(5))

# %%
ax = top_features["Mean AP drop"].iloc[::-1].plot.barh(figsize=(8, 4), color="#237E80")
ax.set_xlabel("Mean validation AP drop after shuffling")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
