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

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from joblib import parallel_config
from sklearn import base as sk_base
from sklearn import inspection as sk_inspection
from sklearn import model_selection as sk_selection
from sklearn import pipeline as sk_pipeline
from sklearn import preprocessing as sk_preprocessing
from sklearn import tree as sk_tree

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
# We compare three ways to encode service area. Target encoding estimates churn
# rates, frequency encoding measures how common each area is, and one-hot
# encoding creates a 0/1 column per area. We compare filling missing numeric values
# with the training median or with zero.
#
# Data exploration found 730 service-area levels including the missing marker.
# Many have fewer than 20 training customers. Frequency and target encoding
# each represent area with one column.
# All options use the same predictors, with mode imputation for yes/no inputs
# and one-hot encoding for the other categorical inputs. Numeric and yes/no
# imputation retain missingness indicators, and we keep numeric values unscaled.
#
# We fix the seed for repeatable results and use the other estimator defaults.

# %%
model = sk_tree.DecisionTreeClassifier(random_state=data.RANDOM_STATE)

encoding_options = [
    ("Target", preprocessing.target_encode_service_area()),
    ("Frequency", preprocessing.ServiceAreaFrequencyEncoder()),
    ("One-hot", sk_preprocessing.OneHotEncoder(handle_unknown="ignore")),
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
    for imputation_name, imputer in imputation_options:
        feature_steps = preprocessing.combine_feature_steps(
            numeric=imputer,
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

# %% [markdown]
# ## Preprocessing decision
#
# **We choose frequency encoding and median imputation.** Mean AP was
# 0.31145 (SD 0.00283). One-hot encoding with zero had the highest mean AP
# (0.31190) and led on 5 of 10 folds. Its advantage was 0.00044 AP (paired
# SD 0.00304), inside our 0.002 near-tie range. Frequency keeps one area
# column and uses only counts. With frequency fixed, median fill led
# zero fill by 0.00118 AP on average and uses a typical observed value.
#
# On each fold we subtract our choice's AP from the alternative's AP. We show
# the mean and SD of those differences and count the folds where the alternative led.

# %%
chosen_row = results.index[
    results["Encoding"].eq("Frequency")
    & results["Imputation"].eq("Median")
][0]
paired_ap = pd.DataFrame(ap_by_option).sub(ap_by_option[chosen_row], axis="columns")
results["AP difference"] = paired_ap.mean(axis=1)
results["Difference SD"] = paired_ap.std(axis=1, ddof=1)
results["Folds ahead"] = paired_ap.gt(0).sum(axis=1)
display(results.round(5))

# %%
chosen_features = preprocessing.combine_feature_steps(
    numeric=preprocessing.numeric_median_imputer(),
    service_area=preprocessing.ServiceAreaFrequencyEncoder(),
)
chosen_pipeline = sk_pipeline.make_pipeline(chosen_features, model)

# %% [markdown]
# ## Hyperparameter tuning
#
# We tune only the decision tree in `chosen_pipeline`, using grid search on
# the same ten validation splits. We keep frequency encoding and median
# imputation fixed and fit both on each fold's training rows.
#
# Hyperparameters are settings we choose before fitting. We tune `max_depth`
# to limit the depth of the tree, `min_samples_leaf` to set the minimum number
# of training customers per leaf, and `class_weight` to compare unweighted
# and balanced training. The depth grid extends to 15 so that the best depth
# does not sit at the edge of the grid.
#
# We use the same ten validation folds. Among settings within 0.002 AP of
# the best, we prefer a shallower tree, then larger leaves, then no class
# weights. We use mean AP if all three choices match.

# %%
model_step = chosen_pipeline.steps[-1][0]
param_grid = {
    f"{model_step}__max_depth": [2, 3, 4, 5, 6, 8, 10, 12, 15],
    f"{model_step}__min_samples_leaf": [25, 100, 250, 500],
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
        "Max depth": cv_results[f"param_{model_step}__max_depth"].astype(int),
        "Min samples per leaf": cv_results[f"param_{model_step}__min_samples_leaf"].astype(int),
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
    .sort_values(
        ["Max depth", "Min samples per leaf", "weighted", "AP mean"],
        ascending=[True, False, True, False],
    )
    .index[0]
)

tuning["AP difference"] = fold_ap.sub(fold_ap.loc[chosen_setting], axis="columns").mean(axis=1)
print("Chosen hyperparameters")
display(tuning.loc[[chosen_setting]].round(5))
display(tuning.sort_values("AP mean", ascending=False).head(15).round(5))

# %% [markdown]
# ## Tuning decision
#
# **We choose `max_depth=10`, `min_samples_leaf=250`, and no class weights.**
# The best mean AP, 0.40071, used a depth of 15, leaves of 250 and balanced
# weights. Six settings fell within 0.002 AP of it, all with leaves of 250 and
# depths from 10 to 15, so we choose the shallowest unweighted one. Its mean AP
# was 0.39984 (SD 0.00519), 0.00086 below the best. Deeper trees did not
# improve on depth 10 by more than our tolerance. The tuned tree beat the
# default tree's 0.31145 AP on all 10 folds.

# %%
tuned_pipeline = sk_base.clone(chosen_pipeline).set_params(
    **cv_results.loc[chosen_setting, "params"]
)
_ = tuned_pipeline.fit(X, y)

# %% [markdown]
# ## Model evaluation
#
# We compare the default and tuned trees with the chosen preprocessing on
# the same ten validation folds used for tuning. See [Model comparison](model_comparison.py)
# for the final test evaluation plan.

# %%
metrics = ["AP mean", "AP SD", "ROC AUC mean"]
comparison = pd.DataFrame(
    [results.loc[chosen_row, metrics], tuning.loc[chosen_setting, metrics]],
    index=["Default tree", "Tuned tree"],
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

# %% [markdown]
# ## Built-in importance
#
# As a supplement, we read the tree's built-in `feature_importances_` from the
# tuned pipeline fitted on the full training partition. Each score is the share
# of the tree's total weighted impurity reduction attributed to splits on that
# transformed feature. This measure can favor features with many possible
# split points and splits the credit for a category across its one-hot columns.

# %%
fitted_features = tuned_pipeline[0]
fitted_tree = tuned_pipeline[-1]

impurity_importance = pd.Series(
    fitted_tree.feature_importances_,
    index=fitted_features.get_feature_names_out(),
    name="Impurity importance",
).sort_values(ascending=False)
impurity_importance.index.name = "Feature"
display(impurity_importance.head(10).round(5).to_frame())
