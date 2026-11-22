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

splitter = sk_selection.RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=data.RANDOM_STATE,
)

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

        scores = sk_selection.cross_validate(
            pipeline,
            X,
            y,
            cv=splitter,
            scoring=["average_precision", "roc_auc"],
            error_score="raise",
            n_jobs=4,
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
# and balanced training.
#
# We select the settings with the highest mean validation AP and report
# ROC AUC alongside it. Grid search then refits the selected pipeline on
# the full training partition.

# %%
param_grid = {
    "decisiontreeclassifier__max_depth": [2, 3, 4, 5, 6, 8, 10],
    "decisiontreeclassifier__min_samples_leaf": [25, 100, 250, 500],
    "decisiontreeclassifier__class_weight": [None, "balanced"],
}
search = sk_selection.GridSearchCV(
    estimator=chosen_pipeline,
    param_grid=param_grid,
    scoring=["average_precision", "roc_auc"],
    refit="average_precision",
    cv=splitter,
    error_score="raise",
    n_jobs=4,
)
search.fit(X, y)
final_pipeline = search.best_estimator_

# %%
tuning_results = pd.DataFrame(
    {
        "Max depth": search.cv_results_["param_decisiontreeclassifier__max_depth"],
        "Min samples per leaf": search.cv_results_[
            "param_decisiontreeclassifier__min_samples_leaf"
        ],
        "Class weight": search.cv_results_["param_decisiontreeclassifier__class_weight"],
        "AP mean": search.cv_results_["mean_test_average_precision"],
        "ROC AUC mean": search.cv_results_["mean_test_roc_auc"],
    }
)
tuning_results["Class weight"] = tuning_results["Class weight"].fillna("Unweighted")
tuning_results = tuning_results.sort_values("AP mean", ascending=False, kind="stable")
display(tuning_results.head(10).round(5))

comparison = pd.DataFrame(
    [
        results.loc[chosen_row, ["AP mean", "ROC AUC mean"]],
        tuning_results.loc[search.best_index_, ["AP mean", "ROC AUC mean"]],
    ],
    index=["Default tree", "Tuned tree"],
)
display(comparison.round(5))

# %% [markdown]
# The highest mean validation AP in this grid was 0.40061 with a depth limit
# of 10, a minimum of 250 customers per leaf, and balanced class weights.
# With the same depth and leaf size, the unweighted tree reached 0.39984 AP.
# The tuned tree improved on the default tree's 0.31145 AP, and its ROC AUC
# rose from 0.54643 to 0.63968. These are development scores from the training
# partition. We reserve test evaluation for the final model comparison.

# %% [markdown]
# ## Feature importance
#
# Each score is the share of the fitted tree's total weighted impurity reduction
# attributed to splits on that transformed feature. Larger scores indicate a
# greater contribution. This measure can favor features with many possible split points.

# %%
processor, tree = final_pipeline[0], final_pipeline[-1]
names = processor.get_feature_names_out()
feature_importance = pd.DataFrame({
    "Feature": names,
    "Importance": tree.feature_importances_,
}).sort_values("Importance", ascending=False, ignore_index=True)
display(feature_importance.head(15).round(5))
