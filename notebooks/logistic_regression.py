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
from sklearn import metrics as sk_metrics
from sklearn import model_selection as sk_selection
from sklearn import pipeline as sk_pipeline
from sklearn import preprocessing as sk_preprocessing

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src import data, evaluation, preprocessing

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
# Logistic regression has few parameters, one weight per input column, so it
# cannot fit the noise in the training folds the way an unrestricted tree can.
# This is why stronger regularization changes validation AP so little.

# %% [markdown]
# ### Tolerance check
#
# Our near-tie tolerance of 0.002 AP is about the size of random fold-to-fold
# variation in mean AP. A choice should not depend on this exact cut-off, so we
# repeat the same selection rule with tolerances of 0.001 and 0.003.

# %%
tolerance_rows = []
for tolerance in [0.001, 0.002, 0.003]:
    within_tolerance = tuning["AP mean"] >= tuning["AP mean"].max() - tolerance
    setting = (
        tuning.loc[within_tolerance]
        .assign(weighted=lambda table: table["Class weight"].ne("Unweighted"))
        .sort_values(["weighted", "C", "AP mean"], ascending=[True, True, False])
        .index[0]
    )
    tolerance_rows.append(
        {
            "Tolerance": tolerance,
            "C": tuning.loc[setting, "C"],
            "Class weight": tuning.loc[setting, "Class weight"],
            "AP mean": tuning.loc[setting, "AP mean"],
            "AP difference": tuning.loc[setting, "AP difference"],
        }
    )
print(f"Chosen setting by tolerance, AP difference to the {AP_TOLERANCE} choice")
display(pd.DataFrame(tolerance_rows).round(5))

# %% [markdown]
# **The chosen `C` changes with the tolerance, but mean AP stays within 0.00125
# of our choice.** At 0.001 the rule picks `C=0.002` with mean AP 0.39855,
# 0.00064 higher. At 0.003 it picks `C=0.001`, the smallest value in our grid,
# with mean AP 0.39666, 0.00125 lower. All three choices are unweighted and
# strongly regularized, which agrees with our finding that `C` barely changes
# validation AP.

# %%
tuned_pipeline = sk_base.clone(chosen_pipeline).set_params(
    **cv_results.loc[chosen_setting, "params"]
)
_ = tuned_pipeline.fit(X, y)

# %% [markdown]
# ## Model evaluation
#
# We compare the default and tuned models with the chosen preprocessing on
# the same ten validation folds used for tuning. We evaluate the tuned model on
# the test set at the end of this notebook. See [Model comparison](model_comparison.py)
# for how we compare all models on the test set.

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
# ## Outreach at a fixed contact budget
#
# AP and ROC AUC do not need a cut-off. To describe a retention campaign, we
# rank each validation fold by predicted churn probability and contact the top
# 10% of customers. We also show 5% and 20% to see how results change with the
# budget. Precision is the share of contacted customers who churn, recall is
# the share of all churners we contact, and lift is precision divided by the
# fold's churn rate. Random contact gives a lift of 1.
#
# We refit the default and tuned models on the same ten training folds. The
# test set stays unused.

# %%
CONTACT_SHARES = [0.05, 0.10, 0.20]
EVALUATION_SHARE = 0.10

default_name, tuned_name = comparison.index
pipelines = {default_name: chosen_pipeline, tuned_name: tuned_pipeline}
outreach_rows = []
# Each customer appears once in the validation folds of the first repeat
contacted_first_repeat = pd.Series(False, index=X.index)

for fold, (fit_rows, valid_rows) in enumerate(splitter.split(X, y)):
    y_valid = y.iloc[valid_rows]
    for name, pipeline in pipelines.items():
        fold_pipeline = sk_base.clone(pipeline).fit(X.iloc[fit_rows], y.iloc[fit_rows])
        churn_probability = fold_pipeline.predict_proba(X.iloc[valid_rows])[:, 1]
        for share in CONTACT_SHARES:
            outreach_rows.append(
                {
                    "Model": name,
                    "Contact share": share,
                    "Fold": fold,
                    **evaluation.outreach_scores(y_valid, churn_probability, share),
                }
            )
        if name == tuned_name and fold < N_SPLITS:
            contacted_first_repeat.iloc[valid_rows] = evaluation.contact_top(
                churn_probability, EVALUATION_SHARE
            )

outreach = pd.DataFrame(outreach_rows)
display(
    outreach.groupby(["Contact share", "Model"])[["Precision", "Recall", "Lift"]]
    .agg(["mean", "std"])
    .round(4)
)

precision_by_fold = outreach.pivot_table(
    index=["Contact share", "Fold"], columns="Model", values="Precision"
)
precision_gain = precision_by_fold[tuned_name] - precision_by_fold[default_name]
print("Tuned model folds ahead on precision, by contact share:")
print(precision_gain.gt(0).groupby(level="Contact share").sum().to_string())

# %%
confusion = pd.crosstab(
    y.map({1: "Churned", 0: "Stayed"}).rename("Actual"),
    contacted_first_repeat.map({True: "Contacted", False: "Not contacted"}).rename("Outreach"),
)
print(f"{tuned_name}, top {EVALUATION_SHARE:.0%} of each validation fold in the first repeat")
display(confusion)

# %% [markdown]
# ## Outreach results
#
# **Contacting the top 10%, 46.9% of contacted customers churn with the tuned
# model (SD 1.6 points), compared with 28.8% for random contact (lift 1.63).**
# This reaches 16.3% of all churners. The default model reached 46.8%, and the
# tuned model had higher precision on only 4 of 10 folds at 5% and 10% and on
# 7 of 10 at 20%. As with AP, tuning makes little difference here. A smaller
# budget raises precision but reaches fewer churners: 50.7% precision and 8.8%
# recall at 5%, and 42.9% and 29.7% at 20%.
#
# In the first repeat, the top 10% contained 1,918 churners and 2,167 customers
# who stayed. The other 9,851 churners were not contacted. The team still has
# to agree on the contact budget before the final test evaluation.

# %% [markdown]
# ## Cost of outreach
#
# We give each outcome a cost per customer. These amounts are assumptions for
# illustration, and the team still has to agree on them.
#
# - Each contact costs `CONTACT_COST`, for example a call and a discount.
# - A churner we do not keep costs `CHURN_LOSS`, set to 12 months of the
#   median monthly revenue in the training data.
# - A share `SAVE_RATE` of contacted churners stay because of the contact.
#
# We compare each contact budget with no campaign, where every churner leaves.
# Savings are shown per 1,000 customers and averaged over the ten validation folds.

# %%
CONTACT_COST = 20
CHURN_LOSS = 12 * train["MonthlyRevenue"].median()
SAVE_RATE = 0.20

cost_matrix = evaluation.cost_matrix(CONTACT_COST, CHURN_LOSS, SAVE_RATE)
print("Cost per customer")
display(cost_matrix.round(2))

outreach["Savings per 1,000 customers"] = evaluation.savings_per_1000(outreach, cost_matrix)
display(
    outreach.pivot_table(
        index="Contact share",
        columns="Model",
        values="Savings per 1,000 customers",
        aggfunc="mean",
    ).round(0)
)

# Contacting a customer pays off when precision exceeds this value
print(f"Break-even precision: {CONTACT_COST / (SAVE_RATE * CHURN_LOSS):.1%}")

# %% [markdown]
# ## Cost results
#
# **With these assumptions, contacting the top 10% chosen by the tuned model
# saves about $3,460 per 1,000 customers compared with no campaign.** The
# default model saves about $3,440. Savings are about $1,950 at 5% and $5,970
# at 20%.
#
# A contact pays off when precision exceeds 17.2%. Even random contact
# reaches 28.8%, so under these assumptions a larger budget always saves more.
# A higher contact cost or a lower save rate would raise this break-even point
# and favor a smaller budget. The result depends on the assumed amounts.

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
# ## What the importance shows
#
# `MonthlyMinutes` and `CurrentEquipmentDays` rank highest. Shuffling them
# lowers AP by 0.03797 and 0.02239 on average, followed by `UniqueSubs` with
# 0.01740. Each of the top 10 inputs lowers AP on all five folds when shuffled.
# Random forest and gradient boosting rank `MonthlyMinutes`,
# `PercChangeMinutes`, `CurrentEquipmentDays` and `MonthsInService` highest.
# The first three are also in our top four, with `PercChangeMinutes` fourth at
# 0.01387, but `MonthsInService` is not in our top 10. These results describe
# what this model uses to rank customers. They do not show that an input
# causes churn.

# %% [markdown]
# ## Test set evaluation
#
# We score the tuned logistic regression model, fitted on the full training
# partition, once on the reserved test customers. We fixed every setting above
# before this step and do not change any of them afterward. We place the test
# scores next to the mean scores on the ten validation folds. Similar values
# mean the validation results carry over to new customers.

# %%
test = data.load_test()
X_test = preprocessing.prepare_predictors(test)
y_test = test["churn"]
test_probability = tuned_pipeline.predict_proba(X_test)[:, 1]
print(f"{len(y_test):,} test customers, churn rate {y_test.mean():.1%}")

test_outreach = pd.DataFrame(
    [
        {
            "Contact share": share,
            **evaluation.outreach_scores(y_test, test_probability, share),
        }
        for share in CONTACT_SHARES
    ]
).set_index("Contact share")
test_outreach["Savings per 1,000 customers"] = evaluation.savings_per_1000(
    test_outreach, cost_matrix
)
validation_outreach = (
    outreach.loc[outreach["Model"].eq(tuned_name)]
    .groupby("Contact share")
    .mean(numeric_only=True)
)

scores = {
    "AP": (
        tuning.loc[chosen_setting, "AP mean"],
        sk_metrics.average_precision_score(y_test, test_probability),
    ),
    "ROC AUC": (
        tuning.loc[chosen_setting, "ROC AUC mean"],
        sk_metrics.roc_auc_score(y_test, test_probability),
    ),
}
for share in CONTACT_SHARES:
    for metric in ["Precision", "Recall", "Savings per 1,000 customers"]:
        scores[f"{metric}, top {share:.0%}"] = (
            validation_outreach.loc[share, metric],
            test_outreach.loc[share, metric],
        )

test_comparison = pd.DataFrame(scores, index=["Validation", "Test"]).T
test_comparison["Test − validation"] = test_comparison["Test"] - test_comparison["Validation"]
display(test_comparison.round(4))

# %%
test_confusion = pd.crosstab(
    y_test.map({1: "Churned", 0: "Stayed"}).rename("Actual"),
    pd.Series(evaluation.contact_top(test_probability, EVALUATION_SHARE), index=y_test.index)
    .map({True: "Contacted", False: "Not contacted"})
    .rename("Outreach"),
)
print(f"{tuned_name}, top {EVALUATION_SHARE:.0%} of test customers")
display(test_confusion)

# %% [markdown]
# ## Test results
#
# **On the 10,210 test customers, the tuned logistic regression model reaches
# AP 0.3919 and ROC AUC 0.6218, compared with 0.3979 and 0.6262 on the
# validation folds.** Contacting the top 5% gives 49.6% precision and 8.6%
# recall, the top 10% gives 45.2% and 15.7%, and the top 20% gives 41.3% and
# 28.7%. In the top 10%, we contact 461 churners and 560 customers who stayed,
# and 2,481 churners are not contacted. Test AP is 0.0060 below validation,
# less than the validation fold SD of 0.00722, so the gap is small, although
# precision in the top 10% falls from 46.9% to 45.2%.
