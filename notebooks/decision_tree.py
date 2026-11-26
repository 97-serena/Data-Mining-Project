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
import numpy as np
import pandas as pd
from IPython.display import display
from joblib import parallel_config
from sklearn import base as sk_base
from sklearn import inspection as sk_inspection
from sklearn import metrics as sk_metrics
from sklearn import model_selection as sk_selection
from sklearn import pipeline as sk_pipeline
from sklearn import preprocessing as sk_preprocessing
from sklearn import tree as sk_tree

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
# We compare three ways to encode service area. Target encoding estimates churn
# rates, frequency encoding measures how common each area is, and one-hot
# encoding creates a 0/1 column per area. We compare filling missing numeric values
# with the training median or with zero.
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
# and balanced training. The depth grid extends to 15 to compare deeper trees
# with depth 10.
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

# %% [markdown]
# ### Tolerance check
#
# We apply the same selection rule at 0.001, 0.002 and 0.003 AP using the
# existing grid scores. AP differences are relative to the choice at 0.002,
# which remains the tolerance used to select the tuned tree.

# %%
tolerance_rows = []
for tolerance in [0.001, AP_TOLERANCE, 0.003]:
    within_tolerance = tuning["AP mean"] >= tuning["AP mean"].max() - tolerance
    setting = (
        tuning.loc[within_tolerance]
        .assign(weighted=lambda table: table["Class weight"].ne("Unweighted"))
        .sort_values(
            ["Max depth", "Min samples per leaf", "weighted", "AP mean"],
            ascending=[True, False, True, False],
        )
        .index[0]
    )
    tolerance_rows.append(
        {
            "Tolerance": tolerance,
            "Max depth": tuning.loc[setting, "Max depth"],
            "Min samples per leaf": tuning.loc[setting, "Min samples per leaf"],
            "Class weight": tuning.loc[setting, "Class weight"],
            "AP mean": tuning.loc[setting, "AP mean"],
            "AP difference": tuning.loc[setting, "AP difference"],
        }
    )
print(f"Chosen setting by tolerance, AP difference to the {AP_TOLERANCE} choice")
display(pd.DataFrame(tolerance_rows).round(5))

# %% [markdown]
# All three tolerances select the same unweighted tree with depth 10 and at
# least 250 customers per leaf. Its mean validation AP is 0.39984, about
# 0.00086 below the best mean AP in the grid.

# %%
tuned_pipeline = sk_base.clone(chosen_pipeline).set_params(
    **cv_results.loc[chosen_setting, "params"]
)
_ = tuned_pipeline.fit(X, y)

# %% [markdown]
# ## Model evaluation
#
# We compare the default and tuned trees with the chosen preprocessing on
# the same ten validation folds used for tuning. The test section at the end
# evaluates this tree separately from the [model comparison](model_comparison.py).

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
# ## Outreach at a fixed contact budget
#
# AP and ROC AUC do not need a cut-off. To describe a retention campaign, we
# rank each validation fold by predicted churn probability and contact the top
# 10% of customers. We also show 5% and 20% to see how results change with the
# budget. Precision is the share of contacted customers who churn, and recall
# is the share of all churners we contact. F1 combines precision and recall
# through their harmonic mean. Lift is precision divided by the fold's churn
# rate. Random contact gives a lift of 1.
#
# We refit the default and tuned models on the same ten training folds. This
# analysis uses only the validation rows from those splits.

# %%
CONTACT_SHARES = [0.05, 0.10, 0.20]
EVALUATION_SHARE = 0.10

default_name, tuned_name = comparison.index
pipelines = {default_name: chosen_pipeline, tuned_name: tuned_pipeline}
outreach_rows = []
tuned_fold_scores = []
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
        if name == tuned_name:
            tuned_fold_scores.append((y_valid, churn_probability))
            if fold < N_SPLITS:
                contacted_first_repeat.iloc[valid_rows] = evaluation.contact_top(
                    churn_probability, EVALUATION_SHARE
                )

outreach = pd.DataFrame(outreach_rows)
display(
    outreach.groupby(["Contact share", "Model"])[["Precision", "Recall", "F1", "Lift"]]
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
# **Contacting the top 10%, 46.7% of contacted customers churn with the tuned
# tree (SD 1.6 points), compared with 28.8% for random contact (lift 1.62).**
# This reaches 16.2% of all churners. The default tree's precision was 34.3%.
# Mean F1 at this budget rose from 0.177 to 0.241 after tuning. The tuned tree
# had higher precision on all 10 folds at every budget. A smaller
# budget raises precision but reaches fewer churners: 50.6% precision and 8.8%
# recall at 5%, and 42.5% and 29.5% at 20%.
#
# The default tree's precision stays near 35% at every budget. Its leaves each
# hold a single class, so it gives every customer a score of 0 or 1. About 30%
# of customers score 1, so every budget up to 20% picks customers from that
# group without ranking them.
#
# In the first repeat, the top 10% contained 1,890 churners and 2,195 customers
# who stayed. The other 9,879 churners were not contacted.

# %% [markdown]
# ## The threshold that maximizes F1
#
# On each validation fold we use the tuned tree's existing predictions to
# find the score threshold that gives the highest F1. We flag customers at
# or above that threshold. If thresholds tie, we choose the highest one
# to keep the contact list shorter. Finding and evaluating the threshold on
# the same fold makes this an exploratory comparison of contact list sizes.

# %%
f1_rows = []
for y_valid, churn_probability in tuned_fold_scores:
    precision, recall, thresholds = sk_metrics.precision_recall_curve(
        y_valid, churn_probability
    )
    # The final precision and recall pair has no corresponding threshold.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = len(thresholds) - 1 - f1[::-1].argmax()
    threshold = thresholds[best]
    f1_rows.append(
        {
            "Threshold": threshold,
            "Share flagged": (churn_probability >= threshold).mean(),
            "Precision": precision[best],
            "Recall": recall[best],
            "F1": f1[best],
        }
    )

f1_threshold_results = pd.DataFrame(f1_rows)
print("Tuned tree at the F1-maximizing threshold, ten validation folds")
display(f1_threshold_results.agg(["mean", "std"]).round(4))

# %% [markdown]
# The F1-maximizing thresholds flag 69.2% of customers on average (SD 6.0
# percentage points). Mean F1 is 0.4841, with 34.4% precision and 82.3%
# recall. This would require about seven times the contact capacity of the
# top-10% policy, whose mean F1 is 0.2408.

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
# **With these assumptions, contacting the top 10% chosen by the tuned tree
# saves about $3,440 per 1,000 customers compared with no campaign.** The
# default tree saves about $2,000. Savings are about $1,940 at 5% and $5,880
# at 20%, and the tuned tree saves more at every budget.
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

# %% [markdown]
# ## What the importance measures show
#
# Both measures rank `CurrentEquipmentDays` and `MonthsInService` first and
# second. Shuffling them lowers validation AP by 0.05835 and 0.04301 on
# average, respectively. `MonthlyMinutes` and `PercChangeMinutes` also rank in
# the top four by both measures. Shuffling any of these four inputs lowers AP
# on all five folds in the first repeat.
#
# The measures can give different impressions of an input. `HandsetWebCapable`
# accounts for 1.73% of the fitted tree's impurity reduction, but shuffling it
# lowers validation AP by just 0.00064 on average. Permutation measures changes
# in validation ranking, while impurity measures split credit in the fitted
# tree. Correlated inputs may substitute for one another, and the tree can
# split credit for a category across multiple encoded columns. These results
# describe model behavior and cannot establish what causes churn.

# %% [markdown]
# ## Test set evaluation
#
# We score the tuned tree, fitted on the full training partition, on the
# reserved test customers. Validation determines the model settings. We
# apply the same contact shares and cost assumptions on the test set and
# compare its scores with the means across the ten validation folds.

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

test_scores = {
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
    for metric in ["Precision", "Recall", "F1", "Lift", "Savings per 1,000 customers"]:
        test_scores[f"{metric}, top {share:.0%}"] = (
            validation_outreach.loc[share, metric],
            test_outreach.loc[share, metric],
        )

test_comparison = pd.DataFrame(test_scores, index=["Validation", "Test"]).T
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
# ## Test set results
#
# On 10,210 test customers, the tuned tree reaches AP 0.4038 and ROC AUC
# 0.6335. Compared with the validation means of 0.3998 and 0.6392, test AP is
# higher by 0.0040 and test ROC AUC is lower by 0.0056.
#
# At the 10% contact budget, test precision is 49.9%, recall is 17.3% and
# F1 is 0.2569, compared with 46.7%, 16.2% and 0.2408 on validation. The
# list contains 509 churners and 512 customers who stayed, and 2,433
# churners are missed. Under the stated cost assumptions, the test list
# saves about $3,799 per 1,000 customers compared with no outreach. Test
# precision is 53.9% at the 5% budget and 42.8% at the 20% budget.
