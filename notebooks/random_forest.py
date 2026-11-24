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
# # Random Forest
#
# We predict churn from information available when we decide whom to contact.
# We estimate churn by averaging predictions from many trees.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn import base as sk_base
from sklearn import ensemble as sk_ensemble
from sklearn import inspection as sk_inspection
from sklearn import model_selection as sk_selection
from sklearn import pipeline as sk_pipeline

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src import data, evaluation, preprocessing

train = data.load_train()
X = preprocessing.prepare_predictors(train)
y = train["churn"]

N_SPLITS = 5
splitter = sk_selection.RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=2,
    random_state=data.RANDOM_STATE,
)
AP_TOLERANCE = 0.002
N_JOBS = 4

# %% [markdown]
# ## Preprocessing options
#
# We compare target encoding (estimated churn rate by service area) with frequency
# encoding (how common each area is). We compare filling missing numeric values
# with the training median or with zero.
#
# We fix the seed and build the trees on `N_JOBS` cores. Other settings use defaults.

# %%
model = sk_ensemble.RandomForestClassifier(random_state=data.RANDOM_STATE, n_jobs=N_JOBS)

encoding_options = [
    ("Target", preprocessing.target_encode_service_area()),
    ("Frequency", preprocessing.ServiceAreaFrequencyEncoder()),
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
        # Both area encoders produce one column, so dense storage is practical.
        feature_steps = preprocessing.combine_feature_steps(
            numeric=imputer,
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
# 0.42686 (SD 0.00589). The top mean, with target encoding and median,
# was higher by 0.00053 AP (paired SD 0.00616) and led on 7 folds. With
# frequency fixed, zero led on 6 folds and improved mean AP by 0.00022
# (paired SD 0.00511). Both mean gaps fall well within our near-tie range.
# We keep area counts and fill numeric blanks with typical observed values.
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
).set_params(sparse_threshold=0.0)
chosen_pipeline = sk_pipeline.make_pipeline(chosen_features, model)

# %% [markdown]
# ## Hyperparameter tuning
#
# We tune only the estimator in `chosen_pipeline`, using validation AP.
# The chosen preprocessing steps fit again on each fold's training rows.
#
# Hyperparameters are settings we choose before fitting. We tune `max_features`
# and `min_samples_leaf` to control how many inputs each split considers and the
# minimum size of tree leaves. Fewer inputs per split make the trees more
# different from each other, and larger leaves stop trees from fitting single
# customers. We keep 100 trees and the other defaults.
#
# We try every combination of four `max_features` values and five
# `min_samples_leaf` values on the same ten folds as the preprocessing
# comparison. We use the same near-tie rule as for preprocessing: among
# settings within 0.002 AP of the top mean, we choose the simplest one, with
# larger leaves first and then fewer inputs per split.

# %%
model_step = chosen_pipeline.steps[-1][0]
max_features_options = ["sqrt", 0.2, 0.33, 0.5]
min_samples_leaf_options = [1, 5, 10, 20, 50]

search = sk_selection.GridSearchCV(
    chosen_pipeline,
    param_grid={
        f"{model_step}__max_features": max_features_options,
        f"{model_step}__min_samples_leaf": min_samples_leaf_options,
    },
    scoring=["average_precision", "roc_auc"],
    cv=splitter,
    refit=False,
    error_score="raise",
)
# store the return value so the notebook does not try to draw the search object
_ = search.fit(X, y)

# %%
cv_results = pd.DataFrame(search.cv_results_)
n_folds = splitter.get_n_splits()
fold_ap = cv_results[[f"split{i}_test_average_precision" for i in range(n_folds)]]
fold_auc = cv_results[[f"split{i}_test_roc_auc" for i in range(n_folds)]]

tuning = pd.DataFrame(
    {
        "max_features": cv_results[f"param_{model_step}__max_features"].astype(str),
        "min_samples_leaf": cv_results[f"param_{model_step}__min_samples_leaf"].astype(int),
        "AP mean": fold_ap.mean(axis=1),
        "AP SD": fold_ap.std(axis=1, ddof=1),
        "ROC AUC mean": fold_auc.mean(axis=1),
        "ROC AUC SD": fold_auc.std(axis=1, ddof=1),
    }
)

# "sqrt" considers about 7 of the 53 inputs, so it is the smallest option
features_rank = {str(value): rank for rank, value in enumerate(max_features_options)}
near_tie = tuning["AP mean"] >= tuning["AP mean"].max() - AP_TOLERANCE
candidates = tuning[near_tie].assign(
    features_rank=tuning["max_features"].map(features_rank)
)
chosen_setting = candidates.sort_values(
    ["min_samples_leaf", "features_rank"], ascending=[False, True]
).index[0]

# On each fold we subtract the chosen setting's AP from each alternative's AP
paired_tuning = fold_ap.sub(fold_ap.loc[chosen_setting].to_numpy(), axis="columns")
tuning["AP difference"] = paired_tuning.mean(axis=1)
tuning["Difference SD"] = paired_tuning.std(axis=1, ddof=1)
tuning["Folds ahead"] = paired_tuning.gt(0).sum(axis=1)
tuning["Near tie"] = near_tie

print("Chosen setting:")
display(tuning.loc[[chosen_setting]].round(5))
display(tuning.sort_values("AP mean", ascending=False).round(5))

# %%
# show mean AP for each combination
display(
    tuning.pivot(index="min_samples_leaf", columns="max_features", values="AP mean")[
        [str(value) for value in max_features_options]
    ].round(4)
)

# %% [markdown]
# ## Tuning decision
#
# **We choose `max_features=0.2` and `min_samples_leaf=20`.** Mean AP was
# 0.44685 (SD 0.00681), up from 0.42856 with the default settings on the same
# folds. Leaves of a single customer gave the lowest AP for every
# `max_features` value, so the default trees fit the training rows too closely.
# The top mean, 0.33 with leaves of 10, was higher by 0.00179 AP (paired SD
# 0.00260) and led on 8 of 10 folds. This is inside our 0.002 near-tie range,
# so we keep the larger leaves and fewer inputs per split. Four settings fall
# in this range and their ROC AUC values differ by at most 0.0004.

# %%
tuned_pipeline = sk_base.clone(chosen_pipeline).set_params(
    **cv_results.loc[chosen_setting, "params"]
)
_ = tuned_pipeline.fit(X, y)

# %% [markdown]
# ## Model evaluation
#
# We compare the default and tuned forests with the chosen preprocessing.
# These development scores use the ten validation folds that informed tuning.

# %%
metrics = ["AP mean", "AP SD", "ROC AUC mean"]
comparison = pd.DataFrame(
    [results.loc[chosen_row, metrics], tuning.loc[chosen_setting, metrics]],
    index=["Default forest", "Tuned forest"],
)
display(comparison.round(5))

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
# We refit the default and tuned forests on the same ten training folds. The
# test set stays unused.

# %%
CONTACT_SHARES = [0.05, 0.10, 0.20]
EVALUATION_SHARE = 0.10

pipelines = {"Default forest": chosen_pipeline, "Tuned forest": tuned_pipeline}
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
        if name == "Tuned forest" and fold < N_SPLITS:
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
precision_gain = precision_by_fold["Tuned forest"] - precision_by_fold["Default forest"]
print("Tuned forest folds ahead on precision, by contact share:")
print(precision_gain.gt(0).groupby(level="Contact share").sum().to_string())

# %%
confusion = pd.crosstab(
    y.map({1: "Churned", 0: "Stayed"}).rename("Actual"),
    contacted_first_repeat.map({True: "Contacted", False: "Not contacted"}).rename("Outreach"),
)
print(f"Tuned forest, top {EVALUATION_SHARE:.0%} of each validation fold in the first repeat")
display(confusion)

# %% [markdown]
# ## Outreach results
#
# **Contacting the top 10%, 52.9% of contacted customers churn with the tuned
# forest (SD 1.4 points), compared with 28.8% for random contact (lift 1.83).**
# This reaches 18.3% of all churners. The default forest reached 50.7% precision.
# The tuned forest had higher precision on 8 of 10 folds at 5% and 10%, and on
# all 10 folds at 20%. A smaller budget raises precision but reaches fewer
# churners: 58.6% precision and 10.2% recall at 5%, and 47.3% and 32.8% at 20%.
#
# In the first repeat, the top 10% contained 2,162 churners and 1,923 customers
# who stayed. The other 9,607 churners were not contacted. The team still has
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
# **With these assumptions, contacting the top 10% chosen by the tuned forest
# saves about $4,150 per 1,000 customers compared with no campaign.** The
# default forest saves about $3,900. Savings are about $2,400 at 5% and $7,000
# at 20%, and the tuned forest saves more at every budget.
#
# A contact pays off when precision exceeds 17.2%. Even random contact
# reaches 28.8%, so under these assumptions a larger budget always saves more.
# A higher contact cost or a lower save rate would raise this break-even point
# and favor a smaller budget. The result depends on the assumed amounts.

# %% [markdown]
# ## Top 10 features by permutation importance
#
# We refit the tuned pipeline on each training fold in the first repeat. Its
# five validation folds cover every training customer once. On each validation
# fold, we shuffle each input five times and measure how much AP drops. We
# shuffle before feature encoding so that a categorical input and all its
# encoded columns count as one feature. We rank inputs by mean AP drop over
# the five folds. Correlated inputs can substitute for each other and share
# importance.

# %%
first_repeat = list(splitter.split(X, y))[:N_SPLITS]
permutation_rows = []
for fit_rows, valid_rows in first_repeat:
    fold_pipeline = sk_base.clone(tuned_pipeline).fit(X.iloc[fit_rows], y.iloc[fit_rows])
    permutation = sk_inspection.permutation_importance(
        fold_pipeline,
        X.iloc[valid_rows],
        y.iloc[valid_rows],
        scoring="average_precision",
        n_repeats=5,
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
# As a supplement, we read the forest's built-in `feature_importances_` from
# the tuned pipeline fitted on the full training partition, labelled with the
# fitted preprocessing's `get_feature_names_out()`. This impurity importance
# measures how much each column reduces uncertainty in the training splits. It
# can favor inputs with many split points, such as continuous values, and it
# splits the credit for a category across its one-hot columns.

# %%
fitted_features = tuned_pipeline[0]
fitted_forest = tuned_pipeline[-1]

impurity_importance = pd.Series(
    fitted_forest.feature_importances_,
    index=fitted_features.get_feature_names_out(),
    name="Impurity importance",
).sort_values(ascending=False)
impurity_importance.index.name = "Feature"
display(impurity_importance.head(10).round(5).to_frame())

# %% [markdown]
# ## What the importance measures show
#
# Both measures rank the same four inputs highest: `CurrentEquipmentDays`,
# `MonthsInService`, `MonthlyMinutes` and `PercChangeMinutes`. Shuffling one of
# them lowers validation AP by 0.024 to 0.043 on every fold. Shuffling any
# other single input lowers AP by less than 0.01.
#
# The measures disagree for some inputs. `MonthlyRevenue`, `ServiceArea` and
# `UnansweredCalls` rank fifth, seventh and twelfth by impurity, but shuffling
# them lowers AP by at most 0.002. They have many distinct values, which favors them in impurity
# importance. Revenue also overlaps with `TotalRecurringCharge` and
# `OverageMinutes`. `CreditRating` ranks sixth by permutation but does not reach
# the top 20 by impurity, because its credit is split across one-hot columns.
#
# Many household and lifestyle flags change AP by less than 0.0005. These
# results describe what this forest uses to rank customers. They do not show
# that an input causes churn.
