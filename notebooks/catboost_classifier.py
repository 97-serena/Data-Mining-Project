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
# # CatBoost
#
# We predict churn from information available when we decide whom to contact.
# CatBoost fits a sequence of trees and can use categorical inputs directly.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from IPython.display import display
from joblib import parallel_config
from sklearn import base as sk_base
from sklearn import compose as sk_compose
from sklearn import inspection as sk_inspection
from sklearn import model_selection as sk_selection
from sklearn import pipeline as sk_pipeline

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src import data, evaluation, preprocessing
from src.features import CATEGORICAL_COLUMNS, SERVICE_AREA

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
# We compare CatBoost's native handling of service area with frequency and
# target encoding. The other predictors pass through, with missing numeric
# values handled by CatBoost. Every option uses CatBoost's default training
# hyperparameters, including its automatic learning rate. We fix the seed
# for repeatability.

# %%
# Prefer native handling, then frequency encoding, when AP scores are close.
encoding_options = [
    ("Native", "passthrough"),
    ("Frequency", preprocessing.ServiceAreaFrequencyEncoder()),
    ("Target", preprocessing.target_encode_service_area()),
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
    # Preserve the other predictors and keep their column names
    feature_steps = sk_compose.ColumnTransformer(
        [("service_area", encoder, [SERVICE_AREA])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    cat_features = tuple(CATEGORICAL_COLUMNS)
    if encoding_name == "Native":
        cat_features += (SERVICE_AREA,)

    model = CatBoostClassifier(
        cat_features=cat_features,
        random_seed=data.RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
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
    auc = scores["test_roc_auc"]
    ap_by_option.append(ap)
    rows.append([
        encoding_name,
        ap.mean(),
        ap.std(ddof=1),
        auc.mean(),
        auc.std(ddof=1),
    ])

results = pd.DataFrame(
    rows,
    columns=["Encoding", "AP mean", "AP SD", "ROC AUC mean", "ROC AUC SD"],
)

# %% [markdown]
# ## Preprocessing decision
#
# Among options within 0.002 AP of the best, we prefer native handling of
# service area, then frequency encoding, then target encoding.
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
# ## Hyperparameter tuning
#
# We refit the chosen preprocessing on each fold's training rows. We
# compare tree depths and L2 leaf regularization at 500 boosting rounds and
# a learning rate of 0.05. These settings apply to the tuning grid.
#
# We compare the grid with the default model on the same ten validation folds.
# To carry a grid candidate forward, its mean AP must be within 0.002 of the
# highest score across both. Among eligible grid candidates, we prefer
# shallower trees, then stronger regularization. If every grid candidate
# falls outside that range, we carry the default model forward and report
# the highest-scoring grid candidate for comparison.

# %%
model_step = chosen_pipeline.steps[-1][0]
tuning_pipeline = sk_base.clone(chosen_pipeline).set_params(
    **{f"{model_step}__iterations": 500, f"{model_step}__learning_rate": 0.05}
)
param_grid = {
    f"{model_step}__depth": [5, 6],
    f"{model_step}__l2_leaf_reg": [3, 10],
}

search = sk_selection.GridSearchCV(
    estimator=tuning_pipeline,
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
        "Depth": cv_results[f"param_{model_step}__depth"],
        "L2 regularization": cv_results[f"param_{model_step}__l2_leaf_reg"],
        "AP mean": cv_results["mean_test_average_precision"],
        "AP SD": fold_ap.std(axis=1, ddof=1),
        "ROC AUC mean": cv_results["mean_test_roc_auc"],
    }
)
default_ap = results.loc[chosen_row, "AP mean"]
best_grid_ap = tuning["AP mean"].max()
best_ap = max(default_ap, best_grid_ap)
tuning["Near tie"] = tuning["AP mean"] >= best_ap - AP_TOLERANCE
eligible = tuning.loc[tuning["Near tie"]]
if eligible.empty:
    chosen_setting = tuning["AP mean"].idxmax()
else:
    chosen_setting = eligible.sort_values(
        ["Depth", "L2 regularization", "AP mean"], ascending=[True, False, False]
    ).index[0]

tuning["AP difference"] = fold_ap.sub(fold_ap.loc[chosen_setting], axis="columns").mean(axis=1)
chosen_ap = tuning.loc[chosen_setting, "AP mean"]
print(f"Default mean AP {default_ap:.5f} and highest grid mean AP {best_grid_ap:.5f}")
print(f"Reporting grid candidate at {chosen_ap:.5f} AP (gap to best {best_ap - chosen_ap:.5f})")
display(tuning.loc[[chosen_setting]].round(5))
display(tuning.sort_values("AP mean", ascending=False).round(5))

# %%
tuned_pipeline = sk_base.clone(tuning_pipeline).set_params(
    **cv_results.loc[chosen_setting, "params"]
)

# %% [markdown]
# ## Model evaluation
#
# We compare the default and tuned estimators with the chosen preprocessing.
# These development scores use the ten validation folds that informed tuning.

# %%
metrics = ["AP mean", "AP SD", "ROC AUC mean"]
comparison = pd.DataFrame(
    [results.loc[chosen_row, metrics], tuning.loc[chosen_setting, metrics]],
    index=["Default CatBoost", "Tuned CatBoost"],
)
default_name, tuned_name = comparison.index
pipelines = {default_name: chosen_pipeline, tuned_name: tuned_pipeline}
selected_name = tuned_name if tuning.loc[chosen_setting, "Near tie"] else default_name
selected_pipeline = pipelines[selected_name]
display(comparison.round(5))
print(f"Carry forward {selected_name} to the model comparison.")

ap_gain = fold_ap.loc[chosen_setting].to_numpy() - ap_by_option[chosen_row]
print(
    f"The grid candidate changes mean AP by {ap_gain.mean():+.5f} from defaults "
    f"and scores higher on {(ap_gain > 0).sum()} of {n_folds} folds."
)
_ = selected_pipeline.fit(X, y)

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
        if name == selected_name and fold < N_SPLITS:
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
print(f"{selected_name}, top {EVALUATION_SHARE:.0%} of each validation fold in the first repeat")
display(confusion)

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
# ## Top 10 features by permutation importance
#
# We refit the selected pipeline on each training fold in the first repeat. Its
# five validation folds cover every training customer once. On each validation
# fold, we shuffle each input five times and measure how much AP drops. We
# shuffle before feature encoding so that a categorical input and all its
# encoded columns count as one feature. We rank inputs by mean AP drop over
# the five folds. Correlated inputs can substitute for each other and share
# importance.

# %%
first_repeat = list(splitter.split(X, y))[:N_SPLITS]
permutation_rows = []
with parallel_config(backend="loky", inner_max_num_threads=1):
    for fit_rows, valid_rows in first_repeat:
        fold_pipeline = sk_base.clone(selected_pipeline).fit(X.iloc[fit_rows], y.iloc[fit_rows])
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
# ## Final test evaluation
#
# Before using the reserved test split, we use the training partition to
# settle the model comparison and fix the outreach budget. We then fit the
# frozen pipelines on the full training partition and score them and the
# baselines on the same test customers. We report test AP and ROC AUC, plus
# precision and recall at the agreed budget. See [Model comparison](model_comparison.py).
