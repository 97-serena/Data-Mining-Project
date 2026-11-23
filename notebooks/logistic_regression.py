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
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn import base as sk_base
from sklearn import linear_model as sk_linear
from sklearn import metrics as sk_metrics
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
# ## How we measure precision
#
# Our team goal is to maximise precision: among the customers we contact, as many
# as possible should be real churners, so that we make few false positives.
# Precision depends on how many customers we contact, so we fix the contact
# budget. We rank customers by predicted churn probability and measure precision
# among the top 10%. We also report the top 5% and top 20%. Contacting customers
# at random gives a precision equal to the churn rate, about 0.288.

# %%
def precision_at_top(y_true, y_score, share):
    """Share of real churners among the `share` of customers with the highest scores."""
    contacted = int(np.ceil(share * len(y_score)))
    top = np.argsort(-np.asarray(y_score), kind="stable")[:contacted]
    return np.asarray(y_true)[top].mean()


def top_precision_scorer(share):
    return sk_metrics.make_scorer(precision_at_top, response_method="predict_proba", share=share)


budget_scoring = {
    "top10_precision": top_precision_scorer(0.10),
    "top5_precision": top_precision_scorer(0.05),
    "top20_precision": top_precision_scorer(0.20),
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
}

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
# precision (AP) to compare the preprocessing options. We also report ROC AUC,
# how much both scores vary across folds, and precision among the top 10%.

# %%
rows = []
ap_by_option = []
top10_by_option = []

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
                scoring={
                    "average_precision": "average_precision",
                    "roc_auc": "roc_auc",
                    "top10_precision": budget_scoring["top10_precision"],
                },
                error_score="raise",
            )

            ap = scores["test_average_precision"]
            ap_by_option.append(ap)
            auc = scores["test_roc_auc"]
            top10 = scores["test_top10_precision"]
            top10_by_option.append(top10)
            rows.append(
                [
                    encoding_name,
                    log_name,
                    imputation_name,
                    ap.mean(),
                    ap.std(ddof=1),
                    auc.mean(),
                    auc.std(ddof=1),
                    top10.mean(),
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
        "Top 10% precision",
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
# The choice also holds for our precision goal. It has the highest top-10%
# precision (0.46781), level with zero imputation. Every option without the log
# step or with one-hot encoding is at least 0.017 lower.
#
# On each fold we subtract our choice's AP from the alternative's AP. We show
# the mean and SD of those differences and count the folds where the alternative led.
# The last column does the same for top-10% precision.

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
paired_top10 = pd.DataFrame(top10_by_option).sub(top10_by_option[chosen_row], axis="columns")
results["Top 10% difference"] = paired_top10.mean(axis=1)
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
# Hyperparameters are settings we choose before fitting. We tune only the
# estimator in `chosen_pipeline`. The chosen preprocessing steps fit again on
# each fold's training rows, and we use the same ten folds as above.
#
# We tune `C`, which controls how strongly we limit large model weights. Smaller
# values limit them more. We also try `class_weight="balanced"`, which gives
# churners more weight during fitting. We pick the setting with the highest mean
# top-10% precision. On matching folds, we compare every setting with the
# defaults (`C=1`, no class weights).

# %%
def tuning_results(search, default):
    """Summarise a grid search fold by fold, with gains over the default setting."""
    cv = search.cv_results_

    def by_fold(metric):
        return np.column_stack([cv[f"split{i}_test_{metric}"] for i in range(search.n_splits_)])

    top10 = by_fold("top10_precision")
    default_row = next(i for i, params in enumerate(cv["params"]) if params == default)
    gain = top10 - top10[default_row]

    table = pd.DataFrame(cv["params"]).rename(columns=lambda name: name.split("__")[-1])
    table["Top 10% precision"] = top10.mean(axis=1)
    table["SD"] = top10.std(axis=1, ddof=1)
    table["Gain vs default"] = gain.mean(axis=1)
    table["Gain SD"] = gain.std(axis=1, ddof=1)
    table["Folds ahead"] = (gain > 0).sum(axis=1)
    table["Top 5%"] = by_fold("top5_precision").mean(axis=1)
    table["Top 20%"] = by_fold("top20_precision").mean(axis=1)
    table["AP"] = by_fold("average_precision").mean(axis=1)
    table["ROC AUC"] = by_fold("roc_auc").mean(axis=1)
    return table, default_row


param_grid = {
    "logisticregression__C": [
        0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
    ],
    "logisticregression__class_weight": [None, "balanced"],
}
default_params = {"logisticregression__C": 1.0, "logisticregression__class_weight": None}

search = sk_selection.GridSearchCV(
    chosen_pipeline,
    param_grid,
    scoring=budget_scoring,
    refit=False,
    cv=splitter,
    n_jobs=-1,
    error_score="raise",
)
search.fit(X, y)

tuning, default_row = tuning_results(search, default_params)
tuning["class_weight"] = tuning["class_weight"].fillna("none")
best_row = tuning["Top 10% precision"].idxmax()
best_params = search.cv_results_["params"][best_row]
display(tuning.round(4))
display(tuning.loc[[best_row, default_row]].round(4))

# %%
plot_colors = {"none": "#2a78d6", "balanced": "#eb6834"}
fig, ax = plt.subplots(figsize=(8, 4.2))
for weight, color in plot_colors.items():
    part = tuning[tuning["class_weight"].eq(weight)]
    ax.plot(part["C"], part["Top 10% precision"], color=color, linewidth=2, marker="o", markersize=5)
    ax.annotate(
        f"class_weight: {weight}",
        (part["C"].iloc[-1], part["Top 10% precision"].iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color="#52514e",
    )
ax.axvline(1.0, color="#b4b3ad", linewidth=1, linestyle="--")
ax.annotate("default C", (1.0, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points", color="#52514e")
ax.set_xscale("log")
ax.set_xlabel("C (smaller means stronger limits on the weights)")
ax.set_ylabel("Mean top-10% precision")
ax.set_title(
    "Logistic regression: top-10% precision by C (narrow scale: all values within 0.01)",
    loc="left",
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Tuning decision
#
# **We choose `C=0.003` with no class weights.** It had the highest mean top-10%
# precision, 0.4726 (SD 0.0176), against 0.4678 for the defaults. That is a gain
# of 0.0048 (paired SD 0.0064), and it led on 8 of 10 folds. AP and ROC AUC
# barely change (0.3991 against 0.3980, and 0.6273 against 0.6266).
#
# Settings from `C=0.002` to `C=0.005` score about the same, and precision falls
# quickly below `C=0.002`. Stronger limits on the weights help because many
# inputs overlap, such as the different call counts. Class weights do not help:
# weighting churners changes the predicted probabilities but barely changes who
# ranks in the top 10%. Every difference in the table is below one percentage
# point, so the chart uses a narrow scale.

# %%
tuned_pipeline = sk_base.clone(chosen_pipeline).set_params(**best_params)
tuned_pipeline

# %% [markdown]
# ## Checking the gain on held-out folds
#
# Picking the best of many settings on the same folds makes the winner's score
# a little optimistic. To measure how much of the gain is real, we repeat the
# whole tuning inside each of five outer folds, using only that fold's training
# rows. We then score the pick on the outer validation rows, which the tuning
# never saw. We also score the defaults and our chosen setting on the same rows.
# The outer folds use a different seed from our shared folds.

# %%
outer = sk_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
inner = sk_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=data.RANDOM_STATE)
top10_scorer = budget_scoring["top10_precision"]

held_out = []
for fold, (fit_rows, check_rows) in enumerate(outer.split(X, y), start=1):
    X_fit, y_fit = X.iloc[fit_rows], y.iloc[fit_rows]
    X_check, y_check = X.iloc[check_rows], y.iloc[check_rows]
    inner_search = sk_selection.GridSearchCV(
        chosen_pipeline, param_grid, scoring=top10_scorer, cv=inner, n_jobs=-1, error_score="raise"
    ).fit(X_fit, y_fit)
    held_out.append(
        {
            "Outer fold": fold,
            "Inner pick: C": inner_search.best_params_["logisticregression__C"],
            "Inner pick: class weight": inner_search.best_params_["logisticregression__class_weight"] or "none",
            "Default": top10_scorer(sk_base.clone(chosen_pipeline).fit(X_fit, y_fit), X_check, y_check),
            "Inner pick": top10_scorer(inner_search.best_estimator_, X_check, y_check),
            "Our setting": top10_scorer(sk_base.clone(tuned_pipeline).fit(X_fit, y_fit), X_check, y_check),
        }
    )

held_out = pd.DataFrame(held_out)
held_out["Inner pick gain"] = held_out["Inner pick"] - held_out["Default"]
held_out["Our setting gain"] = held_out["Our setting"] - held_out["Default"]
display(held_out.round(4))
display(
    pd.DataFrame(
        {
            "Mean gain": [
                held_out["Inner pick gain"].mean(),
                held_out["Our setting gain"].mean(),
                tuning.loc[best_row, "Gain vs default"],
            ],
            "Ahead of default": [
                f"{(held_out['Inner pick gain'] > 0).sum()} of 5 outer folds",
                f"{(held_out['Our setting gain'] > 0).sum()} of 5 outer folds",
                f"{tuning.loc[best_row, 'Folds ahead']} of 10 tuning folds",
            ],
        },
        index=["Inner pick, held-out rows", "Our setting, held-out rows", "Our setting, tuning folds"],
    ).round(4)
)

# %% [markdown]
# On rows the tuning never saw, our setting beat the defaults by 0.0020 on
# average and led on 3 of 5 outer folds. The inner tuning picked `C` between
# 0.0015 and 0.01 every time, never the default, but its picks gained only 0.0015
# on the held-out rows. About half of the gain we measured on the tuning folds
# came from picking the best of many settings. Tuning changes logistic regression
# only a little: we expect about 0.2 percentage points more top-10% precision on
# new customers.

# %% [markdown]
# ## Precision at contact budgets
#
# To see what the tuned model means for outreach, we give every training customer
# a churn probability from a pipeline fitted without them. We do this for both
# repeats of our shared folds and average the results. We then rank customers and
# count real churners and false positives at each budget. The cut-off is the
# lowest predicted probability among the contacted customers. The reserved test
# split stays untouched.

# %%
def out_of_fold_scores(pipeline):
    """One validation-fold probability per customer for each repeat of the shared folds."""
    folds_per_repeat = splitter.get_n_splits() // splitter.n_repeats
    scores = np.full((splitter.n_repeats, len(y)), np.nan)
    for i, (fit_rows, check_rows) in enumerate(splitter.split(X, y)):
        fitted = sk_base.clone(pipeline).fit(X.iloc[fit_rows], y.iloc[fit_rows])
        scores[i // folds_per_repeat, check_rows] = fitted.predict_proba(X.iloc[check_rows])[:, 1]
    return scores


def budget_table(scores, shares):
    """Contacted customers, churners caught and false positives at each budget."""
    churn = y.to_numpy()
    rows = []
    for share in shares:
        per_repeat = []
        for repeat_scores in scores:
            contacted = int(np.ceil(share * len(churn)))
            order = np.argsort(-repeat_scores, kind="stable")
            caught = churn[order[:contacted]].sum()
            per_repeat.append(
                [
                    contacted,
                    caught,
                    contacted - caught,
                    caught / contacted,
                    caught / churn.sum(),
                    repeat_scores[order[contacted - 1]],
                ]
            )
        rows.append([share, *np.mean(per_repeat, axis=0)])
    return pd.DataFrame(
        rows,
        columns=["Budget", "Contacted", "Churners caught", "False positives", "Precision", "Recall", "Cut-off"],
    )


def largest_budget(scores, target, min_contacted=100):
    """Largest share we can contact while keeping precision at or above the target."""
    churn = y.to_numpy()
    per_repeat = []
    for repeat_scores in scores:
        order = np.argsort(-repeat_scores, kind="stable")
        precision = np.cumsum(churn[order]) / np.arange(1, len(churn) + 1)
        allowed = np.flatnonzero(precision >= target)
        allowed = allowed[allowed >= min_contacted - 1]
        contacted = allowed[-1] + 1
        per_repeat.append(
            [
                contacted / len(churn),
                churn[order[:contacted]].sum() / churn.sum(),
                repeat_scores[order[contacted - 1]],
            ]
        )
    return np.mean(per_repeat, axis=0)


tuned_scores = out_of_fold_scores(tuned_pipeline)
display(budget_table(tuned_scores, [0.05, 0.10, 0.20]).round(4))

targets = [0.40, 0.45, 0.50, 0.55]
display(
    pd.DataFrame(
        [largest_budget(tuned_scores, target) for target in targets],
        index=pd.Index(targets, name="Precision at least"),
        columns=["Largest budget", "Recall", "Cut-off"],
    ).round(4)
)

flagged = tuned_scores >= 0.5
churn = y.to_numpy()
display(
    pd.DataFrame(
        {
            "Share flagged": [flagged.mean()],
            "Precision": [np.mean([churn[f].mean() for f in flagged])],
            "Recall": [np.mean([churn[f].sum() / churn.sum() for f in flagged])],
        },
        index=["Default 0.5 cut-off"],
    ).round(4)
)

# %%
shares = np.linspace(0.01, 0.5, 50)
curve = budget_table(tuned_scores, shares)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(curve["Budget"], curve["Precision"], color="#2a78d6", linewidth=2)
ax.plot(curve["Budget"], curve["Recall"], color="#eb6834", linewidth=2)
ax.axhline(y.mean(), color="#8a8984", linewidth=1.2, linestyle="--")
for share in (0.05, 0.10, 0.20):
    ax.axvline(share, color="#e6e5e1", linewidth=1, zorder=0)
end = curve.iloc[-1]
ax.annotate("Precision", (end["Budget"], end["Precision"]), xytext=(6, 0), textcoords="offset points", va="center", color="#52514e")
ax.annotate("Recall", (end["Budget"], end["Recall"]), xytext=(6, 0), textcoords="offset points", va="center", color="#52514e")
ax.annotate("Random contact", (0.30, y.mean()), xytext=(0, 4), textcoords="offset points", va="bottom", color="#52514e")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
ax.set_xticks([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
ax.set_ylim(0, 0.7)
ax.set_xlabel("Share of customers contacted, highest predicted risk first")
ax.set_ylabel("Share")
ax.set_title("Tuned logistic regression: precision and recall by contact budget", loc="left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## What the tuned model means for outreach
#
# Among the top 10% of customers (4,084), the tuned model catches 1,929 churners
# and makes 2,155 false positives: a precision of 0.472 and a recall of 0.164.
# That is 528 false positives per 1,000 contacts, against 712 for random contact.
# A smaller budget raises precision but catches fewer churners. Precision stays
# at or above 0.50 only up to the top 6.3% of customers (recall 0.109), and at or
# above 0.45 up to the top 14.7% (recall 0.229).
#
# The default 0.5 cut-off is not useful here. It flags only 1.7% of customers
# and catches 3.3% of churners. To contact the top 10%, we use a cut-off of about
# 0.39 instead. These cut-offs come from the training folds. We confirm them on
# the reserved test split once all five models are tuned and the team fixes the
# contact budget.
