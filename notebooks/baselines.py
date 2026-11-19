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
# # Reference Baselines
#
# We use three simple references to give the trained model scores context.
#
# - **Constant churn rate.** We give every customer the same score, the share
#   of customers who churned in that fold's training rows.
# - **Stratified random.** We draw random labels using that same churn rate.
#   We use a fixed random seed so we can reproduce the scores.
# - **Fixed tenure rule.** We assign a score of 1 to customers with 11–12
#   months of service and 0 to everyone else, including missing tenure.

# %%
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display
from sklearn import dummy as sk_dummy
from sklearn import model_selection as sk_selection

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src import baselines, data, preprocessing

train = data.load_train()
X = preprocessing.prepare_predictors(train)[["MonthsInService"]]
y = train["churn"]

splitter = sk_selection.RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=data.RANDOM_STATE,
)

baseline_options = [
    ("Constant churn rate", sk_dummy.DummyClassifier(strategy="prior")),
    ("Stratified random", sk_dummy.DummyClassifier(
        strategy="stratified", random_state=data.RANDOM_STATE,
    )),
    ("Fixed tenure (11–12 months)", baselines.TenureRule()),
]

# %% [markdown]
# We use the same ten validation splits as the trained models. We report the
# mean and sample standard deviation for average precision (AP) and ROC AUC.

# %%
rows = []

for baseline_name, model in baseline_options:
    scores = sk_selection.cross_validate(
        model,
        X,
        y,
        cv=splitter,
        scoring=["average_precision", "roc_auc"],
        error_score="raise",
    )

    ap = scores["test_average_precision"]
    auc = scores["test_roc_auc"]
    rows.append([baseline_name, ap.mean(), ap.std(ddof=1), auc.mean(), auc.std(ddof=1)])

results = pd.DataFrame(
    rows,
    columns=["Baseline", "AP mean", "AP SD", "ROC AUC mean", "ROC AUC SD"],
)
display(results.round(4))

# %% [markdown]
# Both the constant score and stratified random guess give AP around 0.288. A
# constant score ties all customers, so its AP equals the churn rate. The
# fixed tenure rule reaches about 0.313. The 11–12 month window came from
# labelled training data, so we treat its score as an exploratory reference.
