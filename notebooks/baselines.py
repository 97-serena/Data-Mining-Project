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
#
# We chose the tenure window while exploring the labelled training data,
# so we use its validation score as a descriptive reference.

# %%
from pathlib import Path
import sys

from IPython.display import display

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.baselines import TenureRule, constant_churn_baseline, stratified_random_baseline
from src.data import load_train
from src.evaluation import compare_models
from src.preprocessing import prepare_predictors

train = load_train()
X = prepare_predictors(train)[["MonthsInService"]]
y = train["churn"]

baselines = {
    "Constant churn rate": constant_churn_baseline(),
    "Stratified random": stratified_random_baseline(),
    "Fixed tenure (11–12 months)": TenureRule(),
}

# %% [markdown]
# We use the same ten validation splits as the trained models. We report the
# mean and sample standard deviation for average precision, which summarizes
# how well we identify churners, and ROC AUC, which measures how well we rank
# churners above customers who stay.

# %%
results = compare_models(baselines, X, y).rename_axis("Baseline")
display(results.round(4))
