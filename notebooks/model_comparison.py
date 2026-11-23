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
# # Model Comparison
#
# We want to find customers at risk of churning before deciding whom to contact.
# We choose the preprocessing first and tune the models afterward. Both decisions
# use the training partition.

# %% [markdown]
# ## Why we use average precision
#
# About 28.8% of our training customers churn. As we move down a ranked outreach
# list, precision is the share of flagged customers who churn and recall is
# the share of all churners we flag. Average precision (AP) summarizes how well
# we maintain precision as recall increases. The constant-score reference gives
# AP of about 0.288. We rank options by mean validation AP and report ROC AUC
# alongside it to show how well churners rank above customers who stay.

# %% [markdown]
# ## Lock preprocessing, then tune the models
#
# We use the same five stratified folds and two repeats to compare preprocessing
# with fixed, untuned estimators. The model notebooks use scikit-learn defaults,
# fix random seeds where applicable, and give logistic regression enough
# iterations to converge. Each option is fitted inside each fold.
#
# We look at the mean AP and subtract scores on matching folds to see how often
# and by how much one option leads. A mean gap of at most 0.002 AP (0.2
# percentage points) is a practical near-tie for this project. We favor a
# simpler choice when scores are this close. The paired standard deviation
# shows how much the differences vary across overlapping folds, so we use it
# descriptively.
#
# We fixed these choices from the development comparisons. Each model
# notebook records its paired AP differences and defines its chosen pipeline.
#
# | Model | Service-area encoding | Numeric steps |
# | --- | --- | --- |
# | [Logistic regression](logistic_regression.py) | Target | Log, median fill, scaling |
# | [Gaussian naive Bayes](gaussian_naive_bayes.py) | Frequency | Log, median fill |
# | [Decision tree](decision_tree.py) | Frequency | Median fill |
# | [Random forest](random_forest.py) | Frequency | Median fill |
# | [Histogram gradient boosting](histogram_gradient_boosting.py) | Frequency | Native missing values |
#
# Next we will vary the model hyperparameters and use validation AP to compare
# them. Each run will fit the chosen preprocessing on that fold's training rows.
# We will use the training results to fix the model settings and our preferred
# model before opening the reserved test set.

# %% [markdown]
# ## Evaluate the final choices
#
# We will agree on the contact budget and how to use customer value measured
# before outreach. Once we fix the decisions, we will fit the chosen pipelines
# on the full training partition and score them and the
# [reference baselines](baselines.py) on the same reserved test customers.
# We will report AP and ROC AUC and show precision and recall at our contact
# budget.
