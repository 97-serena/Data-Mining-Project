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
# We will compare our chosen pipelines, the preprocessing and model steps used
# together, to see how well they predict churn before we decide whom to contact.

# %% [markdown]
# ## Finalize our choices
#
# We will choose preprocessing and tune settings for each model using only the
# training partition. Before opening the test partition, we will record each
# pipeline, choose our preferred model, and fix any risk threshold or contact limit.

# %% [markdown]
# ## Compare on the reserved test data
#
# We will run `prepare_predictors()` on both partitions. We will fit each
# chosen pipeline on the full training partition, then score them all on the
# same reserved test customers in a single evaluation. We will
# report average precision, our main metric for finding churners while limiting
# false alarms, and ROC AUC, the chance that a churner ranks above a customer who
# stays. We will also score our [reference baselines](baselines.py).

# %% [markdown]
# ## Prioritize outreach
#
# We will agree how to measure customer value using information available before
# contact. We will then combine customer value and churn risk to rank customers
# for outreach.
