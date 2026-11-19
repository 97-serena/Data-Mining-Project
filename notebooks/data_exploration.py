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
# # Data Exploration
#
# Our goal is to rank customers by churn risk before choosing whom to contact.
# We use the training partition to understand what information we have at that
# point, how it is recorded, and which patterns a model might pick up. We start
# from the same prepared inputs as the model notebooks and reserve the test
# partition for final evaluation.

# %%
import itertools
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython import display as ipy
from matplotlib import ticker as mpl_ticker
from sklearn import metrics as sk_metrics

project_root = pathlib.Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src import data, features, preprocessing

train = data.load_train()
X = preprocessing.prepare_predictors(train)
y = train["churn"]

print(f"Training customers: {len(X):,}   Predictors: {X.shape[1]}   Churn: {y.mean():.1%}")
print("Excluded inputs:", ", ".join(train.columns.difference(X.columns, sort=False)))

# %% [markdown]
# ## What changes before modelling?
#
# We first check that each input could be used before outreach. Churn is the
# outcome we want to predict, customer ID identifies the record, and the three
# retention fields have uncertain timing. We exclude all of them from the inputs.
#
# Next we compare blanks before and after preparation. Nonpositive household
# ages and negative equipment ages become missing because the values cannot
# represent valid ages. Blank categories get a dedicated label. Existing
# categories such as “Unknown” and income code 0 keep their recorded meaning.

# %%
fields = ["AgeHH1", "AgeHH2", "CurrentEquipmentDays", features.SERVICE_AREA]
raw_missing = train[fields].isna().sum().rename("Raw blanks")
prepared_missing = (
    X[fields].isna() | X[fields].eq(preprocessing.MISSING_CATEGORY)
).sum().rename("Missing after preparation")
ipy.display(pd.concat([raw_missing, prepared_missing], axis=1))

# %% [markdown]
# We found many more missing ages than the raw blanks suggest. After applying
# the rules, the two household age fields are missing for about 29% and 53%
# of customers. That makes the treatment of blanks worth testing. Each model
# compares zero with the training-fold median and retains missing indicators.

# %% [markdown]
# ## Distributions of the prepared inputs
#
# We check the range and typical size of numeric values to see which ones have
# a long upper tail. The median describes a typical value, and the 99th
# percentile shows how large values get near the top. We also check how common
# each yes/no flag is once it is represented as 1/0.

# %%
numeric = X[list(features.NUMERIC_COLUMNS)]
numeric_summary = numeric.describe(percentiles=[0.5, 0.99]).T
numeric_summary["Missing"] = numeric.isna().sum()
ipy.display(numeric_summary[["min", "50%", "99%", "max", "Missing"]].round(2))

binary = X[list(features.BINARY_COLUMNS)]
binary_summary = pd.concat([
    binary.mean().mul(100).rename("Yes among observed (%)"),
    binary.isna().sum().rename("Missing"),
], axis=1)
ipy.display(binary_summary.round(1))

# %% [markdown]
# Monthly minutes have a median of 367 and a 99th percentile of 2,450. This
# motivates testing a log transformation for nonnegative, right-skewed inputs
# in logistic regression and naive Bayes. Signed billing and usage changes
# keep their original values. The yes/no flags also vary widely in how often
# they occur. About 1.3% report owning a motorcycle, and 90.2% have a web-capable
# handset, so a rare flag can describe relatively few customers.

# %% [markdown]
# ## Linear relationships with churn
#
# We start with Pearson's r to see whether higher input values tend to go
# with higher or lower churn. It measures straight-line association on a scale
# from −1 to +1. We use observed prepared inputs before filling any blanks,
# and show the number of customers behind each result.

# %%
measurements = pd.concat([numeric, binary], axis=1)
churn_pearson = measurements.corrwith(y, method="pearson").rename("Pearson r")
churn_correlations = pd.concat([
    churn_pearson,
    measurements.count().rename("Customers"),
], axis=1)

ipy.display(churn_correlations.sort_values("Pearson r", key=abs, ascending=False).round(3))

# %% [markdown]
# Equipment age has the largest absolute correlation, about 0.106, and tenure
# has r of about 0.023. The linear signals are weak on their own. A local
# spike or an interaction can still carry useful information, so we keep the
# eligible inputs and compare how well the models use them.

# %% [markdown]
# ## Linear relationships between inputs
#
# We look for pairs of measurements that tell a similar story. Each pair
# uses customers with both values present. We show the strongest associations
# by absolute r and use the heatmap to see where they cluster.

# %%
pearson = measurements.corr(method="pearson")
rows = []

for first, second in itertools.combinations(measurements.columns, 2):
    rows.append([first, second, pearson.loc[first, second]])

pairs = pd.DataFrame(rows, columns=["First input", "Second input", "Pearson r"])
strongest = pairs.sort_values("Pearson r", key=abs, ascending=False).head(12).copy()
strongest["Customers"] = [
    measurements[[first, second]].dropna().shape[0]
    for first, second in strongest[["First input", "Second input"]].itertuples(index=False)
]
ipy.display(strongest.reset_index(drop=True).round(3))

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    pearson,
    vmin=-1,
    vmax=1,
    center=0,
    cmap="vlag",
    square=True,
    xticklabels=True,
    yticklabels=True,
    ax=ax,
)
ax.tick_params(labelsize=7)
ax.set_title("Pearson correlations between numeric and yes/no inputs")
fig.tight_layout()
plt.show()

# %% [markdown]
# Mail-order purchases and responses to mail offers have r of about 0.97.
# The counts of handsets and handset models have r of about 0.89. We keep
# these pairs in the model comparisons and account for their overlap when
# interpreting individual importance. Low correlations can hide nonlinear
# relationships and interactions that a tree-based model may use.

# %% [markdown]
# ## Categorical inputs
#
# We count the levels in each category and look at examples of observed churn
# rates. We show the customer count beside each rate because an unusually high
# or low rate is less informative for a small group.

# %%
categories = X[[features.SERVICE_AREA, *features.CATEGORICAL_COLUMNS]]
category_summary = pd.concat([
    categories.nunique().rename("Levels in prepared inputs"),
    categories.eq(preprocessing.MISSING_CATEGORY).sum().rename("Missing"),
], axis=1)
ipy.display(category_summary)

area_counts = X[features.SERVICE_AREA].value_counts()
print(f"Service-area levels with fewer than 20 customers: {(area_counts < 20).sum()}")

for column in [features.SERVICE_AREA, "Homeownership"]:
    category_rates = y.groupby(X[column]).agg(["size", "mean"])
    category_rates.columns = ["Customers", "Churn (%)"]
    category_rates["Churn (%)"] *= 100
    most_common = category_rates.sort_values("Customers", ascending=False).head(8)
    ipy.display(most_common.round(1))

# %% [markdown]
# Service area has 730 levels including the missing label, and 434 levels
# have fewer than 20 customers. That motivates testing compact encodings:
# frequency uses how common each area is, and target encoding estimates its
# churn rate with smoothing so that small groups have less influence. We also
# try one-hot encoding in the model notebooks that can use it for service area.
# Category rates describe one input at a time and can hide interactions.

# %% [markdown]
# ## Why flag months 11–12?
#
# We plot churn by months in service to check for local peaks that the overall
# correlation might hide. We look at the customer counts around months 11 and
# 12 to see how much data supports the pattern.

# %%
tenure = X["MonthsInService"]
tenure_summary = y.groupby(tenure).agg(["size", "sum", "mean"])
tenure_summary.columns = ["Customers", "Churners", "Churn rate"]
near_peak = tenure_summary.loc[8:17].copy()
near_peak["Churn (%)"] = 100 * near_peak.pop("Churn rate")
ipy.display(near_peak.round(1))

months_with_support = tenure_summary[tenure_summary["Customers"] >= 100]
ax = months_with_support["Churn rate"].plot(marker="o", figsize=(9, 4))
ax.axhline(y.mean(), linestyle="--", color="grey", label="Overall churn rate")
ax.axvspan(10.5, 12.5, color="tab:orange", alpha=0.2, label="11–12 month window")
ax.set_title("Observed churn by tenure (months with at least 100 customers)")
ax.set_xlabel("Months in service")
ax.set_ylabel("Churn rate")
ax.yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# At months 11 and 12, observed churn is 45.0% and 39.2%, with more than
# 2,000 customers in each group. This is the local peak behind our fixed tenure
# rule. Contract dates are unavailable, so renewal remains a hypothesis.
# The peak is a concrete example of a pattern a model can use even though one
# linear correlation makes tenure look weak.

# %% [markdown]
# ### Could a simple rule do better?
#
# We compare how many customers each rule flags, how often they churn, and how
# many of all churners it catches. Average precision (AP) summarizes the
# ranking given by each rule's two scores, flagged and unflagged. We use the
# same labelled training data that suggested these rules, so the table is
# exploratory. The fixed rule in the baseline notebook uses months 11–12.

# %%
rules = [
    ("Month 11 only", tenure.eq(11)),
    ("Months 11–12", tenure.between(11, 12)),
    ("Months 11–14", tenure.between(11, 14)),
    ("Month 11 onward", tenure.ge(11)),
    ("Equipment age at least a year", X["CurrentEquipmentDays"].ge(365)),
]

rows = []

for name, flagged in rules:
    flagged = flagged.fillna(False)
    flagged_outcomes = y[flagged]
    rows.append([
        name,
        100 * flagged.mean(),
        100 * flagged_outcomes.mean(),
        100 * flagged_outcomes.sum() / y.sum(),
        sk_metrics.average_precision_score(y, flagged),
    ])

rule_results = pd.DataFrame(
    rows,
    columns=["Rule", "Flagged (%)", "Churn among flagged (%)", "Churners caught (%)", "AP"],
)
ipy.display(rule_results.round(3))

# %% [markdown]
# Month 11 alone gives us the most concentrated list (45.0% churn) and flags
# 7.0% of customers. Months 11–12 flag 12.5% at 42.5% churn. Extending through
# month 14 catches more churners and slightly raises AP, while flagging
# 22.2% of customers at 38.6% churn. We would compare rules at the outreach
# capacity we actually have before changing the baseline.

# %% [markdown]
# ## Why select by average precision?
#
# Churners make up about 28.8% of our training customers. For outreach, we want
# churners near the top of the ranking. Precision tells us what fraction of
# the customers we flag actually churn. Recall tells us what fraction of all
# churners we flag. Average precision (AP) summarizes the precision we achieve
# as we move down the ranking and capture more churners. Giving everyone the
# same score achieves AP of about 0.288, the observed churn rate.
#
# We use mean validation AP to select preprocessing and then tune model
# settings. We set the outreach budget and customer value criteria separately,
# and use precision and recall at that budget to interpret the final ranking.

# %% [markdown]
# ## How this carries into the model notebooks
#
# We compare filling numeric blanks with zero or the training-fold median.
# We fill missing yes/no values with the most frequent observed value and add
# missing indicators for numeric and yes/no fields with blanks during fitting.
# Other categorical inputs use one-hot encoding. We fit all learned
# preprocessing inside the training folds and use the same validation splits
# for every candidate.
#
# | Model | Service-area encoding | Numeric steps |
# | --- | --- | --- |
# | Logistic regression | Target or one-hot | Optional log, then zero or median, then scaling |
# | Gaussian naive Bayes | Target or frequency | Optional log, then zero or median |
# | Decision tree | Target, frequency, or one-hot | Zero or median |
# | Random forest and histogram gradient boosting | Target or frequency | Zero or median |
#
# We compare these options at fixed starting model settings. We record which
# preprocessing goes into each later tuning search in the
# [model comparison notebook](model_comparison.py). We use the reserved test
# set after the final choices are locked.
