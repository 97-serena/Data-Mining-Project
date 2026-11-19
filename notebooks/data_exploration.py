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
# We use the same training customers and prepared inputs as the model notebooks.
# We check the inputs, look for linear relationships, and examine the observed
# churn peak behind our tenure rule. The reserved test set is for final evaluation.

# %%
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import average_precision_score

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.data import load_train
from src.features import BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SERVICE_AREA
from src.preprocessing import MISSING_CATEGORY, prepare_predictors

train = load_train()
X = prepare_predictors(train)
y = train["churn"]

print(f"Training customers: {len(X):,}   Predictors: {X.shape[1]}   Churn: {y.mean():.1%}")
print("Excluded inputs:", ", ".join(train.columns.difference(X.columns, sort=False)))

# %% [markdown]
# ## What changes before modelling?
#
# We keep churn as the target. The model inputs exclude customer IDs,
# both churn labels, and the three retention fields because we cannot place
# retention contact and offer acceptance in time.
#
# Our shared preparation treats nonpositive household ages and negative equipment
# ages as missing. It preserves signed billing and usage changes, converts yes/no
# to 1/0, and gives blank categorical fields a missing label. Recorded categories
# such as “Unknown” and income code 0 remain distinct from blanks.

# %%
fields = ["AgeHH1", "AgeHH2", "CurrentEquipmentDays", SERVICE_AREA]
raw_missing = train[fields].isna().sum().rename("Raw blanks")
prepared_missing = (X[fields].isna() | X[fields].eq(MISSING_CATEGORY)).sum()
display(pd.concat([raw_missing, prepared_missing.rename("Missing after preparation")], axis=1))

numeric = X[list(NUMERIC_COLUMNS)]
numeric_summary = numeric.describe(percentiles=[0.5, 0.99]).T
numeric_summary["Missing"] = numeric.isna().sum()
display(numeric_summary[["min", "50%", "99%", "max", "Missing"]].round(2))

binary = X[list(BINARY_COLUMNS)]
binary_summary = pd.concat([
    binary.mean().mul(100).rename("Yes among observed (%)"),
    binary.isna().sum().rename("Missing"),
], axis=1)
display(binary_summary.round(1))

# %% [markdown]
# ## Linear relationships with churn
#
# Pearson's r measures a straight-line relationship, from −1 to +1. For a
# 0/1 outcome like churn, the sign tells us whether higher input values tend
# to go with higher or lower churn. Each row uses customers with an observed
# input and churn label. We use the prepared values before imputation and
# model-specific transformations.
#
# A small r can hide a curve, a local spike, or an interaction with another
# input. More flexible models may find those patterns. We keep eligible inputs
# for the model comparisons regardless of their individual correlations.

# %%
measurements = pd.concat([numeric, binary], axis=1)
churn_pearson = measurements.corrwith(y, method="pearson").rename("Pearson r")
churn_correlations = pd.concat([
    churn_pearson,
    measurements.count().rename("Customers"),
], axis=1)

display(churn_correlations.sort_values("Pearson r", key=abs, ascending=False).round(3))

# %% [markdown]
# ## Linear relationships between inputs
#
# We check for overlap between measurements using customers with both values
# present. A high absolute r shows strong linear overlap. A low r can miss
# nonlinear relationships and interactions that a tree-based model may use.

# %%
pearson = measurements.corr(method="pearson")
rows = []

for first, second in combinations(measurements.columns, 2):
    rows.append([first, second, pearson.loc[first, second]])

pairs = pd.DataFrame(rows, columns=["First input", "Second input", "Pearson r"])
strongest = pairs.sort_values("Pearson r", key=abs, ascending=False).head(12).copy()
strongest["Customers"] = [
    measurements[[first, second]].dropna().shape[0]
    for first, second in strongest[["First input", "Second input"]].itertuples(index=False)
]
display(strongest.reset_index(drop=True).round(3))

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
# ## Categorical inputs
#
# We compare category counts and observed churn rates. Income codes, handset
# prices and area labels are categories in our pipelines. The count helps us
# judge how much weight to give an unusually high or low rate. Category averages
# can also hide interactions that a model may find.

# %%
categories = X[[SERVICE_AREA, *CATEGORICAL_COLUMNS]]
category_summary = pd.concat([
    categories.nunique().rename("Levels in prepared inputs"),
    categories.eq(MISSING_CATEGORY).sum().rename("Missing"),
], axis=1)
display(category_summary)

for column in [SERVICE_AREA, "Homeownership"]:
    category_rates = y.groupby(X[column]).agg(["size", "mean"])
    category_rates.columns = ["Customers", "Churn (%)"]
    category_rates["Churn (%)"] *= 100
    most_common = category_rates.sort_values("Customers", ascending=False).head(8)
    display(most_common.round(1))

# %% [markdown]
# ## Why flag months 11–12?
#
# We chose those months because they form a clear local peak in observed churn,
# with more than 2,000 customers at each month. The data have no contract dates,
# so contract renewal remains a hypothesis. Tenure's Pearson r with churn is
# only about 0.02. A more flexible model could detect the peak using thresholds
# or account for how the pattern changes with other inputs.

# %%
tenure = X["MonthsInService"]
tenure_summary = y.groupby(tenure).agg(["size", "sum", "mean"])
tenure_summary.columns = ["Customers", "Churners", "Churn rate"]
near_peak = tenure_summary.loc[8:17].copy()
near_peak["Churn (%)"] = 100 * near_peak.pop("Churn rate")
display(near_peak.round(1))

months_with_support = tenure_summary[tenure_summary["Customers"] >= 100]
ax = months_with_support["Churn rate"].plot(marker="o", figsize=(9, 4))
ax.axhline(y.mean(), linestyle="--", color="grey", label="Overall churn rate")
ax.axvspan(10.5, 12.5, color="tab:orange", alpha=0.2, label="11–12 month window")
ax.set_title("Observed churn by tenure (months with at least 100 customers)")
ax.set_xlabel("Months in service")
ax.set_ylabel("Churn rate")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.legend()
plt.tight_layout()
plt.show()

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
        average_precision_score(y, flagged),
    ])

rule_results = pd.DataFrame(
    rows,
    columns=["Rule", "Flagged (%)", "Churn among flagged (%)", "Churners caught (%)", "AP"],
)
display(rule_results.round(3))

# %% [markdown]
# Month 11 alone gives us the most concentrated list (45.0% churn) and flags
# 7.0% of customers. Months 11–12 flag 12.5% at 42.5% churn. Extending through
# month 14 catches more churners and slightly raises AP, while flagging
# 22.2% of customers at 38.6% churn. We would compare rules at the outreach
# capacity we actually have before changing the baseline.

# %% [markdown]
# ## How this carries into the model notebooks
#
# In each training fold we fill numeric blanks with medians and yes/no blanks
# with the most frequent value. We add missing indicators for either type
# when blanks are observed during fitting. Other categorical inputs use one-hot
# encoding (one 0/1 column per label). The options vary by model.
#
# | Model | Service-area encoding | Numeric steps |
# | --- | --- | --- |
# | Logistic regression | Target rate or one-hot | Median imputation and scaling, with an optional log step first |
# | Gaussian naive Bayes | Target rate or frequency | Median imputation, with an optional log step first |
# | Decision tree | Target rate, frequency, or one-hot | Median imputation |
# | Random forest and histogram gradient boosting | Target rate or frequency | Median imputation |
#
# Target encoding estimates churn rates inside the training fold. Frequency
# encoding uses each area's share of training rows. Optional log compression
# applies to skewed nonnegative inputs and preserves the signed billing and
# change fields. The model notebooks fit these steps and score the resulting
# pipelines using the same repeated validation splits.
