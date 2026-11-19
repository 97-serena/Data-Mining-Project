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
# We explore patterns in telecom customer data to understand which information may
# help us predict churn before deciding whom to contact. We use only the training
# customers and reserve the test set for final evaluation.
#
# We examine missing values, associations with churn and relationships among
# individual features. We compare preprocessing and models in the model notebooks.

# %%
# locate the repository from its root or the notebooks folder
from pathlib import Path
import sys

project_root = Path.cwd()
if not (project_root / 'src' / 'data.py').is_file():
    project_root = project_root.parent
if not (project_root / 'src' / 'data.py').is_file():
    raise FileNotFoundError('Open this notebook from the repository root or notebooks folder')

# make the shared project modules available
sys.path.insert(0, str(project_root))

# import the shared data loader
import pandas as pd
from IPython.display import display
from src.data import load_train


# import the shared analysis helpers
from src.eda_utils import (
    PLACEHOLDERS,
    overview,
    churn_signal,
    churn_rate_by,
    plot_column,
    corr_heatmap,
)

# keep the tables readable
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', None)

# %%
# load only the committed training split
train = load_train()

# show the training size and churn proportion
print(train.shape)
print('training churn rate:', round(train['churn'].mean(), 4))

# %% [markdown]
# ## 0. Overview of all features
#
# We check which fields are numeric, where values are missing and which recorded
# categories need care.

# %%
all_cols = train.columns.drop(['CustomerID', 'Churn', 'churn']).tolist()
overview(train, all_cols)

# %% [markdown]
# ### 0.1 Preparing the overview
#
# For these comparisons, we make a separate analysis copy and identify numeric, categorical and yes/no columns. Negative equipment ages are masked in this copy because an age cannot be negative.

# %%
# Import the tools for the feature association checks.
import itertools as y_it
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

# create a separate training copy with a simple row index
y_train = train.copy()
y_train = y_train.reset_index(drop=True)

# Retention fields are examined separately because their timing is unclear.
retention_fields = {'RetentionCalls', 'RetentionOffersAccepted', 'MadeCallToRetentionTeam'}
y_features = [name for name in all_cols if name not in retention_fields]

# identify fields with exactly the two observed yes/no values
y_binary = []
for feature in y_features:
    observed_values = y_train[feature].dropna()
    observed_values = set(observed_values)
    if observed_values == {'Yes', 'No'}:
        y_binary.append(feature)

# identify numeric features while treating income groups as categories
y_numeric = []
for feature in y_features:
    has_numeric_type = pd.api.types.is_numeric_dtype(y_train[feature])
    is_income_category = feature == 'IncomeGroup'
    if has_numeric_type and not is_income_category:
        y_numeric.append(feature)

# treat the remaining fields as categorical for these checks
y_categorical = []
for feature in y_features:
    if feature not in y_numeric:
        y_categorical.append(feature)

# keep the original training values separate from the analysis copy
y_clean = y_train[y_features].copy()

# mark impossible equipment ages as missing in the analysis copy only
negative_equipment_age = y_clean['CurrentEquipmentDays'] < 0
y_clean.loc[negative_equipment_age, 'CurrentEquipmentDays'] = np.nan

# %% [markdown]
# ### 0.2 Comparing numerical and yes/no features
#
# We use Pearson correlation to check for a straight-line relationship and
# Spearman correlation to check whether values tend to rise or fall together.
# We represent yes/no values as 1 and 0 for these plots.

# %%
# select the numeric and yes/no features for the correlation checks
correlation_columns = y_numeric + y_binary
y_corr_data = y_clean[correlation_columns].copy()

# represent yes/no values as numbers in this analysis copy
for feature in y_binary:
    y_corr_data[feature] = y_corr_data[feature].map({'Yes': 1, 'No': 0})

# calculate linear correlations and rank correlations separately
y_pearson = y_corr_data.corr(method='pearson')
y_spearman = y_corr_data.corr(method='spearman')

# collect the two correlation values for every unique pair of features
correlation_pairs = []
for feature_1, feature_2 in y_it.combinations(y_corr_data.columns, 2):
    pearson_value = y_pearson.loc[feature_1, feature_2]
    spearman_value = y_spearman.loc[feature_1, feature_2]

    # count customers with a recorded value for both features
    pair_values = y_corr_data[[feature_1, feature_2]]
    recorded_values = pair_values.notna()
    both_values_present = recorded_values.all(axis=1)
    customer_count = int(both_values_present.sum())

    correlation_pairs.append({
        'feature_1': feature_1,
        'feature_2': feature_2,
        'pearson': pearson_value,
        'spearman': spearman_value,
        'n': customer_count,
    })

# turn the collected pair results into a table
y_pairs = pd.DataFrame(correlation_pairs)

# Rank pairs by correlation strength in either direction.
sorted_pairs = y_pairs.sort_values('spearman', key=abs, ascending=False)
display(sorted_pairs.head(20).round(3))

# %% [markdown]
# Strong relationships can reflect similar measurements or overlapping behavior.
# For example, customers who make more calls also have more opportunities for a
# call to go unanswered.

# %%
# draw a separate heatmap for each correlation measure
correlation_matrices = [
    ('Pearson', y_pearson),
    ('Spearman', y_spearman),
]
for y_name, y_matrix in correlation_matrices:
    plt.figure(figsize=(17, 15))
    sns.heatmap(
        y_matrix,
        vmin=-1,
        vmax=1,
        center=0,
        cmap='vlag',
        square=True,
        cbar_kws={'label': 'Correlation'},
    )

    # label and display the correlation heatmap
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(f'{y_name}: numerical and yes/no features')
    plt.tight_layout()
    plt.show()


# %% [markdown]
# We see several relationships between call usage and service quality. Strong correlation can reveal overlap, though it cannot tell us how much each feature helps a model predict churn.

# %% [markdown]
# ### 0.3 Comparing categorical features
#
# Some features contain categories, such as IncomeGroup or whether a customer has a credit card. Giving categories arbitrary numbers would make the results hard to interpret, so we use Cramér’s V to measure how strongly two categorical features are related. It ranges from 0 to 1; it does not tell us which category is higher. We use a correction for small-sample bias and show the number of customers for each pair. Pairs of yes/no features are already shown in the correlation graphs, so they are left blank here.

# %%
def y_cramers(feature_a, feature_b):
    # count how often each combination of categories occurs
    counts = pd.crosstab(feature_a, feature_b)
    count_values = counts.to_numpy()
    sample_size = count_values.sum()
    rows, columns = counts.shape

    # require at least two categories in both features
    has_enough_customers = sample_size > 1
    has_multiple_rows = rows > 1
    has_multiple_columns = columns > 1
    if not has_enough_customers or not has_multiple_rows or not has_multiple_columns:
        return np.nan

    # measure how different the counts are from independence
    chi_square_result = chi2_contingency(counts, correction=False)
    chi_square = chi_square_result[0]
    phi_squared = chi_square / sample_size

    # correct the measure for bias caused by the sample size
    bias = (rows - 1) * (columns - 1) / (sample_size - 1)
    corrected_phi_squared = max(0, phi_squared - bias)

    # correct the number of categories for the same bias
    corrected_rows = rows - 1 - (rows - 1) ** 2 / (sample_size - 1)
    corrected_columns = columns - 1 - (columns - 1) ** 2 / (sample_size - 1)
    denominator = min(corrected_rows, corrected_columns)

    # return an undefined value when the corrected denominator is not positive
    if denominator <= 0:
        return np.nan

    # calculate the corrected association between the categorical features
    association = np.sqrt(corrected_phi_squared / denominator)
    return association


# calculate the association for every unique categorical pair
pair_results = []
for feature_a, feature_b in y_it.combinations(y_categorical, 2):
    association = y_cramers(y_clean[feature_a], y_clean[feature_b])

    # count customers with values recorded for both features
    pair_values = y_clean[[feature_a, feature_b]]
    recorded_values = pair_values.notna()
    both_present = recorded_values.all(axis=1)
    customer_count = int(both_present.sum())

    pair_results.append({
        'feature_1': feature_a,
        'feature_2': feature_b,
        'V': association,
        'n': customer_count,
    })

# collect the categorical results in a table
y_cat_pairs = pd.DataFrame(pair_results)

# identify pairs where both features are yes/no fields
first_is_binary = y_cat_pairs['feature_1'].isin(y_binary)
second_is_binary = y_cat_pairs['feature_2'].isin(y_binary)
y_cat_pairs['both_binary'] = first_is_binary & second_is_binary

# start the association matrix with one on its diagonal
y_v = pd.DataFrame(
    np.eye(len(y_categorical)),
    index=y_categorical,
    columns=y_categorical,
)

# place each result in both halves of the symmetric matrix
for pair in y_cat_pairs.itertuples():
    y_v.loc[pair.feature_1, pair.feature_2] = pair.V
    y_v.loc[pair.feature_2, pair.feature_1] = pair.V

# Omit yes/no pairs because the correlation plots already show them.
y_cat_mask = np.outer(y_v.index.isin(y_binary), y_v.columns.isin(y_binary))

plt.figure(figsize=(12, 10))
sns.heatmap(
    y_v,
    vmin=0,
    vmax=1,
    cmap='Blues',
    square=True,
    mask=y_cat_mask,
    cbar_kws={'label': "Bias-corrected Cramér's V"},
)
plt.title('Categorical feature associations')
plt.tight_layout()
plt.show()

eligible_pairs = y_cat_pairs.loc[~y_cat_pairs['both_binary']]

# show the fifteen strongest categorical associations
sorted_pairs = eligible_pairs.sort_values('V', ascending=False)
strongest_pairs = sorted_pairs.head(15)
display(strongest_pairs.round(3))


# %% [markdown]
# We see strong associations between credit-card ownership and both income code
# and homeownership. We check whether recorded unknown values contribute to
# those patterns below. Rare categories also require care, especially ServiceArea.

# %% [markdown]
# ### 0.4 Comparing numerical features with categories
#
# We also want to see whether numerical values differ among categories. For example, the number of handset models may differ between recorded handset-price categories. The measure below, adjusted η², describes how much of the numerical variation is associated with the categories. It runs from 0 to 1; it does not show a direction or prove that one feature causes the other. Yes/no categories are covered in the earlier correlation graphs.

# %%
def y_eta(numerical_feature, categorical_feature):
    # keep customers with recorded values for both features
    data = pd.DataFrame({
        'value': numerical_feature,
        'category': categorical_feature,
    })
    data = data.dropna()

    # count customers and calculate the mean in each category
    groups = data.groupby('category', observed=True)
    category_values = groups['value']
    group_summary = category_values.agg(['size', 'mean'])
    sample_size = len(data)
    category_count = len(group_summary)

    # measure the total variation around the overall mean
    overall_mean = data['value'].mean()
    differences_from_mean = data['value'] - overall_mean
    squared_differences = differences_from_mean ** 2
    total_variation = squared_differences.sum()

    # measure the variation between category means
    group_differences = group_summary['mean'] - overall_mean
    weighted_group_differences = group_summary['size'] * group_differences ** 2
    between_group_variation = weighted_group_differences.sum()

    # calculate the share of variation explained by the categories
    if total_variation > 0:
        eta_squared = between_group_variation / total_variation
    else:
        eta_squared = np.nan

    # adjust for the sample size and number of categories
    if sample_size > category_count:
        unexplained_variation = 1 - eta_squared
        adjusted_unexplained_variation = unexplained_variation * (sample_size - 1)
        adjusted_unexplained_variation = adjusted_unexplained_variation / (sample_size - category_count)
        adjusted_eta_squared = max(0, 1 - adjusted_unexplained_variation)
    else:
        adjusted_eta_squared = np.nan

    return eta_squared, adjusted_eta_squared, sample_size


# calculate the association for each numeric and categorical combination
mixed_results = []
for numerical_feature in y_numeric:
    for categorical_feature in y_categorical:
        eta_squared, adjusted_eta_squared, customer_count = y_eta(
            y_clean[numerical_feature],
            y_clean[categorical_feature],
        )
        mixed_results.append({
            'numeric': numerical_feature,
            'categorical': categorical_feature,
            'eta2': eta_squared,
            'adjusted_eta2': adjusted_eta_squared,
            'n': customer_count,
        })

# collect the mixed association results in a table
y_mixed = pd.DataFrame(mixed_results)

is_binary_category = y_mixed['categorical'].isin(y_binary)
y_mixed_main = y_mixed.loc[~is_binary_category]

# arrange the adjusted association values as a heatmap matrix
heatmap_values = y_mixed_main.pivot(
    index='numeric',
    columns='categorical',
    values='adjusted_eta2',
)

# show the relationships between numeric and categorical features
plt.figure(figsize=(14, 12))
sns.heatmap(
    heatmap_values,
    vmin=0,
    vmax=1,
    cmap='YlGnBu',
    cbar_kws={'label': 'Adjusted η²'},
)
plt.title('Numerical and categorical feature associations')
plt.tight_layout()
plt.show()

# show the fifteen strongest eligible associations
sorted_pairs = y_mixed_main.sort_values('adjusted_eta2', ascending=False)
strongest_pairs = sorted_pairs.head(15)
display(strongest_pairs.round(3))

# count the unique numeric pairs covered by the correlation checks
numeric_count = len(y_numeric)
numeric_pairs = numeric_count * (numeric_count - 1) // 2

# summarize coverage across the three association checks
y_coverage = pd.Series({
    'numeric_numeric_pairs': numeric_pairs,
    'categorical_categorical_pairs': len(y_cat_pairs),
    'mixed_pairs': len(y_mixed),
})
coverage_table = y_coverage.to_frame('unique_pairs')
display(coverage_table)

# verify that every unique pair of examined features has been checked
feature_count = len(y_features)
expected_pairs = feature_count * (feature_count - 1) // 2
assert y_coverage.sum() == expected_pairs

# %% [markdown]
# We observe differences in handset counts among recorded price categories.
# The pair counts verify that we checked all combinations of numerical and
# categorical features in this screen.

# %% [markdown]
# ### 0.5 Checking whether unknown values drive these relationships
#
# The previous graphs include values such as zero for age and “Unknown” for handset price. We check selected relationships again without those values to see whether the apparent overlap might partly come from missing or undocumented information. This is a comparison between different groups of customers, so it should be treated as a sensitivity check.

# %%
# list the recorded codes used in this sensitivity check
y_placeholders = {
    'AgeHH1': 0,
    'AgeHH2': 0,
    'IncomeGroup': 0,
    'HandsetPrice': 'Unknown',
    'MaritalStatus': 'Unknown',
    'Homeownership': 'Unknown',
}

# choose feature pairs and the association measure for each pair
y_sensitivity_specs = [
    ('AgeHH1', 'IncomeGroup', 'eta'),
    ('IncomeGroup', 'HasCreditCard', 'V'),
    ('IncomeGroup', 'Homeownership', 'V'),
    ('HandsetModels', 'HandsetPrice', 'eta'),
    ('AgeHH1', 'MaritalStatus', 'eta'),
]

# compare each pair before and after excluding its recorded codes
sensitivity_results = []
for feature_a, feature_b, measure in y_sensitivity_specs:
    # start with customers who have both values recorded
    pair_values = y_clean[[feature_a, feature_b]]
    recorded_values = pair_values.notna()
    valid_customers = recorded_values.all(axis=1)

    # remove either feature's recorded code from this diagnostic subset
    pair_features = [feature_a, feature_b]
    for feature in pair_features:
        if feature in y_placeholders:
            placeholder = y_placeholders[feature]
            does_not_use_code = y_clean[feature] != placeholder
            valid_customers = valid_customers & does_not_use_code

    # select the paired values with their recorded codes removed
    filtered_feature_a = y_clean.loc[valid_customers, feature_a]
    filtered_feature_b = y_clean.loc[valid_customers, feature_b]

    # use the categorical measure when both fields are categorical
    if measure == 'V':
        association_before = y_cramers(y_clean[feature_a], y_clean[feature_b])
        association_after = y_cramers(filtered_feature_a, filtered_feature_b)
    else:
        # use adjusted explained variation for a numeric and categorical pair
        original_result = y_eta(y_clean[feature_a], y_clean[feature_b])
        filtered_result = y_eta(filtered_feature_a, filtered_feature_b)
        association_before = original_result[1]
        association_after = filtered_result[1]

    # count the customers and categories remaining in the subset
    remaining_customers = int(valid_customers.sum())
    remaining_categories = filtered_feature_b.nunique()

    # keep the original and sensitivity results side by side
    sensitivity_results.append({
        'feature_1': feature_a,
        'feature_2': feature_b,
        'measure': measure,
        'all_nonmissing': association_before,
        'codes_excluded': association_after,
        'n_codes_excluded': remaining_customers,
        'remaining_levels_feature_2': remaining_categories,
    })

# show the association changes with rounded display values
y_code_sensitivity = pd.DataFrame(sensitivity_results)
display(y_code_sensitivity.round(3))

# %% [markdown]
# The relationship between IncomeGroup and HasCreditCard falls from V = 0.827 to 0.162 when income code 0 is excluded. The relationship between HandsetModels and HandsetPrice falls from 0.514 to 0.051 when “Unknown” prices are excluded. AgeHH1 and IncomeGroup also fall from 0.564 to 0.018 after excluding their zero codes. This suggests that the codes contribute substantially to the original patterns. We should preserve information about unknown values, while avoiding the conclusion that the raw relationships represent only customer behaviour or income.

# %% [markdown]
# ### 0.6 Checking two conclusions before deciding what to keep
#
# We check whether the two cellphone-user flags and the overlapping call counts carry identical information.

# %%
# compare the combined quality count with its two component counts
dropped_calls = y_train['DroppedCalls']
blocked_calls = y_train['BlockedCalls']
combined_calls = y_train['DroppedBlockedCalls']
y_residual = combined_calls - dropped_calls - blocked_calls

# measure the size of each difference without regard to its sign
absolute_difference = y_residual.abs()

# count exact matches and matches within the stated tolerance
exact_match = absolute_difference < 1e-9
within_tolerance = absolute_difference < 0.11
large_difference = absolute_difference > 1

# summarize how often the counts agree and how far apart they can be
comparison_results = {
    'exact_within_1e-9_%': exact_match.mean() * 100,
    'within_0.11_%': within_tolerance.mean() * 100,
    'differences_above_1': large_difference.sum(),
    'largest_difference': absolute_difference.max(),
}

# show the comparison summary as a rounded table
comparison_table = pd.Series(comparison_results)
comparison_table = comparison_table.to_frame('value')
display(comparison_table.round(2))

# %% [markdown]
# We observe 27,363 customers with No for both user-status flags. The combined
# call count is within 0.11 of the sum of dropped and blocked calls for 96.72%
# of customers, with 482 differences above 1.

# %% [markdown]
# ## 1. Account & billing
#
# We look at spending, plan charges, tenure, equipment age and the number of
# lines. For an individual numerical feature, ROC AUC (area under the receiver
# operating characteristic curve) measures how often a churner has a higher
# value than a non-churner, giving half credit for a tie. A score near 0.5 means
# the two are about equally likely to have the higher value.

# %%
# select the account and billing features
cols = [
    'MonthlyRevenue',
    'TotalRecurringCharge',
    'OverageMinutes',
    'MonthsInService',
    'CurrentEquipmentDays',
    'Handsets',
    'HandsetModels',
    'UniqueSubs',
    'ActiveSubs',
    'AdjustmentsToCreditRating',
]

# inspect data quality and each feature's individual churn association
feature_overview = overview(train, cols)
display(feature_overview)
feature_churn_signal = churn_signal(train, cols)
display(feature_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# Compare the numerical features in this section.
corr = corr_heatmap(train, cols)

# %% [markdown]
# ### EDA Findings — Account & billing
#
# The billing variables have a small amount of missing data: `MonthlyRevenue`, `TotalRecurringCharge` and `OverageMinutes` each have 120 missing values (0.29%). `Handsets`, `HandsetModels` and `CurrentEquipmentDays` each have one missing value; the remaining four columns have none. Revenue is negative for 3 customers (minimum -6.17), recurring charge for 8 (minimum -11), and equipment age for 55 (minimum -5). Negative equipment age is invalid, but negative billing values may represent credits or adjustments, so we retain those signed amounts in this workflow. `HandsetModels` never exceeds `Handsets`, and `ActiveSubs` never exceeds `UniqueSubs`.
#
# Revenue and overage are right-skewed: median `MonthlyRevenue` is 48.47, its 99th percentile is 228.33 and its maximum is 1,223.38. `OverageMinutes` is zero for 45.59% of customers, with a median of 3 and maximum of 4,321. `MonthsInService` ranges from 6 to 61 months (median 16), so the data cannot describe the first months of service. Handset and subscription counts generally have a median of 1, although they reach 24 handsets and 18 subscriptions; `AdjustmentsToCreditRating` is zero for 96.41%. Several measures overlap: Spearman correlations are 0.91 for handsets and handset models, 0.83 for unique and active subscriptions, and 0.76 for revenue and recurring charge. Tenure and equipment age are less closely related (0.42).
#
# Tenure has a clear descriptive relationship with churn: 16.8% at 6–10 months, 42.5% at 11–12 months, 30.7% at 13–24 months and 28.8% after that. Contract expiry is a possible explanation, but the data does not establish it. Equipment age is the strongest individual association among these fields (AUC 0.576): churn is 19.3% at 180–300 days, 39.3% at 300–365 days and 32–37% for older handsets. A desire for a new phone is a possible interpretation, not a demonstrated cause.
#
# Recurring charges also vary with churn: 33.7% in the lowest fifth versus 24.2% in the highest (AUC 0.457). Monthly revenue alone is nearly unrelated (AUC 0.485), except for 33.3% churn in its lowest tenth; overage has a weak association (AUC 0.511), rising from about 28% to about 32% in the highest group. Churn is 30.0% with one handset and 25.5% with four or more. The difference between unique and active subscriptions exceeds zero for 13.1% of customers; churn is 27.9% with no inactive line, 33.6% with one and 38.0% with two or more.
# %% [markdown]
# ## 2. Service usage
#
# Billing tells us what customers pay; usage tells us how much they engage with the service.
# Here we look at overall activity, calling patterns and use of additional features. Sparse services
# are worth checking carefully, since a small group of users can produce an unstable churn rate.

# %%
# select the service usage features
cols = [
    'MonthlyMinutes',
    'DirectorAssistedCalls',
    'RoamingCalls',
    'ThreewayCalls',
    'ReceivedCalls',
    'OutboundCalls',
    'InboundCalls',
    'PeakCallsInOut',
    'OffPeakCallsInOut',
    'CallForwardingCalls',
    'CallWaitingCalls',
]

# inspect data quality and individual churn associations
feature_overview = overview(train, cols)
display(feature_overview)
feature_churn_signal = churn_signal(train, cols)
display(feature_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# Compare the numerical features in this section.
corr = corr_heatmap(train, cols)

# keep the tools used for the additional checks available
import pandas as pd
import numpy as np

# keep customers with observed monthly minutes and churn
usage_col = "MonthlyMinutes"
usage_columns = [usage_col, "churn"]
valid = train[usage_columns].dropna().copy()

# divide customers into five groups by monthly usage
valid["usage_group"] = pd.qcut(
    valid[usage_col],
    q=5,
    duplicates="drop",
)

# calculate the size and churn rate of each usage group
usage_groups = valid.groupby("usage_group", observed=True)
usage_summary = usage_groups.agg(
    customers=("churn", "size"),
    churners=("churn", "sum"),
    churn_rate=("churn", "mean"),
)

# express churn rates as percentages for the table
usage_summary["churn_rate"] = usage_summary["churn_rate"] * 100
rounded_usage_summary = usage_summary.round(2)
display(rounded_usage_summary)

# list the services whose use is uncommon
sparse_features = [
    "RoamingCalls",
    "ThreewayCalls",
    "CallForwardingCalls",
    "CallWaitingCalls",
]

# compare customers who used each service with customers who did not
results = []
for feature in sparse_features:
    service_columns = [feature, "churn"]
    data = train[service_columns].dropna().copy()

    # mark whether any use of this service was recorded
    data["used"] = data[feature] > 0

    # count customers and calculate churn for the two states
    service_groups = data.groupby("used")["churn"]
    summary = service_groups.agg(["size", "mean"])

    # record one readable summary row for each state
    for used, row in summary.iterrows():
        customer_count = int(row["size"])
        churn_percentage = round(row["mean"] * 100, 2)
        results.append({
            "feature": feature,
            "used": used,
            "customers": customer_count,
            "churn_rate_%": churn_percentage,
        })

service_usage_summary = pd.DataFrame(results)
display(service_usage_summary)

# reuse the same churn summary for the following account checks
def churn_table(key):
    # group customers and calculate the count and churn proportion
    grouped_churn = train.groupby(key, observed=True)["churn"]
    table = grouped_churn.agg(["size", "mean"])
    return table.round(3)

# count negative values in the billing and equipment fields
negative_value_columns = [
    "MonthlyRevenue",
    "TotalRecurringCharge",
    "CurrentEquipmentDays",
]
negative_counts = {}
for feature in negative_value_columns:
    negative_values = train[feature] < 0
    negative_counts[feature] = int(negative_values.sum())
print("Negative values:", negative_counts)

# check whether the related account counts are logically consistent
models_exceed_handsets = train["HandsetModels"] > train["Handsets"]
active_exceeds_unique = train["ActiveSubs"] > train["UniqueSubs"]
print(
    "HandsetModels > Handsets:", int(models_exceed_handsets.sum()),
    "| ActiveSubs > UniqueSubs:", int(active_exceeds_unique.sum()),
)

# measure the association between customer tenure and equipment age
tenure_equipment_correlation = train["MonthsInService"].corr(
    train["CurrentEquipmentDays"],
    method="spearman",
)
print("Tenure vs handset age (Spearman):", round(tenure_equipment_correlation, 2))

# summarize churn across the selected tenure bands
tenure_bands_for_summary = pd.cut(
    train["MonthsInService"],
    [0, 10, 12, 24, 1000],
    labels=["6-10m", "11-12m", "13-24m", ">24m"],
)
display(churn_table(tenure_bands_for_summary))

# summarize churn across the selected equipment age bands
equipment_bands_for_summary = pd.cut(
    train["CurrentEquipmentDays"],
    [-10, 180, 300, 365, 450, 730, 2000],
)
display(churn_table(equipment_bands_for_summary))

# compare recurring charge groups with similar numbers of customers
recurring_charge_groups = pd.qcut(train["TotalRecurringCharge"], 5)
display(churn_table(recurring_charge_groups))

# combine larger handset counts for this descriptive table only
handset_count_groups = train["Handsets"].clip(upper=4)
handset_count_groups = handset_count_groups.rename("Handsets (4 = 4 or more)")
display(churn_table(handset_count_groups))

# calculate the number of inactive subscriptions for each customer
inactive_subs = train["UniqueSubs"] - train["ActiveSubs"]
has_inactive_line = inactive_subs > 0
inactive_line_share = round(has_inactive_line.mean(), 3)
print("Customers with an inactive line:", inactive_line_share)

# combine counts of two or more inactive lines for the descriptive table
inactive_subscription_groups = inactive_subs.clip(upper=2)
inactive_subscription_groups = inactive_subscription_groups.rename(
    "inactive_subs (2 = 2 or more)"
)
display(churn_table(inactive_subscription_groups))

# %% [markdown]
# ### EDA Findings — Service Usage
#
# `MonthlyMinutes`, `DirectorAssistedCalls` and `RoamingCalls` each have 120 missing training values (0.29%); the other eight usage features are complete. Missing values should remain distinct from zero use. Usage is often sparse: 68.66% have no roaming calls, 72.72% no three-way calls and 99.55% no call-forwarding calls. Only 183 customers (0.45%) used forwarding, while 49.04% have zero call-waiting calls. Monthly usage is right-skewed: the highest quintile spans 841–7,359 minutes, which shows how widely usage varies.
#
# Churn is 34.28% in the lowest monthly-usage quintile (0–122 minutes) and 25.79% in the highest (above 841 minutes), a difference of 8.49 percentage points. Three-way users have 26.93% churn versus 29.53% for non-users, and call-waiting users have 27.44% versus 30.25%. Roaming users and non-users are similar at 28.71% and 28.79%. Forwarding users show 24.59% versus 28.84%, but that comparison is uncertain because only 183 customers used it. These are descriptive associations, not evidence of causation.
# %% [markdown]
# ## 3. Usage change & retention
#
# A customer's current activity is only part of the picture. A change in that activity might reveal
# disengagement that the current level alone misses. We examine what the change
# columns measure and consider the timing of retention contact separately.

# %%
# select the usage change and retention features
cols = [
    'PercChangeMinutes',
    'PercChangeRevenues',
    'RetentionCalls',
    'RetentionOffersAccepted',
    'MadeCallToRetentionTeam',
    'ReferralsMadeBySubscriber',
    'NewCellphoneUser',
    'NotNewCellphoneUser',
]

# inspect data quality and individual churn associations
feature_overview = overview(train, cols)
display(feature_overview)
feature_churn_signal = churn_signal(train, cols)
display(feature_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# Compare the numerical features in this section.
corr = corr_heatmap(train, cols)

# %% [markdown]
# ### 3.1 Are the "PercChange" columns really percentages?
#
# For a nonnegative quantity and a positive baseline, an ordinary percentage decrease cannot be below -100%. Values below that boundary make the documented units uncertain; the check does not establish the exact original formula. This checks how often that happens and whether the change scales with usage. The helper functions defined here are reused in section 7.

# %%
# import the tools used for the usage-change checks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency

# collect chi-square results for the later multiple-testing correction
p_values = {}

def auc_of(s, target=train["churn"]):
    """calculate univariate ROC-AUC using observed finite values"""
    # align the feature with the target and represent it numerically
    values = pd.Series(s, index=target.index).astype(float)

    # retain customers with an observed and finite feature value
    observed_values = values.notna()
    finite_values = np.isfinite(values)
    valid_values = observed_values & finite_values

    # calculate the ranking score for those same customers
    valid_target = target[valid_values]
    valid_feature = values[valid_values]
    score = roc_auc_score(valid_target, valid_feature)
    return round(score, 3)

def chi2_p(flag, name=None, target=train["churn"]):
    """calculate a chi-square p-value and optionally store it by name"""
    # count the combinations of feature states and churn outcomes
    counts = pd.crosstab(flag, target)

    # extract the p-value from the chi-square test result
    test_result = chi2_contingency(counts)
    p = test_result[1]

    # retain named results for the multiple-testing correction
    if name:
        p_values[name] = p
    return p

# keep the source change and usage columns available for later checks
pcm = train["PercChangeMinutes"]
pcr = train["PercChangeRevenues"]
mins = train["MonthlyMinutes"]

# leave normalized change undefined when monthly minutes are zero
nonzero_minutes = mins.replace(0, np.nan)
rel_change_min = pcm / nonzero_minutes

# count values that do not look like ordinary percentage decreases
minutes_below_minus_100 = pcm < -100
revenues_below_minus_100 = pcr < -100
print(
    f"PercChangeMinutes below -100: {minutes_below_minus_100.sum():,} "
    f"({minutes_below_minus_100.mean():.1%})"
)
print(f"PercChangeRevenues below -100: {revenues_below_minus_100.sum():,}")

# check whether the size of the change follows the customer's usage level
absolute_minutes_change = pcm.abs()
change_usage_correlation = absolute_minutes_change.corr(mins, method="spearman")
print("Spearman |PercChangeMinutes| vs MonthlyMinutes:", round(change_usage_correlation, 2))

# inspect recorded change values when monthly usage is zero
zero_minutes = mins == 0
change_for_zero_minutes = pcm[zero_minutes]
zero_minutes_summary = change_for_zero_minutes.agg(["count", "min", "max"])
print("PercChangeMinutes when MonthlyMinutes = 0:", zero_minutes_summary.to_dict())
print("Lowest PercChangeMinutes / MonthlyMinutes:", rel_change_min.min())

# divide normalized minutes change into bands
rel_bins = pd.cut(
    rel_change_min,
    [-1.01, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, np.inf],
    labels=[
        "fell >50%", "fell 20-50%", "fell 5-20%", "about 0",
        "rose 5-20%", "rose 20-50%", "rose >50%",
    ],
)

# calculate customer counts and churn rates within those bands
minutes_change_groups = train.groupby(rel_bins, observed=True)["churn"]
rel_rate = minutes_change_groups.agg(n="size", churn_rate="mean")
rel_rate = rel_rate.round(3)
print("Minutes change (relative to MonthlyMinutes):")
display(rel_rate.T)

# normalize revenue change only when monthly revenue is positive
positive_monthly_revenue = train["MonthlyRevenue"].where(train["MonthlyRevenue"] > 0)
rel_change_rev = pcr / positive_monthly_revenue
rev_bins = pd.cut(
    rel_change_rev,
    [-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf],
    labels=["ratio < -0.2", "-0.2 to -0.05", "near 0", "0.05 to 0.2", "ratio > 0.2"],
)

# summarize the revenue change bands using the same churn measures
revenue_change_groups = train.groupby(rev_bins, observed=True)["churn"]
revenue_change_summary = revenue_change_groups.agg(n="size", churn_rate="mean")
revenue_change_summary = revenue_change_summary.round(3)
print("Revenue change (relative to MonthlyRevenue):")
display(revenue_change_summary.T)

# compare the raw change columns with their normalized versions
raw_change_auc = [auc_of(pcm), auc_of(pcr)]
relative_change_auc = [auc_of(rel_change_min), auc_of(rel_change_rev)]
change_auc_summary = pd.DataFrame({
    "raw column": raw_change_auc,
    "relative to monthly level": relative_change_auc,
}, index=["minutes change", "revenue change"])
change_auc_summary = change_auc_summary.rename_axis("AUC")
display(change_auc_summary)

# use a reproducible sample to show change against the monthly usage level
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
sample = train.sample(4000, random_state=42)
axes[0].scatter(sample["MonthlyMinutes"], sample["PercChangeMinutes"], s=4, alpha=0.3)

# draw the observed lower-bound reference line
x_max = train["MonthlyMinutes"].quantile(0.99)
axes[0].plot(
    [0, x_max],
    [0, -x_max],
    c="red",
    lw=1,
    label="PercChangeMinutes = -MonthlyMinutes",
)
axes[0].set_xlim(0, x_max)
lower_change_limit = train["PercChangeMinutes"].quantile(0.005)
upper_change_limit = train["PercChangeMinutes"].quantile(0.995)
axes[0].set_ylim(lower_change_limit, upper_change_limit)
axes[0].set_xlabel("MonthlyMinutes")
axes[0].set_ylabel("PercChangeMinutes")
axes[0].set_title("No value falls below the red line")
axes[0].legend(fontsize=7)

# show the churn rate for each normalized minutes-change band
axes[1].bar(range(len(rel_rate)), rel_rate["churn_rate"])
axes[1].set_xticks(
    range(len(rel_rate)),
    rel_rate.index,
    rotation=45,
    ha="right",
    fontsize=8,
)
axes[1].axhline(train["churn"].mean(), ls="--", c="grey")
axes[1].set_title("Churn rate by relative change in minutes")
plt.tight_layout()
plt.show()

# %% [markdown]
# The change columns need more care than their names suggest. Values below −100 make an ordinary percentage interpretation doubtful, while their observed boundary is consistent with a signed change in minutes. We retain both raw signed columns and include `RelChangeMinutes` as a constructed measure. A zero or missing `MonthlyMinutes` denominator leaves this ratio undefined; we do not turn it into zero. Its original units remain unverified.

# %% [markdown]
# ### 3.2 Missing change values
#
# Are the missing values the same customers in both columns, and do these customers churn differently?

# %%
# mark customers with missing minutes-change and revenue observations
chg_missing = train["PercChangeMinutes"].isna()
revenue_change_missing = train["PercChangeRevenues"].isna()
rev_missing = train["MonthlyRevenue"].isna()

# count how often both change fields are missing together
both_changes_missing = chg_missing & revenue_change_missing
print(
    "Missing PercChangeMinutes:", chg_missing.sum(),
    "| missing in both change columns:", both_changes_missing.sum(),
)

# check how missing billing observations overlap with missing change values
billing_and_change_missing = rev_missing & chg_missing
print(
    "Missing MonthlyRevenue:", rev_missing.sum(),
    "| of these also missing the change values:", billing_and_change_missing.sum(),
)

# compare churn for customers with observed and missing change values
change_observation_labels = chg_missing.map({False: "change observed", True: "change missing"})
change_observation_groups = train.groupby(change_observation_labels)["churn"]
change_observation_summary = change_observation_groups.agg(n="size", churn_rate="mean")
change_observation_summary = change_observation_summary.round(3)
display(change_observation_summary)

# retain the association test for the later multiple-testing correction
change_missing_p_value = chi2_p(chg_missing, "3.2 change values missing")
print(f"chi-square p-value: {change_missing_p_value:.1e}")

# %% [markdown]
# The same 290 customers have missing values in both change columns, and their observed churn rate is 56.9% compared with 28.6% when observed. We preserve each numeric field's missingness before imputation. `ChangeMissing` is the indicator for `PercChangeMinutes`, not an additional duplicate indicator.

# %% [markdown]
# ### 3.3 Retention variables: redundancy, churn and timing
#
# The data has no dates, so leakage cannot be proven directly. These checks look at redundancy, at how churn differs by type of retention contact, and at when in the customer life the contacts happen.

# %%
# keep the three retention fields available for these descriptive checks
rc = train["RetentionCalls"]
ro = train["RetentionOffersAccepted"]
mc = train["MadeCallToRetentionTeam"]

# compare the yes/no flag with the recorded call count
has_retention_call = rc > 0
call_flag_is_yes = mc == "Yes"
flag_matches_count = call_flag_is_yes == has_retention_call
print(
    "MadeCallToRetentionTeam = Yes exactly when RetentionCalls > 0:",
    bool(flag_matches_count.all()),
)

# check whether an offer was accepted without a recorded retention call
has_accepted_offer = ro > 0
has_no_retention_call = rc == 0
offer_without_call = has_accepted_offer & has_no_retention_call
print("Offers accepted without any retention call:", int(offer_without_call.sum()))
print(
    f"Customers with a retention call: {has_retention_call.mean():.1%} "
    f"| with an accepted offer: {has_accepted_offer.mean():.1%}"
)

# give accepted offers priority when assigning a retention-contact state
contact_conditions = [has_accepted_offer, has_retention_call]
contact_labels = ["accepted an offer", "called, no offer accepted"]
contact_values = np.select(contact_conditions, contact_labels, "no retention contact")
retention_state = pd.Series(contact_values, index=train.index)

# summarize customers and churn for each contact state
retention_groups = train.groupby(retention_state)["churn"]
retention_summary = retention_groups.agg(n="size", churn_rate="mean")
retention_summary = retention_summary.round(3)
display(retention_summary)

# compare any retention contact with no contact
contacted = retention_state != "no retention contact"
contact_p_value = chi2_p(contacted, "3.3 any retention contact")
print(f"p-value, any retention contact vs none: {contact_p_value:.1e}")

# compare offer acceptance only among customers with recorded contact
contacted_states = retention_state[contacted]
accepted_among_contacted = contacted_states == "accepted an offer"
acceptance_p_value = chi2_p(accepted_among_contacted, "3.3 accepted offer vs called only")
print("p-value, accepted an offer vs called without accepting:", round(acceptance_p_value, 3))

# group tenure using the same bands as the other descriptive checks
tenure_band = pd.cut(
    train["MonthsInService"],
    [0, 10, 12, 24, 1000],
    labels=["6-10m", "11-12m", "13-24m", ">24m"],
)
tenure_band = tenure_band.rename("tenure")

# calculate the share with a retention call within each tenure band
retention_call_groups = has_retention_call.groupby(tenure_band, observed=True)
retention_call_share = retention_call_groups.mean()
retention_call_share = retention_call_share.round(3)
retention_call_share = retention_call_share.to_frame("retention_call_share")
print("Share of customers with a retention call, by tenure:")
display(retention_call_share.T)

# %% [markdown]
# We observe that the call flag matches whether `RetentionCalls` is above zero.
# We leave retention fields out of the models because we cannot verify their timing
# relative to our outreach decision.

# %% [markdown]
# ### 3.4 New / not-new cellphone user
#
# The two flags are never both "Yes". This checks what the remaining combinations mean for churn.

# %%
# begin with the state where neither user flag is indicated
new_status = pd.Series("neither indicated", index=train.index)

# preserve the two explicitly indicated states
new_user_is_yes = train["NewCellphoneUser"] == "Yes"
not_new_user_is_yes = train["NotNewCellphoneUser"] == "Yes"
new_status[new_user_is_yes] = "new"
new_status[not_new_user_is_yes] = "not new"

# summarize the customer count and churn rate for each state
status_groups = train.groupby(new_status)["churn"]
status_table = status_groups.agg(n="size", churn_rate="mean")
status_table = status_table.round(3)

# add the median tenure to describe who appears in each state
status_tenure_groups = train.groupby(new_status)["MonthsInService"]
status_table["median_tenure_months"] = status_tenure_groups.median()
display(status_table)

# store the association test for the later multiple-testing correction
new_status_p_value = chi2_p(new_status, "3.4 new-user status (3 states)")
print("chi-square p-value (3 states):", round(new_status_p_value, 3))

# %% [markdown]
# The two flags describe three observed states: new indicated, not-new indicated, and neither indicated. About 67% fall into the last state, whose real-world meaning is unknown.

# %% [markdown]
# ### 3.5 Referrals
#
# Finally, we check referrals. They are rare, so we look at both the number of customers and their churn rate before deciding how much weight to give the difference.

# %%
# identify customers with no referrals and with at least one referral
ref = train["ReferralsMadeBySubscriber"]
has_no_referrals = ref == 0
has_any_referral = ref > 0
print(f"Zero referrals: {has_no_referrals.mean():.1%} | max: {ref.max()}")

# create clear labels for the two referral groups
referral_labels = has_any_referral.map({False: "0 referrals", True: "1+ referrals"})

# calculate the number of customers and churn rate in each group
referral_groups = train.groupby(referral_labels)["churn"]
referral_summary = referral_groups.agg(n="size", churn_rate="mean")
referral_summary = referral_summary.round(3)
display(referral_summary)

# store the association test for the later multiple-testing correction
referral_p_value = chi2_p(has_any_referral, "3.5 any referral")
print("chi-square p-value:", round(referral_p_value, 3))

# %% [markdown]
# Referrals are uncommon and show little standalone association with churn.

# %% [markdown]
# ### EDA Findings — Usage Change & Retention
#
# Both `PercChange` columns are missing for the same 290 customers (0.71%), including all 120 customers with missing billing and usage values. Their units still need confirmation: 21.8% of `PercChangeMinutes` values are below -100, and its observed lower boundary is consistent with a signed absolute change. `PercChangeRevenues` has 466 values below -100. We will keep both signed variables without treating their names as proof of a percentage formula. Minutes change has a median of -6 and long tails from -3,406 to +5,192.
#
# The user-status flags are not complements: 67% of customers have “No” for both, and the real-world meaning of that state is unconfirmed. Referrals are zero for 95.2% of customers. Retention calls are rare (3.4% called and 1.7% accepted an offer), and contact is more common at longer tenures. `MadeCallToRetentionTeam` is “Yes” exactly when `RetentionCalls` is greater than zero in the training data. None of these cross-sectional observations establishes when contact occurred or why it happened.
#
# For the constructed normalized minutes-change measure, the lowest band has about 40.0% churn versus about 25% near zero; the relationship is U-shaped, which a univariate AUC alone cannot fully capture. Missing change values are associated with 56.9% churn versus 28.6% when observed. The normalized revenue-change bands show a weaker U-shape (30.9%, 27.9% and 30.6%). New-user status and referrals provide weak standalone evidence, and the new-user difference does not survive multiple-testing correction. Falling activity could indicate disengagement, but this snapshot is not a monthly forecasting experiment.
#
# Customers in retention-contact categories show 42.3–47.5% churn versus 28.2% without contact. Among contacted customers, the offer-acceptance association is not significant at 0.05 (p = 0.06); it does not estimate whether offers cause customers to stay. We cannot verify whether these fields would be populated when choosing whom to contact.
# %% [markdown]
# ## 4. Service quality
#
# Next we turn to the customer's experience of using the service: dropped, blocked and unanswered calls,
# along with calls to customer care. We look at how these counts relate to call activity.

# %%
# select the service quality features
cols = [
    'DroppedCalls',
    'BlockedCalls',
    'UnansweredCalls',
    'DroppedBlockedCalls',
    'CustomerCareCalls',
]

# inspect data quality and individual churn associations
feature_overview = overview(train, cols)
display(feature_overview)
feature_churn_signal = churn_signal(train, cols)
display(feature_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# Compare the numerical features in this section.
corr = corr_heatmap(train, cols)

# %% [markdown]
# ### EDA Findings — Service Quality
#
# All five service-quality counts are complete and nonnegative in the training data. Their distributions are strongly right-skewed: most customers have relatively few recorded events, while the maxima are much larger than the 99th percentiles. `CustomerCareCalls` is zero for 54.89% of customers.
#
# Individually, the five features show weak associations with churn, with AUCs from 0.470 to 0.491. Customers with more care calls tend to have slightly lower observed churn, but that pattern alone does not establish a protective effect. The counts also overlap: `DroppedCalls` and `DroppedBlockedCalls` have a Spearman correlation of 0.88. We also examine whether these counts reflect how often customers use the service.
# %% [markdown]
# ## 5. Demographics & region
#
# We look at recorded demographic categories and how often the data shows zero
# or “Unknown”. Those values may reflect incomplete records or household structure.

# %%
# select the demographic and region features
cols = [
    'AgeHH1',
    'AgeHH2',
    'ChildrenInHH',
    'IncomeGroup',
    'Occupation',
    'MaritalStatus',
    'Homeownership',
    'CreditRating',
    'PrizmCode',
    'ServiceArea',
]

# inspect data quality and individual churn associations
feature_overview = overview(train, cols)
display(feature_overview)
feature_churn_signal = churn_signal(train, cols)
display(feature_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# Compare the numerical features in this section.
corr = corr_heatmap(train, cols)

# reuse the same customer count and churn summary for the detailed checks
def churn_table(key):
    # group customers and calculate the count and churn proportion
    grouped_churn = train.groupby(key, observed=True)["churn"]
    table = grouped_churn.agg(["size", "mean"])
    return table.round(3)

# list the recorded states whose meaning needs care
demographic_states = [
    ("AgeHH1", 0),
    ("AgeHH2", 0),
    ("IncomeGroup", 0),
    ("MaritalStatus", "Unknown"),
    ("Homeownership", "Unknown"),
]

# compare each recorded state with all remaining observations
for col, value in demographic_states:
    is_placeholder = train[col] == value
    placeholder_share = is_placeholder.mean()
    churn_for_placeholder = train.loc[is_placeholder, "churn"].mean()
    churn_for_other_values = train.loc[~is_placeholder, "churn"].mean()
    print(
        f"{col} = {value}: share {placeholder_share:.3f}, "
        f"churn {churn_for_placeholder:.3f} vs {churn_for_other_values:.3f}"
    )

# check how often the zero-age and zero-income states occur together
age_zero = train["AgeHH1"] == 0
income_zero = train["IncomeGroup"] == 0
income_zero_share_for_age_zero = income_zero[age_zero].mean()
income_zero_share_for_other_ages = income_zero[~age_zero].mean()
print(
    "IncomeGroup = 0 when AgeHH1 = 0:", round(income_zero_share_for_age_zero, 3),
    "| when AgeHH1 > 0:", round(income_zero_share_for_other_ages, 3),
)

# check whether actual age values are missing in the same customer records
first_age_missing = train["AgeHH1"].isna()
second_age_missing = train["AgeHH2"].isna()
ages_missing_together = first_age_missing == second_age_missing
print("AgeHH1 and AgeHH2 missing for the same customers:", bool(ages_missing_together.all()))

# compare age and income correlations with and without the recorded zero states
demo_num = ["AgeHH1", "AgeHH2", "IncomeGroup"]
all_nonzero = (train[demo_num] > 0).all(axis=1)
all_demographic_correlations = train[demo_num].corr(method="spearman")
display(all_demographic_correlations.round(2))
nonzero_demographic_data = train.loc[all_nonzero, demo_num]
nonzero_demographic_correlations = nonzero_demographic_data.corr(method="spearman")
display(nonzero_demographic_correlations.round(2))

# describe churn across the observed nonzero first-age bands
observed_first_age = train["AgeHH1"].where(train["AgeHH1"] > 0)
first_age_bands = pd.cut(observed_first_age, [0, 30, 40, 50, 60, 100])
display(churn_table(first_age_bands))

# describe the recorded categories without imposing an order on their churn rates
demographic_categories = ["IncomeGroup", "CreditRating", "PrizmCode", "Occupation", "MaritalStatus"]
for col in demographic_categories:
    display(churn_table(col))

# count the small service areas and the customers represented by them
area_size = train["ServiceArea"].value_counts()
small_service_areas = area_size < 30
small_area_count = int(small_service_areas.sum())
small_area_customers = area_size[small_service_areas].sum()
small_area_customer_share = round(small_area_customers / len(train), 3)

# count the distinct three-letter prefixes without assuming a geographic meaning
service_area_prefixes = train["ServiceArea"].str[:3]
prefix_count = service_area_prefixes.nunique()
print(
    "Service areas:", area_size.size,
    "| with fewer than 30 customers:", small_area_count,
    "| their share of customers:", small_area_customer_share,
    "| groups by first three letters:", prefix_count,
)

# %% [markdown]
# ### EDA Findings — Demographics & region
#
# Several demographic fields contain codes whose meanings need care. `AgeHH1` is zero for 27.38% of customers, `AgeHH2` for 51.24%, and `IncomeGroup` for 25.17%; `MaritalStatus` and `Homeownership` are recorded as “Unknown” for 38.57% and 33.52%. These states often occur together: income is zero for 81.1% of customers with `AgeHH1` equal to zero, versus 4.1% among the others. They are not automatically interchangeable with missing values: for example, zero for `AgeHH2` could mean there is no second adult. Both ages are actually missing for the same 1.75% of customers, and `ServiceArea` is missing for 0.05%.
#
# Recorded nonzero ages range from 18 to 99, with a 99th percentile of 76. `Occupation` is “Other” for 73.8% of customers, `CreditRating` is most often “2-High” (37.2%), and `PrizmCode` is mostly “Other” or “Suburban”. The shared zero codes also affect correlations: the `AgeHH1` and `IncomeGroup` Spearman correlation falls from 0.58 to 0.11 when zeros are excluded. `ServiceArea` has 729 recorded levels; 488 have fewer than 30 customers, accounting for 9.1% of customers. Extreme area-level churn rates can therefore come from small groups.
#
# The individual age and income AUCs are close to 0.5, although churn falls from 30.8% among customers with known age at most 30 to 25.6% above 60. Recorded unknown states generally have churn rates about one to three percentage points higher than their comparison groups. `CreditRating` varies non-monotonically: rates range from 21.8% for “5-Low” to about 31% for several higher ratings. Rural `PrizmCode` has 31.9% churn among 1,916 customers, versus 28.1% for suburban customers. `ChildrenInHH` differs by 0.8 percentage points; the highest occupation rate comes from only 135 homemakers. We interpret these rates as descriptive patterns.
# %% [markdown]
# ## 6. Handset & lifestyle
#
# Handset characteristics may help explain why customers with similar usage have different churn
# patterns. The lifestyle variables add some wider context, although several are rare or closely
# related. We check the device patterns first, then look at those overlaps.

# %%
# select the handset and lifestyle features
cols = [
    'HandsetPrice',
    'HandsetRefurbished',
    'HandsetWebCapable',
    'TruckOwner',
    'RVOwner',
    'OwnsMotorcycle',
    'OwnsComputer',
    'HasCreditCard',
    'BuysViaMailOrder',
    'RespondsToMailOffers',
    'OptOutMailings',
    'NonUSTravel',
]

# inspect data quality and individual churn associations
feature_overview = overview(train, cols)
display(feature_overview)
feature_churn_signal = churn_signal(train, cols)
display(feature_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# select the yes/no fields for the correlation plot
binary_lifestyle = []
for column in cols:
    observed_values = set(train[column].dropna().unique())
    if observed_values.issubset({'Yes', 'No'}):
        binary_lifestyle.append(column)

# keep the plotting data separate from the training frame
handset_lifestyle_encoded = train[binary_lifestyle].copy()

# represent yes/no values as numbers for this plot only
for column in binary_lifestyle:
    handset_lifestyle_encoded[column] = handset_lifestyle_encoded[column].map({'Yes': 1, 'No': 0})

# show the relationships between the binary fields
corr = corr_heatmap(handset_lifestyle_encoded, binary_lifestyle)

# list the handset and mail features for the detailed churn tables
handset_features = [
    "HandsetPrice",
    "HandsetRefurbished",
    "HandsetWebCapable",
    "BuysViaMailOrder",
    "RespondsToMailOffers",
]

# summarize categories and show the most common ones first
for feature in handset_features:
    print(f"\nFeature: {feature}")
    churn_by_value = train.groupby(feature, dropna=False)["churn"]
    summary = churn_by_value.agg(
        customers="size",
        churners="sum",
        churn_rate="mean",
    )
    summary = summary.sort_values("customers", ascending=False)

    # express churn as a percentage in these descriptive tables
    summary["churn_rate"] = summary["churn_rate"] * 100
    display(summary.round(2))

# list the pairs whose overlap we want to inspect
pairs = [
    ("BuysViaMailOrder", "RespondsToMailOffers"),
    ("TruckOwner", "RVOwner"),
]

# inspect each pair's combinations and percentage agreement
for feature_a, feature_b in pairs:
    print(f"\n{feature_a} vs {feature_b}")

    # count each combination of categories
    category_combinations = pd.crosstab(
        train[feature_a],
        train[feature_b],
        margins=True,
    )
    display(category_combinations)

    # compare only customers with observed values in both fields
    pair_columns = [feature_a, feature_b]
    valid_pair = train[pair_columns].dropna()
    matching_values = valid_pair[feature_a] == valid_pair[feature_b]
    same = matching_values.mean()
    print(f"Exact agreement: {same:.2%}")

# %% [markdown]
# ### EDA Findings — Handset & Lifestyle
#
# The analyzed binary features are complete for all 40,837 training customers. About 57% of `HandsetPrice` values are “Unknown”, so we preserve that recorded category rather than assume a price. Web-capable handsets are common (36,818 customers, or 90.16%), while 5,619 customers (13.76%) have refurbished handsets.
#
# Some lifestyle variables overlap closely. About 36% report buying by mail order and 37% respond to mail offers; the two indicators agree for 98.49% of customers, and everyone recorded as buying by mail is also recorded as responding. `TruckOwner` and `RVOwner` agree 89.53% of the time. Every recorded RV owner is also a truck owner, but many truck owners are not RV owners. These relationships are close, but the pairs are not identical.
#
# Observed churn is 37.20% for customers without web-capable handsets and 27.90% for those with them. It is 32.35% for refurbished handsets versus 28.26% otherwise. Mail-order buyers and offer respondents also have slightly lower observed churn than their counterparts, around 27.4% versus 29.6%. We treat these patterns as descriptive associations.
# %% [markdown]
# ## 7. Relationships among features
#
# We look at equipment age within tenure ranges, call counts relative to usage,
# and spending alongside changes in activity.

# %% [markdown]
# ### 7.1 Are the related columns interchangeable?
#
# We check whether related fields contain exactly the same information.

# %%
# compare the combined quality measure with the sum of dropped and blocked calls
dbc = train["DroppedCalls"] + train["BlockedCalls"]
combined_call_difference = train["DroppedBlockedCalls"] - dbc
absolute_call_difference = combined_call_difference.abs()
nearly_equal_calls = absolute_call_difference < 0.11
nearly_equal_share = nearly_equal_calls.mean()
print(
    "Share within 0.11 of DroppedCalls + BlockedCalls:",
    round(nearly_equal_share, 3),
)

# count the observed combinations of the two cellphone-user flags
user_flag_counts = pd.crosstab(
    train["NewCellphoneUser"],
    train["NotNewCellphoneUser"],
)
print("\nCrosstab NewCellphoneUser vs NotNewCellphoneUser:")
print(user_flag_counts)

# group larger handset counts together for this display only
display_handsets = train["Handsets"].clip(upper=5)
display_handset_models = train["HandsetModels"].clip(upper=5)
handset_counts = pd.crosstab(display_handsets, display_handset_models)
print("\nCrosstab Handsets vs HandsetModels:")
print(handset_counts)

# group larger subscription counts together for this display only
display_unique_subs = train["UniqueSubs"].clip(upper=5)
display_active_subs = train["ActiveSubs"].clip(upper=5)
subscription_counts = pd.crosstab(display_unique_subs, display_active_subs)
print("\nCrosstab UniqueSubs vs ActiveSubs:")
print(subscription_counts)

# %% [markdown]
# These checks show overlap rather than exact interchangeability. The quality total differs from the sum in some records, both user flags can be No, and handset and subscription counts are related without being identical.

# %% [markdown]
# ### 7.2 What the recorded unknown states tell us
#
# We compare churn across recorded zero and Unknown states. This helps decide what information
# preprocessing must preserve without assuming those codes are ordinary values or confirmed missingness.

# %%
# compare each recorded placeholder with the other observed states
rows = []
for column, placeholder_value in PLACEHOLDERS.items():
    if column not in train.columns:
        continue

    # identify customers with and without this recorded state
    is_ph = train[column] == placeholder_value
    has_other_value = ~is_ph

    # calculate the share and churn rates for the two groups
    placeholder_share = is_ph.mean()
    placeholder_churn = train.loc[is_ph, "churn"].mean()
    other_churn = train.loc[has_other_value, "churn"].mean()

    # record one summary row for this feature
    rows.append({
        "column": column,
        "placeholder": placeholder_value,
        "share": round(placeholder_share, 3),
        "churn_if_placeholder": round(placeholder_churn, 3),
        "churn_otherwise": round(other_churn, 3),
    })

# display the feature summaries together
placeholder_summary = pd.DataFrame(rows)
placeholder_summary = placeholder_summary.set_index("column")
display(placeholder_summary)

# %% [markdown]
# Recorded zero and “Unknown” states have different churn rates from other values. We distinguish these recorded states from missing values.

# %% [markdown]
# ### 7.3 Churn by tenure, month by month
# Is there a point in time where churn suddenly changes (e.g. contract expiry)?

# %%
# calculate customer counts and churn rates at each tenure
tenure_churn_groups = train.groupby("MonthsInService")["churn"]
rate = tenure_churn_groups.agg(["size", "mean"])

# keep tenure values supported by at least one hundred customers
has_enough_customers = rate["size"] >= 100
rate = rate.loc[has_enough_customers]

# plot the churn rate at each retained tenure
ax = rate["mean"].plot(
    marker="o",
    figsize=(10, 3),
    title="churn rate by MonthsInService (months with >= 100 customers)",
)

# add the overall training churn rate as a reference line
overall_churn_rate = train["churn"].mean()
ax.axhline(overall_churn_rate, ls="--", c="grey")

# show the completed tenure plot
plt.show()

# %% [markdown]
# We observe that churn varies by tenure. The dataset has no contract expiry
# date, so we cannot determine whether renewal timing explains the pattern.

# %% [markdown]
# ### 7.4 Tenure × spending segments
# We compare observed churn by tenure and monthly revenue.

# %%
# divide tenure into the groups used in the descriptive analysis
tenure_boundaries = [0, 10, 12, 24, 1000]
tenure_labels = ["6-10m", "11-12m", "13-24m", ">24m"]
tenure_grp = pd.cut(
    train["MonthsInService"],
    bins=tenure_boundaries,
    labels=tenure_labels,
)

# divide monthly revenue into three equally sized spending groups
spend_grp = pd.qcut(
    train["MonthlyRevenue"],
    q=3,
    labels=["low", "mid", "high"],
)

# count customers in each tenure and spending combination
tenure_spend_counts = pd.crosstab(tenure_grp, spend_grp)
print("Tenure vs Spend Crosstab Count:")
display(tenure_spend_counts)

# calculate churn rates for the same combinations
tenure_spend_churn = pd.crosstab(
    tenure_grp,
    spend_grp,
    values=train["churn"],
    aggfunc="mean",
)
tenure_spend_churn = tenure_spend_churn.round(3)
print("\nTenure vs Spend Crosstab Mean Churn:")
display(tenure_spend_churn)

# %% [markdown]
# We see similar churn rates across spending levels within the highest-churn
# tenure band.

# %% [markdown]
# ### 7.5 Usage drop x spending level
#
# Does a drop in usage mean the same thing for low and high spenders? Usage change is measured relative to `MonthlyMinutes` (see 3.1).

# %%
# group the normalized usage-change values into three bands
change_boundaries = [-1.01, -0.2, 0.2, np.inf]
change_labels = ["ratio < -0.2", "ratio -0.2 to 0.2", "ratio > 0.2"]
change_band = pd.cut(
    rel_change_min,
    bins=change_boundaries,
    labels=change_labels,
)
change_band = change_band.rename("usage change")

# divide monthly revenue into three equally sized spending bands
spend_band = pd.qcut(
    train["MonthlyRevenue"],
    q=3,
    labels=["low spend", "mid spend", "high spend"],
)
spend_band = spend_band.rename("spending")

# count customers in each combination of usage change and spending
change_spend_counts = pd.crosstab(change_band, spend_band)
print("Customers:")
display(change_spend_counts)

# calculate churn rates for the same combinations
change_spend_churn = pd.crosstab(
    change_band,
    spend_band,
    values=train["churn"],
    aggfunc="mean",
)
change_spend_churn = change_spend_churn.round(3)
print("Churn rate:")
display(change_spend_churn)

# %% [markdown]
# We observe higher churn in the low normalized-change band across spending
# levels. The original units of the change field remain uncertain.

# %% [markdown]
# ### 7.6 Handset age within tenure
#
# Is handset age just another measure of tenure, or does it add information? Tenure bands are the same as in 3.3 and 7.4.

# %%
# leave invalid negative equipment ages out of this age comparison
equipment_days = train["CurrentEquipmentDays"]
nonnegative_equipment_age = equipment_days >= 0
valid_equipment_days = equipment_days.where(nonnegative_equipment_age)

# distinguish handsets up to one year old from older handsets
handset_age = pd.cut(
    valid_equipment_days,
    bins=[-np.inf, 365, np.inf],
    labels=["handset <= 1 year", "handset > 1 year"],
)
handset_age = handset_age.rename("handset age")

# compare customer counts and churn rates within the existing tenure bands
tenure_handset_counts = pd.crosstab(tenure_band, handset_age)
age_rate = pd.crosstab(
    tenure_band,
    handset_age,
    values=train["churn"],
    aggfunc="mean",
)
age_rate = age_rate.round(3)
print("Customers:")
display(tenure_handset_counts)
print("Churn rate:")
display(age_rate)

# describe the relationship between tenure and the original equipment-age column
tenure_equipment_correlation = train["MonthsInService"].corr(
    train["CurrentEquipmentDays"],
    method="spearman",
)
print(
    "Spearman MonthsInService vs CurrentEquipmentDays:",
    round(tenure_equipment_correlation, 2),
)

# calculate the approximate original-handset heuristic without treating it as known history
minimum_original_age = 0.9 * train["MonthsInService"]
minimum_original_age = minimum_original_age * 30.4
original_handset = equipment_days >= minimum_original_age
original_handset = original_handset.astype(float)

# preserve missing and invalid parent values in the heuristic
known_tenure = train["MonthsInService"].notna()
valid_heuristic_parents = nonnegative_equipment_age & known_tenure
original_handset = original_handset.where(valid_heuristic_parents)

# summarize the heuristic and its descriptive churn association
heuristic_share = original_handset.mean()
matches_heuristic = original_handset.eq(1)
does_not_match_heuristic = original_handset.eq(0)
matched_churn = train.loc[matches_heuristic, "churn"].mean()
unmatched_churn = train.loc[does_not_match_heuristic, "churn"].mean()
observed_heuristic = original_handset.dropna()
heuristic_p_value = chi2_p(observed_heuristic, "7.7 original handset")
print(
    f"Approximate original-handset heuristic (valid ages only): {heuristic_share:.1%} of customers | "
    f"churn {matched_churn:.1%} vs {unmatched_churn:.1%} | "
    f"p = {heuristic_p_value:.1e}"
)

# plot the churn rates for handset age within each tenure band
ax = age_rate.plot.bar(figsize=(8, 3), rot=0)
overall_churn_rate = train["churn"].mean()
ax.axhline(overall_churn_rate, ls="--", c="grey")
ax.set_ylabel("churn rate")
ax.set_title("Churn rate by tenure and handset age (no bar = no customers)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Equipment age separates churn rates within tenure bands, so we retain both raw measures. The original-handset heuristic is not included: equipment age and tenure alone cannot confirm that the current handset is the original one.

# %% [markdown]
# ### 7.7 Service-quality counts versus rates
#
# Service-quality counts rise with call volume (see 7.5). This compares the raw counts with the same counts per call and per minute. AUC below 0.5 means "higher value, less churn".

# %%
# select the service-quality measurements to compare
quality_cols = [
    "DroppedCalls",
    "BlockedCalls",
    "UnansweredCalls",
    "DroppedBlockedCalls",
    "CustomerCareCalls",
]

# calculate the activity denominators used in the smoothed ratios
total_calls = train["PeakCallsInOut"] + train["OffPeakCallsInOut"]
smoothed_call_volume = total_calls + 1
smoothed_minutes = train["MonthlyMinutes"] + 1

# compare each raw measurement with its per-call and per-minute versions
quality_comparison_rows = []
for column in quality_cols:
    raw_values = train[column]
    values_per_call = raw_values / smoothed_call_volume
    values_per_minute = raw_values / smoothed_minutes

    quality_comparison_rows.append({
        "raw count": auc_of(raw_values),
        "per call": auc_of(values_per_call),
        "per minute": auc_of(values_per_minute),
    })

# keep the same feature order in the comparison table
quality_comparison = pd.DataFrame(quality_comparison_rows, index=quality_cols)
quality_comparison = quality_comparison.rename_axis("AUC")
display(quality_comparison)

# %% [markdown]
# Normalizing by activity changes some association directions, suggesting that raw counts partly reflect exposure. We treat raw counts and call rates as different measurements.

# %% [markdown]
# ### 7.8 Are customer-care callers just heavy users?
#
# Customers who call customer care churn less. Heavy users call care more often, so this compares callers and non-callers within the same usage level.

# %%
# divide monthly usage into three equally sized bands
usage_band = pd.qcut(
    train["MonthlyMinutes"],
    q=3,
    labels=["low use", "mid use", "high use"],
)
usage_band = usage_band.rename("usage level")

# identify customers with any recorded customer-care call
has_care_call = train["CustomerCareCalls"] > 0
care_labels = np.where(has_care_call, "any care call", "no care call")
care = pd.Series(care_labels, index=train.index, name="customer care")

# count callers and non-callers within each usage band
usage_care_counts = pd.crosstab(usage_band, care)
print("Customers:")
display(usage_care_counts)

# compare churn for the same customer groups
usage_care_churn = pd.crosstab(
    usage_band,
    care,
    values=train["churn"],
    aggfunc="mean",
)
usage_care_churn = usage_care_churn.round(3)
print("Churn rate:")
display(usage_care_churn)

# %% [markdown]
# Care callers have lower observed churn within each broad usage band as well. We use this comparison to describe the observed pattern.

# %% [markdown]
# ### 7.9 Revenue relative to usage and bill structure
#
# Combines billing and usage using revenue / (minutes + 1), an exposure-normalized proxy rather than a literal tariff per minute. Recurring charge / positive revenue is a second exploratory ratio.

# %%
# calculate revenue relative to usage with the original smoothed denominator
smoothed_monthly_minutes = train["MonthlyMinutes"] + 1
rev_per_min = train["MonthlyRevenue"] / smoothed_monthly_minutes

# calculate recurring charge share only where revenue is positive
positive_revenue = train["MonthlyRevenue"] > 0
valid_monthly_revenue = train["MonthlyRevenue"].where(positive_revenue)
recurring_share = train["TotalRecurringCharge"] / valid_monthly_revenue

# compare the univariate scores of the original columns and the two ratios
revenue_comparison_rows = [
    {"AUC": auc_of(train["MonthlyRevenue"])},
    {"AUC": auc_of(train["MonthlyMinutes"])},
    {"AUC": auc_of(rev_per_min)},
    {"AUC": auc_of(recurring_share)},
]
revenue_comparison_labels = [
    "MonthlyRevenue",
    "MonthlyMinutes",
    "revenue per minute",
    "recurring share of bill",
]
revenue_comparison = pd.DataFrame(
    revenue_comparison_rows,
    index=revenue_comparison_labels,
)
display(revenue_comparison)

# divide the revenue-to-usage proxy into five equally sized groups
revenue_ratio_labels = [
    "lowest ratio fifth",
    "2nd",
    "3rd",
    "4th",
    "highest ratio fifth",
]
revenue_ratio_groups = pd.qcut(rev_per_min, q=5, labels=revenue_ratio_labels)

# summarize customer counts and churn rates for those groups
revenue_ratio_churn = train.groupby(revenue_ratio_groups, observed=True)["churn"]
rpm_rate = revenue_ratio_churn.agg(n="size", churn_rate="mean")
rpm_rate = rpm_rate.round(3)
print("Churn rate by revenue/(minutes+1) proxy (fifths):")
display(rpm_rate.T)

# plot the churn rates and the overall training reference
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(range(len(rpm_rate)), rpm_rate["churn_rate"])
ax.set_xticks(range(len(rpm_rate)), rpm_rate.index, fontsize=8)
overall_churn_rate = train["churn"].mean()
ax.axhline(overall_churn_rate, ls="--", c="grey")
ax.set_ylabel("churn rate")
ax.set_title("Churn rate by revenue/(minutes+1) proxy")
plt.tight_layout()
plt.show()

# %% [markdown]
# We observe that churn rises from 24.9% to 34.7% across fifths of the
# revenue-to-usage measure. We use the ratio as a spending-to-activity proxy
# because the dataset does not specify a price per minute.

# %% [markdown]
# ### 7.10 Do unknown values cluster in the same customers?
#
# `Occupation` = "Other" is added here because it covers 74% of customers and may act as a placeholder too.

# %%
# create a separate indicator for each recorded placeholder state
unknown_indicator_columns = {}
for column, placeholder_value in PLACEHOLDERS.items():
    matches_placeholder = train[column] == placeholder_value
    unknown_indicator_columns[column] = matches_placeholder.astype(int)
unknown_flags = pd.DataFrame(unknown_indicator_columns)

# add the broad occupation category without assuming that it means missing
occupation_is_other = train["Occupation"] == "Other"
unknown_flags["Occupation=Other"] = occupation_is_other.astype(int)

# inspect how often the recorded states appear together
unknown_correlations = unknown_flags.corr()
unknown_correlations = unknown_correlations.round(2)
display(unknown_correlations)

# compare other recorded states among customers with and without the first age zero code
age_unknown = unknown_flags["AgeHH1"] == 1
other_age_state = ~age_unknown
income_unknown_with_age_zero = unknown_flags.loc[age_unknown, "IncomeGroup"].mean()
income_unknown_otherwise = unknown_flags.loc[other_age_state, "IncomeGroup"].mean()
occupation_other_with_age_zero = unknown_flags.loc[age_unknown, "Occupation=Other"].mean()
print(
    f"IncomeGroup = 0 among AgeHH1 = 0: {income_unknown_with_age_zero:.1%} "
    f"(otherwise {income_unknown_otherwise:.1%})"
)
print(f"Occupation = Other among AgeHH1 = 0: {occupation_other_with_age_zero:.1%}")

# record the same three descriptive association tests for later correction
placeholder_p_values = {}
for column in ["AgeHH1", "IncomeGroup", "HandsetPrice"]:
    test_name = f"7.11 {column} placeholder"
    p_value = chi2_p(unknown_flags[column], test_name)
    placeholder_p_values[column] = f"{p_value:.1e}"
print("chi-square p-values:", placeholder_p_values)

# count the selected recorded unknown states for each customer
unknown_count_columns = [
    "AgeHH1",
    "IncomeGroup",
    "MaritalStatus",
    "Homeownership",
    "HandsetPrice",
]
n_unknown = unknown_flags[unknown_count_columns].sum(axis=1)

# summarize churn by the number of recorded unknown states
unknown_count_groups = train.groupby(n_unknown)["churn"]
unknown_count_summary = unknown_count_groups.agg(n="size", churn_rate="mean")
unknown_count_summary = unknown_count_summary.round(3)
print("Churn rate by number of unknown values (AgeHH1, IncomeGroup, MaritalStatus, Homeownership, HandsetPrice):")
display(unknown_count_summary.T)

# %% [markdown]
# Several demographic states cluster in the same records, while unknown handset price behaves differently.

# %% [markdown]
# ### 7.11 Which test results survive multiple-testing correction?
#
# Sections 3 and 7 run several chi-square tests, which raises the chance of a false positive. Holm correction adjusts the p-values for that.

# %%
# order the recorded p-values from smallest to largest
tests = pd.Series(p_values)
tests = tests.sort_values()
m = len(tests)

# apply the decreasing multipliers used by the Holm correction
remaining_test_counts = np.arange(m, 0, -1)
scaled_p_values = tests * remaining_test_counts

# keep adjusted values nondecreasing and limit them to one
holm = scaled_p_values.cummax()
holm = holm.clip(upper=1)

# format the original and adjusted values for the display table
original_p_value_text = []
for p_value in tests:
    original_p_value_text.append(f"{p_value:.2g}")

adjusted_p_value_text = []
for p_value in holm:
    adjusted_p_value_text.append(f"{p_value:.2g}")

# identify which associations remain below the chosen significance threshold
significant_after_correction = holm < 0.05
test_summary = pd.DataFrame({
    "p_value": original_p_value_text,
    "holm_adjusted": adjusted_p_value_text,
    "significant_after_correction": significant_after_correction,
}, index=tests.index)
display(test_summary)

# %% [markdown]
# After correction for multiple tests, some associations are less clear.
# We interpret these tests as descriptions of the training customers.

# %% [markdown]
# ## Inputs for predicting churn
#
# We keep churn as the outcome and use only inputs available when we decide whom
# to contact. We use the customer identifier and split label to track rows.
# We exclude the three retention fields because we cannot verify when contact or
# offer acceptance occurred.
#
# We keep signed billing and usage-change measurements, and treat negative equipment
# ages as missing. We keep recorded categories such as “Unknown” distinct from
# missing entries. The model notebooks compare how we fill missing values and
# encode categories, meaning how we turn their labels into numeric inputs.
#
# We use monthly revenue to describe spending in this analysis. It is a spending
# measure, and the team can use it alongside churn risk when defining outreach priorities.
