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
# Our project asks how much different customer information helps us predict telecom churn.
# We look at the individual feature groups and their relationships, then bring the findings together
# into the final feature and preprocessing decisions.
#
# **Rules**
# - Use only the training split from `load_train()` for analysis and decisions
# - Leave the test outcomes untouched until final evaluation
#
# We use the shared `src/` loader and helpers from this local repository. Follow the root README to create the environment and open this Jupytext file in JupyterLab or VS Code. Preprocessing remains in its own
# notebook; the model baseline, group-addition order, training and evaluation belong to the modeling stage.

# %%
# locate the repository root
from pathlib import Path
import sys

project_root = Path.cwd()
if not (project_root / 'src' / 'data.py').is_file():
    raise FileNotFoundError('Open this notebook from the repository root')

# make the shared project modules available
sys.path.insert(0, str(project_root))

# import the shared data loader
import pandas as pd
from IPython.display import display
from src.data import load_train


# import the shared analysis helpers
from src.eda_utils import (
    FEATURE_GROUPS,
    PLACEHOLDERS,
    overview,
    churn_signal,
    churn_rate_by,
    plot_column,
    corr_heatmap,
    summary_skeleton,
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
# Before looking at churn relationships, we check the shape of the data: which columns are numeric,
# where values are missing, and which recorded categories need care. This gives us a common starting
# point for the six feature groups.

# %%
# collect the feature names from all six groups
all_cols = []
for group_columns in FEATURE_GROUPS.values():
    for column in group_columns:
        all_cols.append(column)

# show the data types and data quality checks for every feature
overview(train, all_cols)

# %% [markdown]
# ### 0.1 Preparing the overview
#
# For these comparisons, we make a separate analysis copy and identify numeric, categorical and yes/no columns. Negative equipment ages are masked in this copy because an age cannot be negative.

# %%
# import the tools used for the cross-group analysis
import itertools as y_it
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

# create a separate training copy with a simple row index
y_train = train.copy()
y_train = y_train.reset_index(drop=True)

# # copy the feature lists so the shared group definitions stay unchanged
y_groups = {}
for group_name, group_features in FEATURE_GROUPS.items():
    y_groups[group_name] = list(group_features)

# record which group each feature belongs to
y_owner = {}
for group_name, group_features in y_groups.items():
    for feature in group_features:
        y_owner[feature] = group_name

# list all predictors and identify the retention information
y_features = list(y_owner)
y_retention = [
    'RetentionCalls',
    'RetentionOffersAccepted',
    'MadeCallToRetentionTeam',
]

# keep retention information outside the main feature list
y_main = []
for feature in y_features:
    if feature not in y_retention:
        y_main.append(feature)

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

# check that the full and main feature counts match the project definition
assert len(y_features) == 56
assert len(y_main) == 53

# %% [markdown]
# The main set contains 53 raw features. The shared overview helpers describe stored data types; for the final model representation we treat IncomeGroup and HandsetPrice as categories, preserving code 0 and “Unknown”. For this analysis, negative equipment ages are treated as missing because an equipment age cannot be negative.

# %% [markdown]
# ### 0.2 Comparing numerical and yes/no features across groups
#
# We continue by looking at features from different groups together. This can reveal when two groups describe similar customer behaviour, which matters when we later compare their value to a model. Pearson correlation looks for a straight-line relationship; Spearman correlation looks for a consistent increase or decrease, even when the relationship is not a straight line. Yes/no values are temporarily represented as 1 and 0.

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

# identify pairs whose features belong to different groups
first_feature_group = y_pairs['feature_1'].map(y_owner)
second_feature_group = y_pairs['feature_2'].map(y_owner)
y_pairs['cross_group'] = first_feature_group != second_feature_group

# identify pairs that contain a retention feature
first_feature_is_retention = y_pairs['feature_1'].isin(y_retention)
second_feature_is_retention = y_pairs['feature_2'].isin(y_retention)
y_pairs['retention_pair'] = first_feature_is_retention | second_feature_is_retention

# select the correlation columns that belong to the main representation
y_safe_corr = []
for feature in y_corr_data.columns:
    if feature in y_main:
        y_safe_corr.append(feature)

# keep cross-group results without retention information
is_cross_group = y_pairs['cross_group']
has_no_retention = ~y_pairs['retention_pair']
eligible_pair_rows = is_cross_group & has_no_retention
eligible_pairs = y_pairs.loc[eligible_pair_rows]

# sort by correlation strength so strong negative values also appear near the top
sorted_pairs = eligible_pairs.sort_values(
    'spearman',
    key=abs,
    ascending=False,
)

# show the twenty strongest pairs with rounded display values
top_pairs = sorted_pairs.head(20)
display(top_pairs.round(3))

# %% [markdown]
# The table shows the strongest cross-group relationships, together with the number of customers used for each pair. Off-peak calls and unanswered calls have a Spearman correlation of 0.826, peak calls and unanswered calls have a correlation of 0.821. Customers who make more calls may naturally have more unanswered calls. This suggests that usage and service-quality features overlap, so we should compare the groups in a model before deciding whether either adds useful information.
#
# The following graphs show all numerical and yes/no features in the main set. The blank squares are pairs from the same group, which are already covered in the individual group sections. Darker colours show stronger relationships between different groups.

# %%
# create a mask that hides pairs from the same feature group
y_same_group = pd.DataFrame(
    False,
    index=y_safe_corr,
    columns=y_safe_corr,
)
for feature_1 in y_safe_corr:
    for feature_2 in y_safe_corr:
        same_feature_group = y_owner[feature_1] == y_owner[feature_2]
        y_same_group.loc[feature_1, feature_2] = same_feature_group

# draw a separate heatmap for each correlation measure
correlation_matrices = [
    ('Pearson', y_pearson),
    ('Spearman', y_spearman),
]
for y_name, y_matrix in correlation_matrices:
    # select only the features in the main representation
    main_correlations = y_matrix.loc[y_safe_corr, y_safe_corr]

    # use the same colour range and group mask for both plots
    plt.figure(figsize=(17, 15))
    sns.heatmap(
        main_correlations,
        vmin=-1,
        vmax=1,
        center=0,
        cmap='vlag',
        square=True,
        mask=y_same_group,
        cbar_kws={'label': 'Correlation'},
    )

    # label and display the correlation heatmap
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(f'{y_name}: cross-group numerical and yes/no features')
    plt.tight_layout()
    plt.show()


# %% [markdown]
# Both graphs show a clear block of relationships between call usage and service quality. The Spearman graph is darker for several pairs, suggesting that some relationships are easier to see in the order of customers than in a straight-line comparison. Strong correlation alone is not a reason to remove a feature; the groups may still provide different useful information.

# %% [markdown]
# ### 0.3 Comparing categories across groups
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

# identify pairs from different feature groups
group_1 = y_cat_pairs['feature_1'].map(y_owner)
group_2 = y_cat_pairs['feature_2'].map(y_owner)
y_cat_pairs['cross_group'] = group_1 != group_2

# identify pairs where both features are yes/no fields
first_is_binary = y_cat_pairs['feature_1'].isin(y_binary)
second_is_binary = y_cat_pairs['feature_2'].isin(y_binary)
y_cat_pairs['both_binary'] = first_is_binary & second_is_binary

# identify pairs containing retention information
first_is_retention = y_cat_pairs['feature_1'].isin(y_retention)
second_is_retention = y_cat_pairs['feature_2'].isin(y_retention)
y_cat_pairs['retention_pair'] = first_is_retention | second_is_retention

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

# select the categorical features in the main representation
y_cat_main = []
for feature in y_categorical:
    if feature in y_main:
        y_cat_main.append(feature)

# create a mask for pairs already covered by the other checks
y_cat_mask = pd.DataFrame(
    False,
    index=y_cat_main,
    columns=y_cat_main,
)
for feature_a in y_cat_main:
    for feature_b in y_cat_main:
        same_group = y_owner[feature_a] == y_owner[feature_b]
        feature_a_is_binary = feature_a in y_binary
        feature_b_is_binary = feature_b in y_binary
        both_binary = feature_a_is_binary and feature_b_is_binary
        y_cat_mask.loc[feature_a, feature_b] = same_group or both_binary

# select and plot the main categorical association matrix
main_associations = y_v.loc[y_cat_main, y_cat_main]
plt.figure(figsize=(12, 10))
sns.heatmap(
    main_associations,
    vmin=0,
    vmax=1,
    cmap='Blues',
    square=True,
    mask=y_cat_mask,
    cbar_kws={'label': "Bias-corrected Cramér's V"},
)
plt.title('Categorical main predictors across groups')
plt.tight_layout()
plt.show()

# keep cross-group pairs without retention information or two binary fields
is_cross_group = y_cat_pairs['cross_group']
has_no_retention = ~y_cat_pairs['retention_pair']
is_not_binary_pair = ~y_cat_pairs['both_binary']
eligible_pair_rows = is_cross_group & has_no_retention & is_not_binary_pair
eligible_pairs = y_cat_pairs.loc[eligible_pair_rows]

# show the fifteen strongest categorical associations
sorted_pairs = eligible_pairs.sort_values('V', ascending=False)
strongest_pairs = sorted_pairs.head(15)
display(strongest_pairs.round(3))


# %% [markdown]
# The strongest cross-group pairs include IncomeGroup and HasCreditCard (V = 0.827), and Homeownership and HasCreditCard (V = 0.800). These values suggest considerable overlap between parts of the demographic and lifestyle groups. However, unknown or placeholder values may contribute to these patterns, so we check that below before drawing conclusions. Rare categories, especially ServiceArea, also need careful interpretation.

# %% [markdown]
# ### 0.4 Comparing numerical features with categories
#
# We also want to see whether numerical values differ across categories in another feature group. For example, the number of handset models may differ between recorded handset-price categories. The measure below, adjusted η², describes how much of the numerical variation is associated with the categories. It runs from 0 to 1; it does not show a direction or prove that one feature causes the other. Yes/no categories are covered in the earlier correlation graphs.

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

# identify pairs from different feature groups
numerical_group = y_mixed['numeric'].map(y_owner)
categorical_group = y_mixed['categorical'].map(y_owner)
y_mixed['cross_group'] = numerical_group != categorical_group

# identify pairs containing retention information
numerical_is_retention = y_mixed['numeric'].isin(y_retention)
categorical_is_retention = y_mixed['categorical'].isin(y_retention)
y_mixed['retention_pair'] = numerical_is_retention | categorical_is_retention

# keep cross-group pairs without retention information or binary categories
is_cross_group = y_mixed['cross_group']
has_no_retention = ~y_mixed['retention_pair']
is_binary_category = y_mixed['categorical'].isin(y_binary)
eligible_pair_rows = is_cross_group & has_no_retention & ~is_binary_category
y_mixed_main = y_mixed.loc[eligible_pair_rows]

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
plt.title('Numerical and categorical features across groups')
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

# verify that every unique pair of original features has been checked
feature_count = len(y_features)
expected_pairs = feature_count * (feature_count - 1) // 2
assert y_coverage.sum() == expected_pairs

# %% [markdown]
# The strongest cross-group relationships include HandsetModels with HandsetPrice (adjusted η² = 0.514) and Handsets with HandsetPrice (0.505). These are useful leads for the comparison, but categories with many rare values can make the measures less reliable. The three screens together calculate all 1,540 pairs of the 56 original predictors; the tables and graphs focus on pairs from different groups.

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
# The cross-group checks compare the two cellphone-user flags and the call-count columns. Looking at the exact combinations and the size of the differences explains why we retain these overlapping measurements in their raw groups.

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
# The user-status check, revisited in section 7.1, shows 27,363 customers with No for both
# flags. They are not exact opposites, and we retain both. DroppedBlockedCalls is within 0.11 of
# DroppedCalls plus BlockedCalls for 96.72% of customers, but 482 differences exceed 1 and the largest
# is 81.7. These are overlapping rather than exactly interchangeable measurements, so all three
# raw quality columns remain in the chosen representation.

# %% [markdown]
# ### 0.7 How the groups will be compared
#
# The main question asks what further information adds to prediction. We therefore choose
# **incremental group addition**: build up the available information in complete groups and measure
# the gain when another group becomes available. This gives the later evaluation a direct connection
# to the research question. Leave-one-group-out is not the selected comparison for this workflow.
#
# The starting baseline and the order of additions will be specified in the modeling notebook
# before scores are compared. No baseline or addition schedule is constructed here. Because groups
# overlap, an improvement is conditional on the groups already present; it is not an order-independent
# importance ranking. All six groups must eventually be represented.
#
# The table below only records the raw membership of each main group. Derived features are specified
# after the detailed checks. A derivative becomes available only when every parent group is present;
# this applies equally to cross-group ratios and to missingness indicators.

# %%
# list the retention fields used only in the sensitivity comparison
retention_features = [
    'RetentionCalls',
    'RetentionOffersAccepted',
    'MadeCallToRetentionTeam',
]

# keep each group together and leave retention fields out of the main comparison
main_feature_groups = {}
for group_name, group_features in FEATURE_GROUPS.items():
    main_features = []
    for feature in group_features:
        if feature not in retention_features:
            main_features.append(feature)
    main_feature_groups[group_name] = main_features

# collect the raw predictors in the main and sensitivity representations
main_raw_features = []
sensitivity_raw_features = []
for group_name, group_features in FEATURE_GROUPS.items():
    sensitivity_raw_features.extend(group_features)
    main_raw_features.extend(main_feature_groups[group_name])

# count the raw predictors and their unique names
main_feature_count = len(main_raw_features)
unique_main_feature_count = len(set(main_raw_features))
sensitivity_feature_count = len(sensitivity_raw_features)
unique_sensitivity_feature_count = len(set(sensitivity_raw_features))

# check that no feature is duplicated and that the agreed counts are preserved
assert main_feature_count == 53
assert unique_main_feature_count == 53
assert sensitivity_feature_count == 56
assert unique_sensitivity_feature_count == 56

# verify that retention information is absent from the main representation
retention_in_main = set(main_raw_features).intersection(retention_features)
assert not retention_in_main

# show the membership of each group without defining an experiment order
group_summary = []
for group_name, group_features in main_feature_groups.items():
    group_summary.append({
        'group': group_name,
        'raw_predictors': len(group_features),
        'features': ', '.join(group_features),
    })
group_summary_table = pd.DataFrame(group_summary)
display(group_summary_table)

# %% [markdown]
# The raw membership is now fixed. The complete main set contains 53 predictors, while the
# separate retention representation contains 56. Later group additions will use these memberships;
# their starting point and ordering are modeling decisions.

# %% [markdown]
# ### 0.8 Keeping later comparisons consistent
#
# The proposal uses stratified five-fold cross-validation within the training split. The modeling
# notebook will implement the shared fold assignments and the baseline, then apply the same validation
# customers when groups are added. Imputation, encoding and scaling must be learned only from the
# training portion of each fold. The test set remains reserved for final evaluation.

# %% [markdown]
# ### 0.9 From the overall view to the individual groups
#
# Usage, service quality, demographics and handset information overlap. Some apparent relationships
# also change substantially when recorded unknown states are excluded. The next sections look more
# closely at each group so the final choices can preserve those distinctions rather than assume that
# every correlation describes customer behavior or that every zero means missing data.

# %% [markdown]
# ## 1. Account & billing
#
# The first group describes the customer relationship: spending, plan charges, time with the provider,
# equipment age and the number of lines. We look for patterns that could explain churn while checking
# whether similar-looking account measures really carry the same information.
#
# Columns: `MonthlyRevenue`, `TotalRecurringCharge`, `OverageMinutes`, `MonthsInService`, `CurrentEquipmentDays`, `Handsets`, `HandsetModels`, `UniqueSubs`, `ActiveSubs`, `AdjustmentsToCreditRating`

# %%
# select the account and billing features
cols = FEATURE_GROUPS["account_billing"]

# inspect data quality and each feature's individual churn association
group_overview = overview(train, cols)
display(group_overview)
group_churn_signal = churn_signal(train, cols)
display(group_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# compare the numerical features within the group
corr = corr_heatmap(train, cols)

# %% [markdown]
# ### EDA Findings — Account & billing
#
# The billing variables have a small amount of missing data: `MonthlyRevenue`, `TotalRecurringCharge` and `OverageMinutes` each have 120 missing values (0.29%). `Handsets`, `HandsetModels` and `CurrentEquipmentDays` each have one missing value; the remaining four columns have none. Revenue is negative for 3 customers (minimum -6.17), recurring charge for 8 (minimum -11), and equipment age for 55 (minimum -5). Negative equipment age is invalid, but negative billing values may represent credits or adjustments, so we retain those signed amounts in this workflow. `HandsetModels` never exceeds `Handsets`, and `ActiveSubs` never exceeds `UniqueSubs`.
#
# Revenue and overage are right-skewed: median `MonthlyRevenue` is 48.47, its 99th percentile is 228.33 and its maximum is 1,223.38. `OverageMinutes` is zero for 45.59% of customers, with a median of 3 and maximum of 4,321. `MonthsInService` ranges from 6 to 61 months (median 16), so the data cannot describe the first months of service. Handset and subscription counts generally have a median of 1, although they reach 24 handsets and 18 subscriptions; `AdjustmentsToCreditRating` is zero for 96.41%. Several measures overlap: Spearman correlations are 0.91 for handsets and handset models, 0.83 for unique and active subscriptions, and 0.76 for revenue and recurring charge. Tenure and equipment age are less closely related (0.42).
#
# Tenure has a clear descriptive relationship with churn: 16.8% at 6–10 months, 42.5% at 11–12 months, 30.7% at 13–24 months and 28.8% after that. Contract expiry is a possible explanation, but the data does not establish it. Equipment age is the strongest individual feature in this group (AUC 0.576): churn is 19.3% at 180–300 days, 39.3% at 300–365 days and 32–37% for older handsets. A desire for a new phone is a possible interpretation, not a demonstrated cause.
#
# Recurring charges also vary with churn: 33.7% in the lowest fifth versus 24.2% in the highest (AUC 0.457). Monthly revenue alone is nearly unrelated (AUC 0.485), except for 33.3% churn in its lowest tenth; overage has a weak association (AUC 0.511), rising from about 28% to about 32% in the highest group. Churn is 30.0% with one handset and 25.5% with four or more. The difference between unique and active subscriptions exceeds zero for 13.1% of customers; churn is 27.9% with no inactive line, 33.6% with one and 38.0% with two or more.
#
# We retain all ten raw predictors in the group, including the overlapping and sparse columns. We add `ContractWindow`, the 11–12 month flag, and `inactive_subs`, the difference between unique and active subscriptions. The window was selected after inspecting outcomes, so its apparent value may be optimistic. We also retain `RevenuePerMinute`, defined as `MonthlyRevenue / (MonthlyMinutes + 1)`, once both billing and usage groups are present. It is an exposure-normalized proxy, not a verified tariff. For preprocessing, negative equipment ages become missing; signed billing values remain. We impute genuine numeric missing values with training-fold medians and preserve missingness indicators. We do not clip or log-transform the skewed counts in the specified workflow.

# %% [markdown]
# ## 2. Service usage
#
# Billing tells us what customers pay; usage tells us how much they engage with the service.
# Here we look at overall activity, calling patterns and use of additional features. Sparse services
# are worth checking carefully, since a small group of users can produce an unstable churn rate.
#
# Columns: `MonthlyMinutes`, `DirectorAssistedCalls`, `RoamingCalls`, `ThreewayCalls`, `ReceivedCalls`, `OutboundCalls`, `InboundCalls`, `PeakCallsInOut`, `OffPeakCallsInOut`, `CallForwardingCalls`, `CallWaitingCalls`

# %%
# select the service usage features
cols = FEATURE_GROUPS["usage"]

# inspect data quality and individual churn associations
group_overview = overview(train, cols)
display(group_overview)
group_churn_signal = churn_signal(train, cols)
display(group_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# compare the numerical features within the group
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
# `MonthlyMinutes`, `DirectorAssistedCalls` and `RoamingCalls` each have 120 missing training values (0.29%); the other eight usage features are complete. Missing values should remain distinct from zero use. Usage is often sparse: 68.66% have no roaming calls, 72.72% no three-way calls and 99.55% no call-forwarding calls. Only 183 customers (0.45%) used forwarding, while 49.04% have zero call-waiting calls. Monthly usage is right-skewed: the highest quintile spans 841–7,359 minutes, so we preserve those observed extremes in the chosen representation.
#
# Churn is 34.28% in the lowest monthly-usage quintile (0–122 minutes) and 25.79% in the highest (above 841 minutes), a difference of 8.49 percentage points. Three-way users have 26.93% churn versus 29.53% for non-users, and call-waiting users have 27.44% versus 30.25%. Roaming users and non-users are similar at 28.71% and 28.79%. Forwarding users show 24.59% versus 28.84%, but that comparison is uncertain because only 183 customers used it. These are descriptive associations, not evidence of causation.
#
# We retain all eleven raw usage predictors, including roaming and call forwarding. Missing values are not zero use: the later preprocessing notebook will use training-fold medians and missingness indicators. We keep the measured extremes without clipping or logarithmic transformation. We do not add binary calling-use flags or call-volume ratios in this workflow. The chosen `RevenuePerMinute` and `RelChangeMinutes` measures combine this group with other groups and appear only once all of their parent variables are available.

# %% [markdown]
# ## 3. Usage change & retention
#
# A customer's current activity is only part of the picture. A change in that activity might reveal
# disengagement that the current level alone misses. This group also contains retention contacts,
# so we first check what the change columns mean and which information would be available before a prediction.
#
# Columns: `PercChangeMinutes`, `PercChangeRevenues`, `RetentionCalls`, `RetentionOffersAccepted`, `MadeCallToRetentionTeam`, `ReferralsMadeBySubscriber`, `NewCellphoneUser`, `NotNewCellphoneUser`

# %%
# select the usage change and retention features
cols = FEATURE_GROUPS["usage_change_retention"]

# inspect data quality and individual churn associations
group_overview = overview(train, cols)
display(group_overview)
group_churn_signal = churn_signal(train, cols)
display(group_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# compare the numerical features within the group
corr = corr_heatmap(train, cols)

# %% [markdown]
# ### 3.1 Are the "PercChange" columns really percentages?
#
# For a nonnegative quantity and a positive baseline, an ordinary percentage decrease cannot be below -100%. Values below that boundary make the documented units uncertain; the check does not establish the exact original formula. This checks how often that happens and whether the change scales with usage. The helper functions defined here are reused in section 7.

# %%
# import the tools used for the change and cross-group checks
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

# group normalized minutes change using the original analysis bands
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
# The call flag records the same information as whether `RetentionCalls` is above zero in this training sample. Retention contact is associated with churn, but its timing is unknown. We keep all three original retention fields together only in the separate sensitivity comparison; the observation does not establish leakage or a causal offer effect.

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
# The two flags describe three observed states: new indicated, not-new indicated, and neither indicated. About 67% fall into the last state, whose real-world meaning is unknown. We retain both raw flags in the main group and do not add a replacement `NewUserStatus` field.

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
# Referrals are uncommon and show little standalone association with churn. We retain the raw referral count in the main group because that descriptive result does not determine its contribution alongside other predictors.

# %% [markdown]
# ### EDA Findings — Usage Change & Retention
#
# Both `PercChange` columns are missing for the same 290 customers (0.71%), including all 120 customers with missing billing and usage values. Their units still need confirmation: 21.8% of `PercChangeMinutes` values are below -100, and its observed lower boundary is consistent with a signed absolute change. `PercChangeRevenues` has 466 values below -100. We will keep both signed variables without treating their names as proof of a percentage formula. Minutes change has a median of -6 and long tails from -3,406 to +5,192.
#
# The user-status flags are not complements: 67% of customers have “No” for both, and the real-world meaning of that state is unconfirmed. Referrals are zero for 95.2% of customers. Retention calls are rare (3.4% called and 1.7% accepted an offer), and contact is more common at longer tenures. `MadeCallToRetentionTeam` is “Yes” exactly when `RetentionCalls` is greater than zero in the training data. None of these cross-sectional observations establishes when contact occurred or why it happened.
#
# For the constructed normalized minutes-change measure, the lowest band has about 40.0% churn versus about 25% near zero; the relationship is U-shaped, which a univariate AUC alone cannot fully capture. Missing change values are associated with 56.9% churn versus 28.6% when observed. The normalized revenue-change bands show a weaker U-shape (30.9%, 27.9% and 30.6%). New-user status and referrals provide weak standalone evidence, and the new-user difference does not survive multiple-testing correction. Falling activity could indicate disengagement, but this snapshot is not a monthly forecasting experiment.
#
# Customers in retention-contact categories show 42.3–47.5% churn versus 28.2% without contact. Among contacted customers, the offer-acceptance association is not significant at 0.05 (p = 0.06); it does not estimate whether offers cause customers to stay. The prediction-time availability of these variables is uncertain, so all three are retained only in the separate retention sensitivity comparison.
#
# We retain all five non-retention raw predictors in the main group, including referrals and both user-status flags. We include `RelChangeMinutes` as `PercChangeMinutes / MonthlyMinutes`, only once both parent groups are present. A zero or missing denominator produces an undefined value rather than a fabricated zero; the raw values and their missingness indicators remain available. `ChangeMissing` is the missingness indicator for `PercChangeMinutes`, not a second copy of that indicator. We retain signed change values and do not clip them. The initial representation does not add `NewUserStatus` or `RetentionContact`.

# %% [markdown]
# ## 4. Service quality
#
# Next we turn to the customer's experience of using the service: dropped, blocked and unanswered calls,
# along with calls to customer care. These counts also depend on how much someone uses the service,
# so their overlap with the usage group will matter when we compare groups later.
#
# Columns: `DroppedCalls`, `BlockedCalls`, `UnansweredCalls`, `DroppedBlockedCalls`, `CustomerCareCalls`

# %%
# select the service quality features
cols = FEATURE_GROUPS["service_quality"]

# inspect data quality and individual churn associations
group_overview = overview(train, cols)
display(group_overview)
group_churn_signal = churn_signal(train, cols)
display(group_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# compare the numerical features within the group
corr = corr_heatmap(train, cols)

# %% [markdown]
# ### EDA Findings — Service Quality
#
# All five service-quality counts are complete and nonnegative in the training data. Their distributions are strongly right-skewed: most customers have relatively few recorded events, while the maxima are much larger than the 99th percentiles. `CustomerCareCalls` is zero for 54.89% of customers. The specified workflow preserves those observed tails without clipping or logarithmic transformation.
#
# Individually, the five features show weak associations with churn, with AUCs from 0.470 to 0.491. Customers with more care calls tend to have slightly lower observed churn, but that pattern alone does not establish a protective effect. The counts also overlap: `DroppedCalls` and `DroppedBlockedCalls` have a Spearman correlation of 0.88. Calling activity may help explain some of the relationships, so the cross-group checks examine exposure as well.
#
# We retain all five raw predictors. `DroppedBlockedCalls` is not an exact sum of the separate counts, and correlation alone does not establish what this group contributes. The chosen representation adds no service-quality rates or binary care-call indicator.

# %% [markdown]
# ## 5. Demographics & region
#
# Customer profiles provide a different view from bills and calling activity. Before comparing this
# group with the behavioral groups, we look at its recorded categories and the many zero or Unknown
# values. Those states may describe incomplete records, household structure, or a mixture of both.
#
# Columns: `AgeHH1`, `AgeHH2`, `ChildrenInHH`, `IncomeGroup`, `Occupation`, `MaritalStatus`, `Homeownership`, `CreditRating`, `PrizmCode`, `ServiceArea`

# %%
# select the demographic and region features
cols = FEATURE_GROUPS["demographics_region"]

# inspect data quality and individual churn associations
group_overview = overview(train, cols)
display(group_overview)
group_churn_signal = churn_signal(train, cols)
display(group_churn_signal)

# show the distribution and churn pattern for each feature
for feature in cols:
    plot_column(train, feature)

# compare the numerical features within the group
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
# The individual age and income AUCs are close to 0.5, although churn falls from 30.8% among customers with known age at most 30 to 25.6% above 60. Recorded unknown states generally have churn rates about one to three percentage points higher than their comparison groups. `CreditRating` varies non-monotonically: rates range from 21.8% for “5-Low” to about 31% for several higher ratings. Rural `PrizmCode` has 31.9% churn among 1,916 customers, versus 28.1% for suburban customers. `ChildrenInHH` differs by 0.8 percentage points; the highest occupation rate comes from only 135 homemakers. These are descriptive patterns, not proof of a group's incremental value.
#
# We retain all ten raw demographic and region predictors, including `AgeHH2`, `Occupation` and `ChildrenInHH`. We preserve recorded age zeros and include `age1_zero` and `age2_zero` as distinct flags; actual missing ages are median-imputed within training folds with missingness indicators. Income code 0 and recorded “Unknown” categories remain distinct from the `__MISSING__` category used for actual categorical missing values. Nonbinary categories, including `IncomeGroup`, `CreditRating` and `ServiceArea`, use one-hot encoding with unseen categories ignored. This workflow does not use prefix grouping, target encoding or a combined unknown indicator.

# %% [markdown]
# ## 6. Handset & lifestyle
#
# Handset characteristics may help explain why customers with similar usage have different churn
# patterns. The lifestyle variables add some wider context, although several are rare or closely
# related. We check the device patterns first, then look at those overlaps.
#
# Columns: `HandsetPrice`, `HandsetRefurbished`, `HandsetWebCapable`, `TruckOwner`, `RVOwner`, `OwnsMotorcycle`, `OwnsComputer`, `HasCreditCard`, `BuysViaMailOrder`, `RespondsToMailOffers`, `OptOutMailings`, `NonUSTravel`

# %%
# select the handset and lifestyle features
cols = FEATURE_GROUPS["handset_lifestyle"]

# inspect data quality and individual churn associations
group_overview = overview(train, cols)
display(group_overview)
group_churn_signal = churn_signal(train, cols)
display(group_churn_signal)

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
    feature_groups = train.groupby(feature, dropna=False)["churn"]
    summary = feature_groups.agg(
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
# Observed churn is 37.20% for customers without web-capable handsets and 27.90% for those with them. It is 32.35% for refurbished handsets versus 28.26% otherwise. Mail-order buyers and offer respondents also have slightly lower observed churn than their counterparts, around 27.4% versus 29.6%. These associations do not establish causes or show how much the features add once other groups are included.
#
# We keep all twelve raw handset and lifestyle predictors, including both mail and vehicle indicators. The later preprocessing notebook maps yes/no values to 0/1 and one-hot encodes `HandsetPrice`, preserving “Unknown” separately from actual missing values. We do not include handset-usage interactions or remove an overlapping indicator in this specified representation.

# %% [markdown]
# ## 7. Cross-group checks
#
# The individual groups show useful patterns, but customers do not fit into one group at a time.
# We now bring the groups together: does an old handset matter beyond tenure, do quality counts mainly
# reflect usage, and does spending mean something different when activity is falling? These checks
# help us decide which relationships to carry into the later comparison.

# %% [markdown]
# ### 7.1 Are the related columns interchangeable?
#
# Before settling the representation, we check whether apparent redundancies are exact. Similar
# values alone would not justify deleting information from a group.

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
# These checks show overlap rather than exact interchangeability. The quality total differs from the sum in some records, both user flags can be No, and handset and subscription counts are related without being identical. We retain all of these raw features in their groups.

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
# Recorded zero and “Unknown” states have different churn rates from other values. We retain them as observed, distinct from actual missing values. The specified age-zero flags preserve the two age codes; no combined demographic-unknown indicator is added.

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
# The peak around months 11–12 motivates `ContractWindow`, which is included with the account and billing group. Contract expiry is a possible explanation, but there is no contract field to confirm it. Because the window was selected after observing churn, its apparent value may be optimistic.

# %% [markdown]
# ### 7.4 Tenure × spending segments
# These segments will be used later for the error analysis (which groups contain the most missed churners).

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
# The 11–12-month peak appears across spending levels, at roughly 42–43% churn. There is little support here for an additional high-spending effect in that window. The account group retains the underlying tenure and revenue variables, and includes the chosen `ContractWindow` flag.

# %% [markdown]
# ### 7.5 Strongest correlations between groups
#
# Spearman correlation between numeric features that belong to different groups. This and the following checks use the helper functions from 3.1.

# %%
# record which feature group owns each raw column
group_of = {}
for group_name, group_columns in FEATURE_GROUPS.items():
    for column in group_columns:
        group_of[column] = group_name

# select columns stored as numbers for this correlation check
num_cols = []
for column in group_of:
    is_numeric_column = pd.api.types.is_numeric_dtype(train[column])
    if is_numeric_column:
        num_cols.append(column)

# calculate rank correlations between the numeric columns
numeric_group_data = train[num_cols]
spearman = numeric_group_data.corr(method="spearman")

# inspect each unique pair and keep strong relationships across groups
cross_group_rows = []
for first_position, first_feature in enumerate(num_cols):
    remaining_features = num_cols[first_position + 1:]
    for second_feature in remaining_features:
        first_group = group_of[first_feature]
        second_group = group_of[second_feature]
        correlation = spearman.loc[first_feature, second_feature]

        same_group = first_group == second_group
        strong_correlation = abs(correlation) >= 0.5
        if same_group or not strong_correlation:
            continue

        cross_group_rows.append({
            "feature_1": first_feature,
            "group_1": first_group,
            "feature_2": second_feature,
            "group_2": second_group,
            "spearman": round(correlation, 2),
        })

# sort by correlation strength so negative and positive relationships both stand out
cross_pair_columns = ["feature_1", "group_1", "feature_2", "group_2", "spearman"]
cross_pairs = pd.DataFrame(cross_group_rows, columns=cross_pair_columns)
cross_pairs = cross_pairs.sort_values("spearman", key=abs, ascending=False)
cross_pairs = cross_pairs.reset_index(drop=True)

# summarize how many strong relationships connect each pair of groups
group_pair_counts = cross_pairs.groupby(["group_1", "group_2"]).size()
group_pair_counts = group_pair_counts.to_frame("n_pairs")
print("Cross-group pairs with |Spearman| >= 0.5:", len(cross_pairs))
display(group_pair_counts)

# show the strongest individual feature pairs
display(cross_pairs.head(12))

# %% [markdown]
# Much of the overlap connects usage with service quality. Incremental group comparisons will therefore measure the added value conditional on the groups already present. A later modeling stage will specify the baseline, ordering and model.

# %% [markdown]
# ### 7.6 Usage drop x spending level
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
# The low normalized-change band has higher churn across spending levels. We include `RelChangeMinutes` as a constructed association feature when both usage-change and usage parents are present; its units are not a verified original percentage formula. Zero and missing denominators remain undefined.

# %% [markdown]
# ### 7.7 Handset age within tenure
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
# ### 7.8 Service-quality counts versus rates
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
# Normalizing by activity changes some association directions, suggesting that raw counts partly reflect exposure. We retain all five quality counts and do not include service-quality rates in the specified representation.

# %% [markdown]
# ### 7.9 Are customer-care callers just heavy users?
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
# Care callers have lower observed churn within each broad usage band as well. This stratification does not establish a causal effect. `CustomerCareCalls` remains in the main group without an additional binary indicator.

# %% [markdown]
# ### 7.10 Revenue relative to usage and bill structure
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
# The revenue-to-usage proxy separates churn more clearly than revenue alone: observed churn rises from 24.9% to 34.7% across its fifths. We include `RevenuePerMinute` as revenue divided by minutes plus one only once both parent groups are present. It is a proxy, not a literal tariff.

# %% [markdown]
# ### 7.11 Do unknown values cluster in the same customers?
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
# Several demographic states cluster in the same records, while unknown handset price behaves differently. We preserve each raw state and the separate age-zero flags. We do not add a combined unknown indicator in the specified representation.

# %% [markdown]
# ### 7.12 Reviewing the engineered proposals
#
# The table compares univariate signal for several proposed derivatives with their parent variables.
# A numeric AUC mainly summarizes a ranking and can miss U-shaped patterns. We use these screens
# together with the earlier model notes and the raw-data checks to choose the final representation,
# which is recorded in section 8. A diagnostic calculation here does not mean a proposal is included.

# %%
# count the four demographic states used in the candidate screen
demographic_unknown_columns = ["AgeHH1", "IncomeGroup", "MaritalStatus", "Homeownership"]
demo_unknown = unknown_flags[demographic_unknown_columns].sum(axis=1)

# calculate the other candidate values without changing the training frame
contract_window = train["MonthsInService"].between(11, 12)
change_missing_values = train["PercChangeMinutes"].isna()
drop_block_per_minute = train["DroppedBlockedCalls"] / (train["MonthlyMinutes"] + 1)

# record the meaning and dependencies of each candidate using named fields
candidates = [
    {
        "feature": 'RelChangeMinutes',
        "group": 'usage_change_retention',
        "formula": 'PercChangeMinutes / MonthlyMinutes',
        "values": rel_change_min,
        "parent": 'PercChangeMinutes',
        "other_group": 'usage',
    },
    {
        "feature": 'ChangeMissing',
        "group": 'usage_change_retention',
        "formula": 'PercChangeMinutes is missing',
        "values": change_missing_values,
        "parent": None,
        "other_group": '',
    },
    {
        "feature": 'RevenuePerMinute',
        "group": 'account_billing',
        "formula": 'MonthlyRevenue / (MonthlyMinutes + 1)',
        "values": rev_per_min,
        "parent": 'MonthlyMinutes',
        "other_group": 'usage',
    },
    {
        "feature": 'RecurringShare',
        "group": 'account_billing',
        "formula": 'TotalRecurringCharge / MonthlyRevenue',
        "values": recurring_share,
        "parent": 'TotalRecurringCharge',
        "other_group": '',
    },
    {
        "feature": 'OriginalHandset',
        "group": 'account_billing',
        "formula": 'CurrentEquipmentDays >= 0.9 x tenure in days',
        "values": original_handset,
        "parent": 'CurrentEquipmentDays',
        "other_group": '',
    },
    {
        "feature": 'ContractWindow',
        "group": 'account_billing',
        "formula": 'MonthsInService in 11-12',
        "values": contract_window,
        "parent": 'MonthsInService',
        "other_group": '',
    },
    {
        "feature": 'DropBlockPerMinute',
        "group": 'service_quality',
        "formula": 'DroppedBlockedCalls / (MonthlyMinutes + 1)',
        "values": drop_block_per_minute,
        "parent": 'DroppedBlockedCalls',
        "other_group": 'usage',
    },
    {
        "feature": 'DemoUnknownCount',
        "group": 'demographics_region',
        "formula": 'number of unknowns in AgeHH1, IncomeGroup, MaritalStatus, Homeownership',
        "values": demo_unknown,
        "parent": None,
        "other_group": '',
    },
]

# calculate a descriptive score for each candidate and its named parent
rows = []
for candidate in candidates:
    name = candidate["feature"]
    group = candidate["group"]
    formula = candidate["formula"]
    values = candidate["values"]
    parent = candidate["parent"]
    other = candidate["other_group"]
    candidate_auc = auc_of(values)
    candidate_signal = round(abs(candidate_auc - 0.5), 3)

    # leave parent comparisons empty when no single parent is specified
    parent_auc = None
    parent_signal = None
    if parent:
        parent_auc = auc_of(train[parent])
        parent_signal = round(abs(parent_auc - 0.5), 3)

    # use a clear display label for candidates without a single parent
    parent_label = "-"
    if parent:
        parent_label = parent

    # show whether the formula also depends on another feature group
    additional_group_label = "-"
    if other:
        additional_group_label = other

    rows.append({
        "feature": name,
        "group": group,
        "formula": formula,
        "auc": candidate_auc,
        "signal": candidate_signal,
        "parent": parent_label,
        "parent_signal": parent_signal,
        "also_uses_group": additional_group_label,
    })

# display the candidate screen with feature names as row labels
candidate_summary = pd.DataFrame(rows)
candidate_summary = candidate_summary.set_index("feature")
display(candidate_summary)

# %% [markdown]
# The screens support a fixed initial representation: `ContractWindow`, `inactive_subs`, `RevenuePerMinute`, `RelChangeMinutes`, `ChangeMissing`, `age1_zero` and `age2_zero`. The earlier screens are descriptive and historical; they do not establish the independent contribution of each engineered field. Other engineered proposals are excluded from this workflow.

# %% [markdown]
# ### 7.13 What the earlier model checks suggest
#
# The earlier version of this analysis included a quick model check using simplified preprocessing
# and leaving out ServiceArea. Adding six candidates together reportedly increased Average Precision
# from about 0.375 to 0.417 for logistic regression and from 0.451 to 0.464 for gradient boosting.
# The strongest reported individual additions were ContractWindow (+0.026 AP) and RevenuePerMinute
# (+0.011) for logistic regression, RelChangeMinutes (+0.012) for boosting, and a smaller ChangeMissing
# gain (+0.006) for logistic regression.
#
# Those results help explain why we include the tenure window, normalized change, spending-to-activity
# proxy and change missingness in the final specification. The later evaluation will estimate performance
# using consistent preprocessing and training folds. The tenure window was also selected after seeing the churn pattern, so its apparent gain may
# be optimistic. We keep these figures as earlier exploratory evidence; they are not rerun here or
# presented as the final group-comparison results. Preprocessing and later model fitting will follow separately.

# %% [markdown]
# ### 7.14 Which test results survive multiple-testing correction?
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
# After correction for multiple tests, some small differences no longer stand out. This tempers interpretation of individual patterns; it does not determine the contribution of a feature group. We retain the full raw group representation for the later incremental comparison.

# %% [markdown]
# ### EDA Findings — Cross-group relationships
#
# The groups overlap in ways that matter for interpreting their contribution. Of the 35 cross-group numerical pairs with absolute Spearman correlation at least 0.5, 25 link usage and service quality. Several recorded demographic unknown states also cluster: four indicators correlate between 0.62 and 0.81, and `Occupation` is “Other” for 99.0% of customers with `AgeHH1` equal to zero. `HandsetPrice` “Unknown” does not show the same relationship. The two cellphone-user flags are not complements, and `DroppedBlockedCalls` is close to, but not always equal to, the sum of its components. Overlap is a reason to compare groups, not sufficient evidence to delete variables before the comparison.
#
# Looking at combinations gives a more detailed picture than single variables. Churn in the normalized usage-change band below −0.2 is about 34% across spending levels, compared with 23–26% in the central band. Among customers with more than 24 months of tenure, churn is 34.6% with a handset older than one year versus 19.8% with a newer handset. Tenure and equipment age are related (Spearman 0.42) but are not interchangeable; the age-to-tenure heuristic does not prove whether a customer still has their original handset. Within each usage level, care callers show lower observed churn than non-callers, although this comparison does not establish causation.
#
# The revenue divided by `MonthlyMinutes + 1` proxy also varies with churn: rates rise from 24.9% to 34.7% across its fifths, with an AUC of 0.549 compared with 0.485 for revenue and 0.461 for minutes. It is an exposure-normalized proxy rather than a verified price per minute. Likewise, quality counts and per-minute measures have different but individually weak associations; exposure is a plausible explanation, not proof that one measure is correct. For tenure and spending, the 11–12 month group shows 42–43% churn at every spending level, so the data do not support an extra risk for high spenders in that window.
#
# The chosen representation includes `ContractWindow`, `inactive_subs`, `RevenuePerMinute`, `RelChangeMinutes`, `ChangeMissing`, `age1_zero` and `age2_zero`. `ChangeMissing` is the numeric missingness indicator for the original minutes-change field, not a duplicate feature. Earlier exploratory model evidence helps explain some choices, but it is not a final performance estimate; in particular, the tenure window was selected after inspecting outcomes. We do not include the other engineered proposals in this workflow.
#
# We will compare groups by adding them incrementally. `RevenuePerMinute` and `RelChangeMinutes` enter only when all their parent groups are present; any measured increase is conditional on the order and on the groups already included. The baseline, ordering and model specification belong to the later modeling stage. The main analysis retains all 53 raw non-retention predictors, and all three retention variables are reserved for the separate sensitivity comparison.

# %% [markdown]
# ## 8. Final feature and preprocessing decisions
#
# We retain 53 non-retention raw predictors in the main reference and reserve the three retention predictors for a separate sensitivity comparison. Inclusion preserves each group for comparison; it does not establish individual importance.
#
# We exclude `CustomerID` from predictors because it is an identifier, and `Churn` / `churn` because they are the target. Split and fold labels are evaluation metadata, not predictors.
#
# In the tables, **numeric**, **binary** and **nominal** refer to the preprocessing rules below. Special handling is shown alongside the affected features.

# %% [markdown]
# ### Raw features

# %% [markdown]
# #### Account and billing
#
# | Feature | Decision | Reason | Handling |
# | --- | --- | --- | --- |
# | `MonthlyRevenue` | include | Monthly spending has weak individual AUC (0.485), but defines spending segments and may help in combination. | numeric; retain signed values |
# | `TotalRecurringCharge` | include | Recurring charges distinguish churn across lowest and highest fifths (33.7% versus 24.2%) in exploratory analysis. | numeric; retain signed values |
# | `OverageMinutes` | include | Overage is zero for 45.59% of customers; unusually high values may capture plan use despite weak individual AUC. | numeric |
# | `MonthsInService` | include | Tenure is associated with a churn peak around months 11–12; the cause is not known. | numeric |
# | `CurrentEquipmentDays` | include | Equipment age has an exploratory AUC of 0.576; negative ages are invalid. | numeric; negatives become missing |
# | `Handsets` | include | Handset count reflects equipment history and overlaps with HandsetModels, without establishing redundancy. | numeric |
# | `HandsetModels` | include | Model count describes equipment variety; its correlation with Handsets does not establish no added value. | numeric |
# | `UniqueSubs` | include | Total subscription count can reflect account structure and pairs with ActiveSubs. | numeric |
# | `ActiveSubs` | include | Active subscription count describes account structure and helps calculate inactive subscriptions. | numeric |
# | `AdjustmentsToCreditRating` | include | Credit adjustments are rare (96.41% zero); weak individual AUC does not rule out interactions. | numeric |

# %% [markdown]
# #### Service usage
#
# | Feature | Decision | Reason | Handling |
# | --- | --- | --- | --- |
# | `MonthlyMinutes` | include | Monthly usage differs in churn across lowest and highest quintiles (34.28% versus 25.79%). | numeric |
# | `DirectorAssistedCalls` | include | Directory-assisted call volume may characterize a distinct aspect of calling behavior. | numeric |
# | `RoamingCalls` | include | Roaming is zero for 68.66%; user/non-user churn is similar, but continuous values may still contribute. | numeric |
# | `ThreewayCalls` | include | Three-way use is sparse; users and non-users have exploratory churn of 26.93% and 29.53%. | numeric |
# | `ReceivedCalls` | include | Received call volume captures use and overlaps with other calling measures without proving redundancy. | numeric |
# | `OutboundCalls` | include | Outgoing call volume captures calling behavior and overlaps with other usage measures. | numeric |
# | `InboundCalls` | include | Incoming call volume describes customer activity and may complement outgoing calls. | numeric |
# | `PeakCallsInOut` | include | Peak-hour activity describes when customers use the service and complements off-peak activity. | numeric |
# | `OffPeakCallsInOut` | include | Off-peak activity describes when customers use the service and complements peak-hour activity. | numeric |
# | `CallForwardingCalls` | include | Only 183 customers used forwarding; its low frequency makes its descriptive churn rate uncertain. | numeric |
# | `CallWaitingCalls` | include | Call-waiting is zero for 49.04%; users and non-users show exploratory churn of 27.44% and 30.25%. | numeric |

# %% [markdown]
# #### Usage change and retention
#
# | Feature | Decision | Reason | Handling |
# | --- | --- | --- | --- |
# | `PercChangeMinutes` | include | Change in minutes has a nonlinear relationship with churn and appears to measure absolute change rather than a percentage. | numeric; use `ChangeMissing` once |
# | `PercChangeRevenues` | include | Revenue change is weak and U-shaped in exploration; it may add information alongside other features. | numeric; own missingness indicator |
# | `RetentionCalls` | sensitivity only | Retention contact is associated with churn, but its timing relative to prediction is uncertain. | numeric |
# | `RetentionOffersAccepted` | sensitivity only | Accepted offers describe retention intervention, but their timing relative to prediction is uncertain. | numeric |
# | `MadeCallToRetentionTeam` | sensitivity only | This flag matches whether RetentionCalls is positive in the training data; timing is uncertain. | binary |
# | `ReferralsMadeBySubscriber` | include | Referrals are 95.2% zero; weak marginal evidence does not establish no group contribution. | numeric |
# | `NewCellphoneUser` | include | New and not-new flags are not complements; many customers have No for both. | binary |
# | `NotNewCellphoneUser` | include | This flag and NewCellphoneUser are not complements; both can be No. | binary |

# %% [markdown]
# #### Service quality
#
# | Feature | Decision | Reason | Handling |
# | --- | --- | --- | --- |
# | `DroppedCalls` | include | Dropped-call counts describe service interruptions and overlap with the combined failure count. | numeric |
# | `BlockedCalls` | include | Blocked attempts describe failed connections and overlap with the combined failure count. | numeric |
# | `UnansweredCalls` | include | Unanswered calls relate to usage volume and can contribute information about service patterns. | numeric |
# | `DroppedBlockedCalls` | include | Combined failures are close to, but not always equal to, dropped plus blocked calls; differences can be large. | numeric |
# | `CustomerCareCalls` | include | Support calls are zero for 54.89%; observed associations do not establish a causal effect. | numeric |

# %% [markdown]
# #### Demographics and region
#
# | Feature | Decision | Reason | Handling |
# | --- | --- | --- | --- |
# | `AgeHH1` | include | Primary household age includes a zero code of uncertain meaning and explicit missing observations. | numeric; retain zero and add `age1_zero` |
# | `AgeHH2` | include | Second household age has many zero codes that could represent no second adult; explicit missing values also occur. | numeric; retain zero and add `age2_zero` |
# | `ChildrenInHH` | include | Household children status has a small marginal churn difference; its conditional contribution remains untested. | binary |
# | `IncomeGroup` | include | IncomeGroup 0 has undocumented meaning and overlaps with other unknown-like codes; retain it as a category. | nominal; preserve code 0 |
# | `Occupation` | include | Occupation is Other for 73.8% of customers; this recorded category does not necessarily mean missing. | nominal; preserve recorded Other |
# | `MaritalStatus` | include | Unknown marital status is common (38.57%) and should remain an explicit recorded category. | nominal; preserve recorded Unknown |
# | `Homeownership` | include | The observed states are Known and Unknown; they are not a verified owner/renter distinction. Unknown covers 33.52% of customers. | nominal; preserve recorded Unknown |
# | `CreditRating` | include | Exploratory churn varies across credit categories non-monotonically; avoid assuming equal category spacing. | nominal |
# | `PrizmCode` | include | Market segment categories differ descriptively in churn, though this does not establish independent importance. | nominal |
# | `ServiceArea` | include | 729 observed areas include many rare categories, requiring careful encoding and validation. | nominal; preserve areas; 21 training values missing |

# %% [markdown]
# #### Handset and lifestyle
#
# | Feature | Decision | Reason | Handling |
# | --- | --- | --- | --- |
# | `HandsetPrice` | include | About 57% of handset prices are Unknown; preserve the recorded category rather than inventing a price. | nominal; preserve recorded Unknown |
# | `HandsetRefurbished` | include | Refurbished status shows descriptive churn of 32.35% versus 28.26%; this is not causal evidence. | binary |
# | `HandsetWebCapable` | include | Web-capable status shows descriptive churn of 27.90% versus 37.20%; this is not causal evidence. | binary |
# | `TruckOwner` | include | Truck ownership and RV ownership are nested but not identical, so one cannot automatically replace the other. | binary |
# | `RVOwner` | include | RV owners are a subset of truck owners in the training data, but not all truck owners own an RV. | binary |
# | `OwnsMotorcycle` | include | Motorcycle ownership may capture lifestyle differences; weak individual signal does not establish group irrelevance. | binary |
# | `OwnsComputer` | include | Computer ownership may capture lifestyle differences; its independent predictive value remains to be tested. | binary |
# | `HasCreditCard` | include | Credit-card ownership describes financial profile and may interact with credit or demographic characteristics. | binary |
# | `BuysViaMailOrder` | include | Mail-order buying agrees with offer response in 98.49% of cases, but the two flags are not identical. | binary |
# | `RespondsToMailOffers` | include | Response to mail offers agrees with mail-order buying in 98.49% of cases, but the two flags are not identical. | binary |
# | `OptOutMailings` | include | Mail opt-out records marketing preference; whether it adds predictive information requires validation. | binary |
# | `NonUSTravel` | include | Travel status may capture differences in customer behavior; its independent value requires validation. | binary |

# %% [markdown]
# ### Engineered features
#
# We include the following seven features with their corresponding groups. A derivative is available only when all of its parent groups are present. `ChangeMissing` is the existing missingness indicator for `PercChangeMinutes`, counted once.
#
# | Feature | Parents | Definition | Decision and reason |
# | --- | --- | --- | --- |
# | `ContractWindow` | `MonthsInService` | 1 if 11 <= MonthsInService <= 12, otherwise 0; missing parent remains missing | Include. Encode the observed nonlinear tenure window; earlier logistic check reported +0.026 AP. The outcome-chosen window may overstate later CV gains. |
# | `inactive_subs` | `UniqueSubs`, `ActiveSubs` | UniqueSubs - ActiveSubs; missing parent leaves result missing | Include. Observed churn increases from 27.9% with no inactive line to 38.0% with at least two. |
# | `RevenuePerMinute` | `MonthlyRevenue`, `MonthlyMinutes` | MonthlyRevenue / (MonthlyMinutes + 1); missing input leaves result missing | Include. Spending-to-activity proxy: churn rises from 24.9% to 34.7% across fifths; earlier logistic check reported +0.011 AP. This is not a literal tariff. |
# | `RelChangeMinutes` | `PercChangeMinutes`, `MonthlyMinutes` | PercChangeMinutes / MonthlyMinutes; denominator <= 0 or missing input gives missing | Include. Normalized change has clearer descriptive separation and earlier boosting check reported +0.012 AP. Original units are unconfirmed. |
# | `ChangeMissing` | `PercChangeMinutes` | 1 when PercChangeMinutes is missing before imputation, otherwise 0; alias of its generic missingness indicator | Include. Observed churn is 56.9% with missing changes versus 28.6% when observed. Generate this indicator once, not twice. |
# | `age1_zero` | `AgeHH1` | 1 when the original AgeHH1 value is 0, otherwise 0; actual missingness has its own indicator | Include. Preserve the distinct recorded zero state without inventing a normal age for it. |
# | `age2_zero` | `AgeHH2` | 1 when the original AgeHH2 value is 0, otherwise 0; actual missingness has its own indicator | Include. Zero may mean no second household adult; preserve that state without asserting it is missing. |
#
# We do not construct the following candidates in the specified workflow; their raw parent features remain available under the main or sensitivity reference.
#
# | Candidate | Parents | Reason |
# | --- | --- | --- |
# | `equipment_over_300` | `CurrentEquipmentDays` | Keep continuous equipment age. A second outcome-chosen threshold has no demonstrated additional model benefit in the supplied analysis. |
# | `RecurringShare` | `TotalRecurringCharge`, `MonthlyRevenue` | Earlier screen was weaker than the parent variables; retain both raw billing amounts without this extra ratio. |
# | `OriginalHandset` | `CurrentEquipmentDays`, `MonthsInService` | The age-to-tenure heuristic cannot establish actual handset history and the earlier screen was weaker than its parents. |
# | `RoamingUsed` | `RoamingCalls` | Any-use churn rates are nearly identical; retain the full roaming measurement without another thresholded copy. |
# | `ThreewayUsed` | `ThreewayCalls` | Retain the raw three-way measurement; no added benefit of this thresholded copy is demonstrated. |
# | `CallWaitingUsed` | `CallWaitingCalls` | Retain the raw call-waiting measurement; no added benefit of this thresholded copy is demonstrated. |
# | `PeakToTotal` | `PeakCallsInOut`, `OffPeakCallsInOut` | No incremental benefit is demonstrated; retain both raw peak and off-peak activity measures. |
# | `InboundToOutbound` | `InboundCalls`, `OutboundCalls` | No incremental benefit is demonstrated; retain both raw directional measures and avoid another near-zero-denominator ratio. |
# | `PeakToOffPeak` | `PeakCallsInOut`, `OffPeakCallsInOut` | No incremental benefit is demonstrated; retain the original peak/off-peak measurements. |
# | `NewUserStatus` | `NewCellphoneUser`, `NotNewCellphoneUser` | The two retained raw flags already preserve all three observed states, including neither indicated. |
# | `RetentionContact` | `RetentionCalls`, `RetentionOffersAccepted` | Keep the three original retention fields together in the separate sensitivity representation; do not add a collapsed duplicate. |
# | `drop_rate` | `DroppedCalls`, `PeakCallsInOut`, `OffPeakCallsInOut` | Exposure-normalized service measures remain weak and no added model benefit is shown; retain the raw count and activity information. |
# | `block_rate` | `BlockedCalls`, `PeakCallsInOut`, `OffPeakCallsInOut` | Exposure-normalized service measures remain weak and no added model benefit is shown; retain the raw count and activity information. |
# | `unanswered_rate` | `UnansweredCalls`, `PeakCallsInOut`, `OffPeakCallsInOut` | Exposure-normalized service measures remain weak and no added model benefit is shown; unanswered calls are not confirmed network failures. |
# | `HasCustomerCareCalls` | `CustomerCareCalls` | Retain the raw customer-care count; an additional binary version has no demonstrated incremental benefit. |
# | `DropBlockPerMinute` | `DroppedBlockedCalls`, `MonthlyMinutes` | Earlier exploratory model notes found little added value; retain its raw parents. |
# | `DemoUnknownCount` | `AgeHH1`, `IncomeGroup`, `MaritalStatus`, `Homeownership` | Earlier exploratory model notes found little added value; aggregation also loses which individual state is recorded. |
# | `CombinedUnknownIndicator` | `AgeHH1`, `IncomeGroup`, `MaritalStatus`, `Homeownership`, `Occupation` | Preserve the individual recorded states. Combining them loses detail and Other occupation is not confirmed missing. |
# | `ServiceAreaPrefix` | `ServiceArea` | The geographic meaning of the prefix is unverified; retain and one-hot encode the original area. |
# | `WebCapabilityUsageInteraction` | `HandsetWebCapable`, `MonthlyMinutes` | No incremental benefit is demonstrated; retain the raw handset and usage measurements. |
# | `Call_Failure_Rate` | `DroppedCalls`, `BlockedCalls`, `MonthlyMinutes` | No demonstrated added benefit, and minutes are not a verified denominator for call-failure probability; retain the raw parents. |

# %% [markdown]
# ### Preprocessing decisions
#
# | Topic | Decision |
# | --- | --- |
# | Comparison | Incrementally add feature groups; interpret each contribution in the context of its chosen order. |
# | Baseline and order | Defined in the modeling notebook before comparing model scores; not specified here. |
# | Raw membership | 53 non-retention raw predictors included; the 3 retention predictors are sensitivity-only. |
# | Missing markers | `NA`, `empty` |
# | Invalid equipment age | Negative CurrentEquipmentDays becomes missing before calculating any derivative. |
# | Signed amounts | Retain signed billing and change values as recorded; their exact source semantics remain unconfirmed. |
# | Numeric missing values | Training-fold median imputation plus per-column missingness indicators where missing values occur; no parameters learned from validation or test rows. |
# | ChangeMissing | Use the PercChangeMinutes missingness indicator under this name once; do not append a duplicate. PercChangeRevenues retains its own indicator. |
# | Age zeros | Retain recorded AgeHH1/AgeHH2 zeros and add age1_zero/age2_zero before imputation. Genuine missing ages remain distinct. |
# | Categorical missing values | Use __MISSING__ for genuine nonbinary categorical missing values; preserve recorded Unknown/Other and IncomeGroup=0 separately. Binary missingness follows the separate binary rule. |
# | Binary encoding | Yes=1, No=0. If binary values are missing, use training-fold mode imputation with a missingness indicator; reject unexpected nonmissing labels. |
# | Nominal encoding | One-hot encode nonbinary categories, explicitly including IncomeGroup, CreditRating, HandsetPrice and ServiceArea; handle_unknown=ignore, sparse where the model supports it. |
# | Service area | Keep original areas. No target encoding, prefix grouping or rare-level pooling. |
# | Extremes and transformations | No row deletion for extremes, no numerical clipping and no log/power transformation. Plot-only clipping does not change model inputs. |
# | Linear models | Training-fold StandardScaler on continuous numerical columns and numeric derived values; leave binary indicators and one-hot columns as encoded. |
# | Tree models | Use unscaled numerical values after the same validity and missing-value handling. |
# | Derived values | Compute selected derivatives before imputation; preserve undefined/missing states and impute inside folds with indicators. Do not turn undefined ratios into zero. |
# | Group dependencies | A derivative or missingness indicator is available only when all of its raw parent groups are present. |
# | Evaluation boundary | The modeling stage implements shared stratified five-fold training CV, model selection, test evaluation and the baseline; no training or fitted preprocessing is executed in this notebook. |
