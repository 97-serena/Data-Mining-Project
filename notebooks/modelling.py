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
# # Modelling
#
# This chapter evaluates models for telecom customer churn and examines the predictive
# value of different customer information. We establish baseline scores, compare
# preprocessing choices and five supervised models, and analyse missed churners across
# tenure and spending groups.
#
# The analysis uses the predefined training partition and five shared validation folds.
# All reported results describe model development. The test partition remains reserved
# for an independent evaluation after the modelling choices have been fixed.

# %% [markdown]
# ## 1. Analysis setup
#
# We use a common data loader, feature definitions and evaluation procedure throughout
# the analysis. This ensures that the models use the same customers, target definition
# and validation partitions.

# %%
from pathlib import Path
import sys
import warnings

# locate the repository from its root or the notebooks folder
project_root = Path.cwd()
shared_loader_file = project_root / 'src' / 'data.py'
if not shared_loader_file.is_file():
    project_root = project_root.parent
    shared_loader_file = project_root / 'src' / 'data.py'
# stop if the project data loader is unavailable
if not shared_loader_file.is_file():
    raise FileNotFoundError('Open this notebook from the repository root or notebooks folder')
# make the shared project modules available
sys.path.insert(0, str(project_root))

# import numerical, display and evaluation tools
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

# import the shared modelling procedures
from src.baselines import TenureRule
from src.data import RANDOM_STATE, load_train
from src.evaluation import (
    evaluate_classifier,
    make_model_pipeline,
    make_shared_folds,
    summarize_scores,
)
from src.features import (
    MAIN_FEATURE_GROUPS,
    RETENTION_COLUMNS,
    build_feature_configurations,
    build_feature_subsets,
    get_feature_spec,
)

from src.models import make_early_stopping_pipeline
from src.group_comparison import group_contributions, summarize_contributions
from src.thresholds import select_f1_threshold, flag_top_fraction

# show complete result tables
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# ## 2. Training data and churn prevalence
#
# The training partition contains 80% of the labelled customers. We preserve its row
# order and represent churn as the positive class, with 1 indicating churn and 0
# indicating continued service. The identifier and both forms of the target are excluded
# from the predictor table. Churn prevalence is the proportion of customers who churn.

# %%
# load the predefined training partition
train = load_train()
assert train['CustomerID'].is_unique

# separate the predictors from identifiers and outcomes
excluded_columns = ['CustomerID', 'Churn', 'churn']
X_train = train.drop(columns=excluded_columns)
y_train = train['churn']
training_prevalence = y_train.mean()

# summarise training size and churn prevalence
training_summary = pd.DataFrame({
    'customers': [len(train)],
    'churners': [int(y_train.sum())],
    'churn_prevalence': [training_prevalence],
})
display(training_summary)

# %% [markdown]
# ## 3. Evaluation procedure
#
# We use stratified five-fold cross-validation. Each model is fitted on four folds and
# evaluated on the remaining fold, so every customer receives one validation prediction.
# Stratification preserves similar churn proportions across folds. Shuffling with seed 42
# produces reproducible assignments, which are shared by all models and feature sets.
#
# Average precision (AP) is the primary metric. It summarises precision as recall
# increases: precision is the proportion of flagged customers who churn, and recall is
# the proportion of churners identified. We report the mean and standard deviation of
# the five AP scores. The standard deviation describes fold variation and is not a
# confidence interval. AP measures ranking performance rather than classification accuracy.
#
# Preprocessing is fitted within each training fold. Medians, category encodings,
# log-transformation choices and scaling therefore exclude the validation observations.
# ServiceArea target encoding uses additional cross-fitting within the training fold,
# so a training customer's encoding excludes that customer's own outcome. Gradient
# boosting also separates its internal stopping sample before fitting preprocessing.
#
# The final test evaluation is performed only after the input groups, preprocessing,
# model settings and decision threshold have been fixed.

# %%
# create the five shared training and validation assignments
shared_folds = make_shared_folds(X_train, y_train)

# record sample sizes and class proportions for each fold
fold_rows = []
for fold_number, (training_rows, validation_rows) in enumerate(shared_folds, start=1):
    fold_rows.append({
        'fold': fold_number,
        'training_rows': len(training_rows),
        'validation_rows': len(validation_rows),
        'training_churn_prevalence': y_train.iloc[training_rows].mean(),
        'validation_churn_prevalence': y_train.iloc[validation_rows].mean(),
    })
fold_summary = pd.DataFrame(fold_rows)
display(fold_summary.round(4))

# %% [markdown]
# ## 4. Background information and behavioural groups
#
# The background representation contains account, demographic and handset information.
# We compare all eight combinations of three additional groups: usage, usage change and
# service quality. A sequential comparison adds these groups in that order and describes
# their gains under one fixed information sequence.
#
# Calculated features are included only when all their parent groups are available.
# The usage-change group excludes RetentionCalls, RetentionOffersAccepted and
# MadeCallToRetentionTeam. These fields are assessed separately because their
# availability at prediction time is uncertain.

# %%
# define the fixed background information
baseline_groups = [
    'account_billing',
    'demographics_region',
    'handset_lifestyle',
]
# define the behavioural groups and their sequential comparison order
addition_order = ['usage', 'usage_change_retention', 'service_quality']
# construct the sequential and complete group configurations
feature_configurations = build_feature_configurations(baseline_groups, addition_order)
feature_subset_configurations = build_feature_subsets(baseline_groups, addition_order)

# describe the inputs available in each combination
configuration_rows = []
for name, groups in feature_subset_configurations.items():
    specification = get_feature_spec(groups)
    configuration_rows.append({
        'configuration': name,
        'groups': ', '.join(groups),
        'raw_predictors': len(specification['raw_features']),
    })
configuration_summary = pd.DataFrame(configuration_rows)
display(configuration_summary)

# %% [markdown]
# ## 5. Dummy classifiers
#
# The prior dummy assigns every validation customer the churn prevalence observed in
# the corresponding training fold. It learns no relationship between predictors and
# churn. Its constant scores produce an AP equal to validation prevalence, providing
# a reference for the learned models. No predictor preprocessing is required.

# %%
# evaluate constant scores based on training-fold prevalence
dummy = DummyClassifier(strategy='prior')
dummy_results, dummy_predictions = evaluate_classifier(
    dummy, X_train, y_train, shared_folds, name='Dummy (prior)',
)
assert np.allclose(dummy_results['average_precision'], dummy_results['churn_prevalence'])
display(dummy_results.round(4))

# %% [markdown]
# The stratified dummy randomly assigns a class according to the training-fold class
# proportions. Its probability output is a one-hot class assignment. This baseline is
# left unseeded to illustrate random variation; the validation assignments remain fixed.

# %%
# evaluate random class assignments on the same folds
random_dummy = DummyClassifier(strategy='stratified')
random_dummy_results, random_dummy_predictions = evaluate_classifier(
    random_dummy, X_train, y_train, shared_folds, name='Dummy (stratified)',
)
display(random_dummy_results.round(4))

# %% [markdown]
# ## 6. Fixed tenure rule
#
# The tenure rule assigns score 1 to customers with 11–12 months of service and score 0
# to all other customers. These scores represent two priority levels. Missing tenure
# receives score 0 because membership of the window is unknown. The rule requires no
# fitted preprocessing or model parameters.
#
# The window was identified during labelled exploratory analysis. Its development score
# therefore provides a descriptive baseline rather than an independent confirmation
# of the tenure pattern.

# %%
# evaluate the fixed tenure window
tenure_rule = TenureRule()
tenure_results, tenure_predictions = evaluate_classifier(
    tenure_rule, X_train, y_train, shared_folds, name='Fixed tenure (11-12 months)',
)
display(tenure_results.round(4))

# %% [markdown]
# ## 7. Baseline comparison
#
# We summarise the three baselines using mean AP, fold standard deviation and mean
# validation prevalence. Gains are absolute AP differences: a gain of 0.01 represents
# one AP point. All baselines use the same validation observations.

# %%
# combine the baseline fold scores and calculate their summary
baseline_fold_results = pd.concat(
    [dummy_results, random_dummy_results, tenure_results],
    ignore_index=True,
)
baseline_summary = summarize_scores(baseline_fold_results)
display(baseline_summary.round(4))

print(f'Training churn prevalence: {training_prevalence:.4f}')
for name, row in baseline_summary.iterrows():
    mean_ap = row['mean_ap']
    fold_std = row['std_ap']
    print(f'{name}: mean AP {mean_ap:.4f}, fold standard deviation {fold_std:.4f}')

# %% [markdown]
# ## 8. Out-of-fold predictions
#
# Each out-of-fold score is produced by a model fitted without that customer's
# validation fold. Predictions retain the training-row order and are linked to customer
# identifiers for threshold selection and segment analysis.
#
# Model AP is summarised from the five fold scores. Computing AP from pooled predictions
# can give a different result, particularly when the prior dummy assigns slightly
# different constant scores in different folds.

# %%
# collect baseline scores in the training-row order
baseline_predictions = pd.concat([
    dummy_predictions,
    random_dummy_predictions,
    tenure_predictions,
])
baseline_predictions = baseline_predictions.join(train[['CustomerID']])

# verify that every baseline has one score per customer
baseline_prediction_counts = baseline_predictions.groupby('model').size()
assert baseline_prediction_counts.eq(len(train)).all()
assert baseline_predictions['churn_score'].notna().all()
display(baseline_predictions.head())

# %% [markdown]
# ## 9. Preprocessing comparison
#
# We evaluate eight preprocessing combinations using all main feature groups and a
# fixed logistic regression model. The model uses the lbfgs solver, L2 regularisation
# with C=1 and at most 2,000 iterations. A convergence warning prevents an unfinished
# fit from entering the comparison.
#
# The age treatment either retains recorded zeros with zero indicators or replaces
# them with training-fold medians of positive ages. ServiceArea uses target or one-hot
# encoding. The log option applies log1p to eligible skewed measurements. All learned
# transformations are fitted within the training fold.
#
# Mean fold AP determines the preferred representation. Configurations within 0.001 AP
# of the best score are treated as practically similar. Within that tolerance, the
# preference is one-hot encoding, no log transformation, and median replacement of age
# zeros, in that order. This tolerance expresses a practical choice rather than a
# statistical test.

# %%
# define the fixed logistic regression model
all_groups = list(MAIN_FEATURE_GROUPS)
logistic_model = LogisticRegression(
    C=1.0,
    solver='lbfgs',
    max_iter=2000,
    random_state=RANDOM_STATE,
)

# list all eight preprocessing combinations
preprocessing_options = []
for age_strategy in ['keep', 'median']:
    for area_encoding in ['target', 'onehot']:
        for use_log in [True, False]:
            preprocessing_options.append({
                'age_strategy': age_strategy,
                'service_area_encoding': area_encoding,
                'log_transform': use_log,
            })

# evaluate each combination with fold-fitted preprocessing
preprocessing_results = []
preprocessing_predictions = {}
for options in preprocessing_options:
    name = (
        f"age={options['age_strategy']}, "
        f"area={options['service_area_encoding']}, "
        f"log={options['log_transform']}"
    )
    pipeline = make_model_pipeline(
        logistic_model,
        all_groups,
        model_kind='linear',
        include_retention=False,
        **options,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        fold_results, predictions = evaluate_classifier(
            pipeline, X_train, y_train, shared_folds, name=name,
        )
    for option, value in options.items():
        fold_results[option] = value
    preprocessing_results.append(fold_results)
    preprocessing_predictions[name] = predictions

# summarise the scores by configuration and fold
preprocessing_fold_results = pd.concat(preprocessing_results, ignore_index=True)
preprocessing_summary = summarize_scores(preprocessing_fold_results)
preprocessing_summary = preprocessing_summary.sort_values('mean_ap', ascending=False)
preprocessing_fold_ap = preprocessing_fold_results.pivot(
    index='model', columns='fold', values='average_precision',
)
display(preprocessing_summary.round(4))
display(preprocessing_fold_ap.round(4))


# %% [markdown]
# Paired differences compare preprocessing choices on the same validation fold while
# holding the other choices fixed. A positive difference favours the second treatment.

# %%
# compare one preprocessing choice while matching folds and other options
def compare_preprocessing_options(option, first, second):
    """compare one preprocessing choice while holding the others and fold fixed"""
    option_columns = ['age_strategy', 'service_area_encoding', 'log_transform']
    matching_columns = ['fold']
    for column in option_columns:
        if column != option:
            matching_columns.append(column)

    first_rows = preprocessing_fold_results[
        preprocessing_fold_results[option] == first
    ]
    second_rows = preprocessing_fold_results[
        preprocessing_fold_results[option] == second
    ]
    comparison = first_rows.merge(
        second_rows,
        on=matching_columns,
        suffixes=('_first', '_second'),
        validate='one_to_one',
    )
    comparison['ap_difference'] = (
        comparison['average_precision_second'] - comparison['average_precision_first']
    )
    return comparison[matching_columns + ['ap_difference']]


age_differences = compare_preprocessing_options('age_strategy', 'keep', 'median')
area_differences = compare_preprocessing_options('service_area_encoding', 'target', 'onehot')
log_differences = compare_preprocessing_options('log_transform', False, True)
print('Age: median minus keep')
display(age_differences.round(4))
print('ServiceArea: one-hot minus target')
display(area_differences.round(4))
print('Log: log1p minus no log')
display(log_differences.round(4))

# %% [markdown]
# The selected representation replaces age zeros with positive-age medians, uses
# target encoding for ServiceArea and applies log1p to eligible measurements. Its mean
# AP is 0.4269, compared with 0.4270 for the corresponding representation that retains
# age zeros. The difference falls within the predefined tolerance, favouring median
# replacement. The one-hot and no-log alternatives fall outside that tolerance.
#
# This representation is fixed for subsequent comparisons and matches the defaults of
# the shared preprocessing functions. Because the same folds informed this choice,
# the resulting AP is a development estimate.

# %%
# record the selected preprocessing choices explicitly
selected_preprocessing = {
    'age_strategy': 'median',
    'service_area_encoding': 'target',
    'log_transform': True,
}
selected_name = (
    f"age={selected_preprocessing['age_strategy']}, "
    f"area={selected_preprocessing['service_area_encoding']}, "
    f"log={selected_preprocessing['log_transform']}"
)
selection_tolerance = 0.001
best_mean_ap = preprocessing_summary['mean_ap'].max()
selected_mean_ap = preprocessing_summary.loc[selected_name, 'mean_ap']
if best_mean_ap - selected_mean_ap > selection_tolerance:
    print('Selected preprocessing is outside the predefined 0.001 AP tolerance.')

# the shared code defaults must match the selected preprocessing
import inspect
from src.preprocessing import make_preprocessor

default_options = inspect.signature(make_preprocessor).parameters
for option, value in selected_preprocessing.items():
    assert default_options[option].default == value, (
        f'make_preprocessor default for {option} differs from the selected preprocessing'
    )

display(pd.DataFrame([selected_preprocessing]))

# %% [markdown]
# ## 10. Retention sensitivity
#
# We add the three retention fields together while keeping preprocessing, model settings
# and validation folds fixed. Paired AP differences measure the change for the same
# validation observations. Positive values favour the representation with retention
# information.
#
# The retention fields remain outside the main analysis because their prediction-time
# availability is uncertain. A predictive gain alone does not establish that they are
# available before churn or that retention activity causes a change in churn.

# %%
# verify which predictors enter the retention comparison
main_spec = get_feature_spec(all_groups)
retention_spec = get_feature_spec(all_groups, include_retention=True)
assert len(main_spec['raw_features']) == 53
assert len(retention_spec['raw_features']) == 56
assert set(retention_spec['raw_features']) - set(main_spec['raw_features']) == set(RETENTION_COLUMNS)

# reuse the selected main representation and evaluate the retention representation
main_results = preprocessing_fold_results[
    preprocessing_fold_results['model'] == selected_name
].copy()
retention_pipeline = make_model_pipeline(
    logistic_model,
    all_groups,
    model_kind='linear',
    include_retention=True,
    **selected_preprocessing,
)
with warnings.catch_warnings():
    warnings.simplefilter('error', ConvergenceWarning)
    retention_results, retention_predictions = evaluate_classifier(
        retention_pipeline, X_train, y_train, shared_folds, name='With retention',
    )

# match the same validation folds before calculating AP changes
retention_comparison = main_results[['fold', 'average_precision']].merge(
    retention_results[['fold', 'average_precision']],
    on='fold',
    suffixes=('_main', '_retention'),
    validate='one_to_one',
)
retention_comparison['ap_difference'] = (
    retention_comparison['average_precision_retention']
    - retention_comparison['average_precision_main']
)
display(retention_comparison.round(4))
main_retention_ap = retention_comparison['average_precision_main'].mean()
with_retention_ap = retention_comparison['average_precision_retention'].mean()
retention_ap_gain = retention_comparison['ap_difference'].mean()
print(f"Mean AP without retention: {main_retention_ap:.4f}")
print(f"Mean AP with retention: {with_retention_ap:.4f}")
print(f"Mean paired AP difference: {retention_ap_gain:+.4f}")

# %% [markdown]
# Retention information increases mean AP from 0.4269 to 0.4313, with an unrounded mean
# gain of approximately 0.0045. Four folds improve and one declines. The gain is therefore
# small and not consistent across all folds.

# %% [markdown]
# ## 11. Logistic regression
#
# Logistic regression expresses churn log-odds as a weighted sum of the prepared
# predictors. We use the fixed lbfgs model with L2 regularisation, C=1 and at most 2,000
# iterations. L2 regularisation shrinks the fitted weights and limits their sensitivity
# to correlated inputs.
#
# The model uses the selected linear preprocessing on all six main feature groups.
# We reuse the scores and out-of-fold predictions obtained during preprocessing
# evaluation, since the model and representation are identical.

# %%
# reuse the matching logistic regression scores and predictions
logistic_results = main_results.copy()
logistic_results['model'] = 'Logistic Regression'
logistic_predictions = preprocessing_predictions[selected_name].copy()
logistic_predictions['model'] = 'Logistic Regression'


def paired_ap_difference(reference_results, results):
    """AP of `results` minus AP of `reference_results` on each shared fold"""
    paired = reference_results[['fold', 'average_precision']].merge(
        results[['fold', 'average_precision']],
        on='fold',
        suffixes=('_reference', '_model'),
        validate='one_to_one',
    )
    paired['ap_difference'] = (
        paired['average_precision_model'] - paired['average_precision_reference']
    )
    return paired


logistic_comparison = pd.concat(
    [baseline_fold_results, logistic_results],
    ignore_index=True,
)
logistic_comparison_summary = summarize_scores(logistic_comparison)
display(logistic_comparison_summary.round(4))

logistic_vs_tenure = paired_ap_difference(tenure_results, logistic_results)
print('Logistic Regression minus fixed tenure rule, per fold:')
display(logistic_vs_tenure.round(4))

# report performance from the current fold scores
logistic_mean_ap = logistic_results['average_precision'].mean()
logistic_fold_std = logistic_results['average_precision'].std()
logistic_mean_difference = logistic_vs_tenure['ap_difference'].mean()
logistic_performance_text = (
    f"Logistic regression achieved mean AP {logistic_mean_ap:.4f} "
    f"with fold standard deviation {logistic_fold_std:.4f}. "
    f"The mean AP difference relative to the tenure rule was "
    f"{logistic_mean_difference:+.4f}."
)
display(Markdown(logistic_performance_text))

# %% [markdown]
# Logistic regression improves ranking performance over both the prior dummy and the
# fixed tenure rule. This indicates that the combined information groups contain
# predictive information beyond prevalence and the single tenure window. The score
# remains a development result because preprocessing was selected using these folds.

# %% [markdown]
# ## 12. Gaussian naive Bayes
#
# Gaussian naive Bayes estimates a mean and variance for each prepared column within
# each churn class. It combines these distributions under an assumption of conditional
# independence between predictors. We retain the default variance smoothing and use
# the selected linear preprocessing.
#
# The estimator requires dense input. A conversion step within the model pipeline
# changes the sparse prepared matrix to dense form after fold-specific preprocessing.

# %%
# prepare dense input for Gaussian naive Bayes
from scipy import sparse
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def to_dense(values):
    """GaussianNB needs a dense matrix; the shared preprocessing can return a sparse one"""
    if sparse.issparse(values):
        return values.toarray()
    return values


naive_bayes = Pipeline([
    ('dense', FunctionTransformer(to_dense, accept_sparse=True)),
    ('classifier', GaussianNB()),
])
naive_bayes_pipeline = make_model_pipeline(naive_bayes, all_groups, model_kind='linear')
naive_bayes_results, naive_bayes_predictions = evaluate_classifier(
    naive_bayes_pipeline, X_train, y_train, shared_folds, name='Naive Bayes (Gaussian)',
)
display(naive_bayes_results.round(4))

naive_bayes_vs_logistic = paired_ap_difference(logistic_results, naive_bayes_results)
print('Naive Bayes minus Logistic Regression, per fold:')
display(naive_bayes_vs_logistic.round(4))

# report performance from the current fold scores
naive_bayes_mean_ap = naive_bayes_results['average_precision'].mean()
naive_bayes_fold_std = naive_bayes_results['average_precision'].std()
naive_bayes_mean_difference = naive_bayes_vs_logistic['ap_difference'].mean()
naive_bayes_performance_text = (
    f"Gaussian naive Bayes achieved mean AP {naive_bayes_mean_ap:.4f} "
    f"with fold standard deviation {naive_bayes_fold_std:.4f}. "
    f"The mean AP difference relative to logistic regression was "
    f"{naive_bayes_mean_difference:+.4f}."
)
display(Markdown(naive_bayes_performance_text))

# %% [markdown]
# We examine the independence assumption using correlations between usage measurements
# within each churn class. Strong within-class correlations provide evidence against
# independence. This is a descriptive assessment of the training data.

# %%
# calculate usage correlations separately within each churn class
usage_columns = MAIN_FEATURE_GROUPS['usage']
class_correlations = {}
for churn_value, label in [(0, 'stayed'), (1, 'churned')]:
    class_rows = train.loc[train['churn'] == churn_value, usage_columns]
    class_correlations[label] = class_rows.corr()

correlation_rows = []
for position, first in enumerate(usage_columns):
    for second in usage_columns[position + 1:]:
        correlation_rows.append({
            'pair': f'{first} / {second}',
            'correlation_stayed': class_correlations['stayed'].loc[first, second],
            'correlation_churned': class_correlations['churned'].loc[first, second],
        })
# arrange each unique pair as one table row
usage_correlations = pd.DataFrame(correlation_rows)
usage_correlations = usage_correlations.set_index('pair')
# retain pairs whose correlations are large in both churn classes
absolute_correlations = usage_correlations.abs()
usage_correlations['strongest'] = absolute_correlations.min(axis=1)
usage_correlations = usage_correlations.sort_values('strongest', ascending=False)
largest_correlations = usage_correlations.drop(columns='strongest').head(8)
display(largest_correlations.round(3))
strong_pairs = int((usage_correlations['strongest'] > 0.7).sum())
print(f'Usage pairs with |correlation| > 0.7 in both classes: {strong_pairs} of {len(usage_correlations)}')

# %% [markdown]
# Gaussian naive Bayes improves on the tenure baseline but performs below logistic
# regression on all five folds. Several usage measurements remain strongly correlated
# within both classes, which conflicts with its independence assumption. For example,
# MonthlyMinutes and ReceivedCalls have correlations of approximately 0.82.
#
# This dependence is a plausible explanation for weaker performance, although the
# correlation assessment does not isolate its effect on AP. The model provides a simple
# probabilistic comparison with the jointly fitted logistic regression model.

# %% [markdown]
# ## 13. Decision tree
#
# The decision tree divides customers through successive splits on individual
# predictors and estimates churn prevalence within each terminal group. We fix the
# maximum depth at 5 and require at least 50 customers per leaf. These constraints
# limit small terminal groups and keep the decision structure relatively simple.
#
# The tree uses the selected imputation and encoding procedures without log
# transformation or scaling. Its splits depend on the ordering of predictor values.

# %%
# define and evaluate the shallow decision tree
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=50,
    random_state=RANDOM_STATE,
)
tree_pipeline = make_model_pipeline(decision_tree, all_groups, model_kind='tree')
tree_results, tree_predictions = evaluate_classifier(
    tree_pipeline, X_train, y_train, shared_folds, name='Decision Tree (depth 5)',
)
display(tree_results.round(4))

tree_vs_logistic = paired_ap_difference(logistic_results, tree_results)
print('Decision Tree minus Logistic Regression, per fold:')
display(tree_vs_logistic.round(4))

# report performance from the current fold scores
tree_mean_ap = tree_results['average_precision'].mean()
tree_fold_std = tree_results['average_precision'].std()
tree_mean_difference = tree_vs_logistic['ap_difference'].mean()
tree_performance_text = (
    f"The decision tree achieved mean AP {tree_mean_ap:.4f} "
    f"with fold standard deviation {tree_fold_std:.4f}. "
    f"The mean AP difference relative to logistic regression was "
    f"{tree_mean_difference:+.4f}."
)
display(Markdown(tree_performance_text))

# %% [markdown]
# A depth sensitivity analysis compares depths 3, 5 and 8 using the same folds and
# minimum leaf size. The reference depth remains 5; these additional scores describe
# sensitivity rather than select a new model.

# %%
# evaluate depth sensitivity while reusing the depth-5 scores
depth_results = []
for depth in [3, 5, 8]:
    if depth == 5:
        depth_results.append(tree_results)
        continue
    depth_model = DecisionTreeClassifier(
        max_depth=depth, min_samples_leaf=50, random_state=RANDOM_STATE,
    )
    depth_pipeline = make_model_pipeline(
        depth_model,
        all_groups,
        model_kind='tree',
    )
    results, _ = evaluate_classifier(
        depth_pipeline, X_train, y_train, shared_folds, name=f'Decision Tree (depth {depth})',
    )
    depth_results.append(results)
depth_fold_results = pd.concat(depth_results, ignore_index=True)
depth_summary = summarize_scores(depth_fold_results)
display(depth_summary.round(4))

# %% [markdown]
# We fit a tree on the training portion of the first fold to describe its structure.
# Impurity importance measures the share of total split improvement assigned to each
# prepared predictor. It describes this fitted tree and does not establish causality
# or independent predictive value.

# %%
# fit one training-fold tree for descriptive inspection
first_training_rows = shared_folds[0][0]
inspected_tree = make_model_pipeline(decision_tree, all_groups, model_kind='tree')
inspected_tree.fit(X_train.iloc[first_training_rows], y_train.iloc[first_training_rows])
tree_feature_names = inspected_tree.named_steps['preprocess'].get_feature_names_out()
tree_importance = pd.Series(
    inspected_tree.named_steps['model'].feature_importances_,
    index=tree_feature_names,
    name='importance',
)
# retain the predictors used in at least one split
used_by_tree = tree_importance > 0
tree_importance = tree_importance[used_by_tree]
tree_importance = tree_importance.sort_values(ascending=False)
tree_importance_table = tree_importance.head(10).to_frame()
display(tree_importance_table.round(3))
print(f'Predictors used in at least one split: {len(tree_importance)} of {len(tree_feature_names)}')

# %% [markdown]
# The depth-5 tree performs above the tenure and naive Bayes models but below logistic
# regression on every fold. The deeper sensitivity model improves AP, indicating that
# the depth limit affects predictive performance.
#
# The inspected tree uses a small subset of the prepared predictors. Equipment age,
# relative usage change and tenure account for substantial split improvement. These
# importances describe one fitted structure and may favour measurements with many
# distinct values.

# %% [markdown]
# ### Comparison of the first three classifiers
#
# The baseline and classifier scores share the same validation folds. We combine their
# scores and customer-level predictions to maintain a consistent format across models.

# %%
# combine the available classifier and baseline scores
model_fold_results = pd.concat(
    [baseline_fold_results, logistic_results, naive_bayes_results, tree_results],
    ignore_index=True,
)
initial_model_summary = summarize_scores(model_fold_results)
initial_model_summary = initial_model_summary.sort_values('mean_ap', ascending=False)
display(initial_model_summary.round(4))

# combine customer scores before attaching identifiers
baseline_scores = baseline_predictions.drop(columns='CustomerID')
model_predictions = pd.concat([
    baseline_scores,
    logistic_predictions,
    naive_bayes_predictions,
    tree_predictions,
])
model_predictions = model_predictions.join(train[['CustomerID']])
model_prediction_counts = model_predictions.groupby('model').size()
assert model_prediction_counts.eq(len(train)).all()

# %% [markdown]
# ## 14. Random forest
#
# The random forest averages 300 decision trees fitted to random samples of customers.
# Each split considers a random subset of predictors. These two sources of variation
# reduce dependence on the structure of any single tree.
#
# We require at least 20 customers per leaf and consider the square root of the number
# of prepared predictors at each split. Tree depth is unrestricted. The input uses
# the selected tree preprocessing without log transformation or scaling.

# %%
# define and evaluate the random forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=20,
    max_features='sqrt',
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
forest_pipeline = make_model_pipeline(random_forest, all_groups, model_kind='tree')
forest_results, forest_predictions = evaluate_classifier(
    forest_pipeline, X_train, y_train, shared_folds, name='Random Forest (300 trees)',
)
display(forest_results.round(4))

forest_vs_logistic = paired_ap_difference(logistic_results, forest_results)
print('Random Forest minus Logistic Regression, per fold:')
display(forest_vs_logistic.round(4))

# report performance from the current fold scores
forest_mean_ap = forest_results['average_precision'].mean()
forest_fold_std = forest_results['average_precision'].std()
forest_mean_difference = forest_vs_logistic['ap_difference'].mean()
forest_performance_text = (
    f"The random forest achieved mean AP {forest_mean_ap:.4f} "
    f"with fold standard deviation {forest_fold_std:.4f}. "
    f"The mean AP difference relative to logistic regression was "
    f"{forest_mean_difference:+.4f}."
)
display(Markdown(forest_performance_text))

# %% [markdown]
# We inspect a forest fitted on the training portion of the first fold. Predictor
# importance is the average impurity reduction across its trees. The table describes
# which prepared inputs contribute to splits in this fitted forest.

# %%
# fit one training-fold forest for descriptive inspection
first_training_rows = shared_folds[0][0]
inspected_forest = make_model_pipeline(random_forest, all_groups, model_kind='tree')
inspected_forest.fit(X_train.iloc[first_training_rows], y_train.iloc[first_training_rows])
forest_feature_names = inspected_forest.named_steps['preprocess'].get_feature_names_out()
forest_importance = pd.Series(
    inspected_forest.named_steps['model'].feature_importances_,
    index=forest_feature_names,
    name='importance',
)
forest_importance = forest_importance.sort_values(ascending=False)
forest_importance_table = forest_importance.head(10).to_frame()
display(forest_importance_table.round(3))
used_by_forest = int((forest_importance > 0).sum())
print(f'Predictors with positive importance: {used_by_forest} of {len(forest_importance)}')

# %% [markdown]
# The forest improves AP over logistic regression on every fold, supporting the value
# of a model that captures nonlinear relationships and interactions. Impurity
# importance remains descriptive: columns with many distinct values can receive
# larger values, and positive importance does not establish a contribution on new data.

# %% [markdown]
# ## 15. Gradient boosting
#
# Gradient boosting builds small trees sequentially, with each tree improving the
# current predictions. We use a learning rate of 0.05, at most 300 rounds, at most
# 15 leaves per tree and at least 50 customers per leaf.
#
# A stratified 10% sample of each outer training fold is reserved for early stopping
# before any preprocessing is fitted. Preprocessing uses only the remaining fitting
# rows. The stopping sample is transformed using those fitted operations, and stopping
# occurs after 20 rounds without improvement in log loss. The outer validation fold
# is used only for AP evaluation.
#
# Prepared inputs are converted to dense form for the estimator. The fitted model is
# retained after stopping; the stopping observations are not used for a further refit.

# %%
# define and evaluate gradient boosting with a separate stopping sample
from sklearn.ensemble import HistGradientBoostingClassifier

gradient_boosting = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=15,
    min_samples_leaf=50,
    early_stopping=True,
    scoring='loss',
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)
boosting_pipeline = make_early_stopping_pipeline(gradient_boosting, all_groups)
boosting_results, boosting_predictions = evaluate_classifier(
    boosting_pipeline, X_train, y_train, shared_folds, name='Gradient Boosting (HistGB)',
)
display(boosting_results.round(4))
boosting_vs_logistic = paired_ap_difference(logistic_results, boosting_results)
boosting_vs_forest = paired_ap_difference(forest_results, boosting_results)
print('Gradient Boosting minus Logistic Regression, per fold:')
display(boosting_vs_logistic.round(4))
print('Gradient Boosting minus Random Forest, per fold:')
display(boosting_vs_forest.round(4))

# report performance from the current fold scores
boosting_mean_ap = boosting_results['average_precision'].mean()
boosting_fold_std = boosting_results['average_precision'].std()
boosting_mean_difference = boosting_vs_forest['ap_difference'].mean()
boosting_performance_text = (
    f"Gradient boosting achieved mean AP {boosting_mean_ap:.4f} "
    f"with fold standard deviation {boosting_fold_std:.4f}. "
    f"The mean AP difference relative to the random forest was "
    f"{boosting_mean_difference:+.4f}."
)
display(Markdown(boosting_performance_text))

# %% [markdown]
# We examine the first outer training fold to record the number of boosting rounds
# and the sizes of its internal fitting and stopping samples.

# %%
# inspect the internal fitting and stopping samples in the first outer fold
first_training_rows = shared_folds[0][0]
inspected_boosting = make_early_stopping_pipeline(gradient_boosting, all_groups)
inspected_boosting.fit(X_train.iloc[first_training_rows], y_train.iloc[first_training_rows])
used_rounds = inspected_boosting.estimator_.n_iter_
print(f'Boosting rounds used on fold 1 (maximum 300): {used_rounds}')
internal_fitting_count = len(inspected_boosting.fitting_rows_)
internal_stopping_count = len(inspected_boosting.stopping_rows_)
print(f'Internal fitting rows: {internal_fitting_count}; stopping rows: {internal_stopping_count}')

# %% [markdown]
# The combined model table compares mean AP and fold variation across all baselines
# and classifiers. Gradient boosting and random forest provide the strongest
# development scores. Their smaller performance difference requires more cautious
# interpretation than their larger gains over logistic regression.
#
# Preprocessing and model comparison use the same development folds. The reported
# scores therefore do not provide an independent estimate of final model performance.

# %%
# summarise all classifier and baseline scores
model_fold_results = pd.concat(
    [baseline_fold_results, logistic_results, naive_bayes_results, tree_results,
     forest_results, boosting_results], ignore_index=True,
)
model_summary = summarize_scores(model_fold_results)
model_summary = model_summary.sort_values('mean_ap', ascending=False)
display(model_summary.round(4))
forest_mean_ap = model_summary.loc['Random Forest (300 trees)', 'mean_ap']
boosting_mean_ap = model_summary.loc['Gradient Boosting (HistGB)', 'mean_ap']
print(f'Random Forest mean AP: {forest_mean_ap:.4f}')
print(f'Gradient Boosting mean AP: {boosting_mean_ap:.4f}')

# retain one score per model and customer in the training-row order
baseline_scores = baseline_predictions.drop(columns='CustomerID')
model_predictions = pd.concat([
    baseline_scores,
    logistic_predictions, naive_bayes_predictions, tree_predictions,
    forest_predictions, boosting_predictions,
])
model_predictions = model_predictions.join(train[['CustomerID']])
model_prediction_counts = model_predictions.groupby('model').size()
assert model_prediction_counts.eq(len(train)).all()
assert model_predictions['churn_score'].notna().all()

# %% [markdown]
# ## 16. Information-group comparison
#
# We evaluate all eight combinations of usage, usage change and service quality while
# retaining the account, demographic and handset background. Gradient boosting and
# logistic regression provide nonlinear and linear comparisons. Each model retains
# the same settings, preprocessing rules and validation folds across combinations.
#
# For each group, we calculate three AP contributions. The gain when added alone
# measures improvement over the background. The loss when removed from the full model
# measures its contribution when the other groups are available. The average gain
# across all six addition orders distributes the full-minus-background improvement
# between the groups, accounting for the different information already available.
# This last measure is a Shapley allocation of the measured AP gain.
#
# The six-order calculations reuse the eight fitted combinations. A fixed sequence
# of usage, usage change and service quality also provides a sequential comparison.
# These analyses rank the three behavioural groups conditional on the background,
# model, metric and feature definitions.

# %%
# label the behavioural groups in the result tables
group_labels = {
    'usage': 'usage',
    'usage_change_retention': 'usage change',
    'service_quality': 'service quality',
}

# match the sequential group sets to the eight evaluated configurations
fixed_order_names = []
for sequential_groups in feature_configurations.values():
    for name, groups in feature_subset_configurations.items():
        if set(groups) == set(sequential_groups):
            fixed_order_names.append(name)

fixed_order_labels = []
for group in addition_order:
    fixed_order_labels.append(group_labels[group])

# %%
# evaluate every behavioural combination with gradient boosting
boosting_group_runs = []
for name, groups in feature_subset_configurations.items():
    if name == 'full':
        # reuse the full-model evaluation on the same folds
        group_results = boosting_results.copy()
        group_results['model'] = name
    else:
        # fit preprocessing and the model separately within every fold
        group_pipeline = make_early_stopping_pipeline(gradient_boosting, groups)
        group_results, _ = evaluate_classifier(
            group_pipeline, X_train, y_train, shared_folds, name=name,
        )
    boosting_group_runs.append(group_results)

# summarise AP for the eight combinations in their defined order
boosting_group_results = pd.concat(boosting_group_runs, ignore_index=True)
boosting_summary = summarize_scores(boosting_group_results)
boosting_summary = boosting_summary.reindex(feature_subset_configurations)
display(boosting_summary.round(4))

# %%
# calculate each group's value alone, in the full model and across all orders
boosting_contributions, boosting_order_gains = group_contributions(
    boosting_group_results, feature_subset_configurations, baseline_groups, addition_order,
)
boosting_contribution_summary = summarize_contributions(boosting_contributions)
boosting_contribution_summary = boosting_contribution_summary.rename(index=group_labels)
display(boosting_contribution_summary.round(4))

# calculate paired gains along the fixed usage-to-service-quality sequence
boosting_fold_ap = boosting_group_results.pivot(
    index='fold', columns='model', values='average_precision',
)
boosting_sequential_ap = boosting_fold_ap[fixed_order_names]
boosting_sequential_gains = boosting_sequential_ap.diff(axis=1)
boosting_sequential_gains = boosting_sequential_gains.iloc[:, 1:]
boosting_sequential_gains.columns = fixed_order_labels
print('Gradient boosting: AP gains in the fixed addition order')
display(boosting_sequential_gains.round(4))

# %%
# average each addition-order gain across the same five validation folds
boosting_order_groups = boosting_order_gains.groupby(['order', 'group'], sort=False)
boosting_order_means = boosting_order_groups['ap_gain'].mean()
boosting_order_means = boosting_order_means.unstack()
boosting_order_means = boosting_order_means.rename(columns=group_labels)
print('Gradient boosting: mean AP gains in all six addition orders')
display(boosting_order_means.round(4))

# report the largest average contribution from the current results
boosting_mean_contributions = boosting_contribution_summary['mean_over_orders']
boosting_leading_group = boosting_mean_contributions.idxmax()
boosting_leading_gain = boosting_mean_contributions.max()
boosting_contribution_text = (
    f"For gradient boosting, {boosting_leading_group} had the largest average "
    f"contribution across addition orders, with AP gain {boosting_leading_gain:.4f}."
)
display(Markdown(boosting_contribution_text))

# %% [markdown]
# We apply the same group comparisons to logistic regression. Agreement between the
# linear and nonlinear models supports consistency across these modelling choices.
# Disagreement indicates that measured group value depends on the model.

# %%
# evaluate every behavioural combination with logistic regression
logistic_group_runs = []
for name, groups in feature_subset_configurations.items():
    if name == 'full':
        # reuse the full-model evaluation on the same folds
        group_results = logistic_results.copy()
        group_results['model'] = name
    else:
        # fit preprocessing and the model separately within every fold
        group_pipeline = make_model_pipeline(logistic_model, groups, model_kind='linear')
        group_results, _ = evaluate_classifier(
            group_pipeline, X_train, y_train, shared_folds, name=name,
        )
    logistic_group_runs.append(group_results)

# summarise AP for the eight combinations in their defined order
logistic_group_results = pd.concat(logistic_group_runs, ignore_index=True)
logistic_summary = summarize_scores(logistic_group_results)
logistic_summary = logistic_summary.reindex(feature_subset_configurations)
display(logistic_summary.round(4))

# %%
# calculate each group's value alone, in the full model and across all orders
logistic_contributions, logistic_order_gains = group_contributions(
    logistic_group_results, feature_subset_configurations, baseline_groups, addition_order,
)
logistic_contribution_summary = summarize_contributions(logistic_contributions)
logistic_contribution_summary = logistic_contribution_summary.rename(index=group_labels)
display(logistic_contribution_summary.round(4))

# calculate paired gains along the fixed usage-to-service-quality sequence
logistic_fold_ap = logistic_group_results.pivot(
    index='fold', columns='model', values='average_precision',
)
logistic_sequential_ap = logistic_fold_ap[fixed_order_names]
logistic_sequential_gains = logistic_sequential_ap.diff(axis=1)
logistic_sequential_gains = logistic_sequential_gains.iloc[:, 1:]
logistic_sequential_gains.columns = fixed_order_labels
print('Logistic regression: AP gains in the fixed addition order')
display(logistic_sequential_gains.round(4))

# %%
# average each addition-order gain across the same five validation folds
logistic_order_groups = logistic_order_gains.groupby(['order', 'group'], sort=False)
logistic_order_means = logistic_order_groups['ap_gain'].mean()
logistic_order_means = logistic_order_means.unstack()
logistic_order_means = logistic_order_means.rename(columns=group_labels)
print('Logistic regression: mean AP gains in all six addition orders')
display(logistic_order_means.round(4))

# report the largest average contribution from the current results
logistic_mean_contributions = logistic_contribution_summary['mean_over_orders']
logistic_leading_group = logistic_mean_contributions.idxmax()
logistic_leading_gain = logistic_mean_contributions.max()
logistic_contribution_text = (
    f"For logistic regression, {logistic_leading_group} had the largest average "
    f"contribution across addition orders, with AP gain {logistic_leading_gain:.4f}."
)
display(Markdown(logistic_contribution_text))

# %% [markdown]
# Usage has the largest average contribution in both models. Usage change has a
# similar contribution in gradient boosting, while service quality contributes less.
# The sequential comparison attributes the largest gradient-boosting gain to usage
# change when it follows usage. This difference illustrates why a single addition
# order does not determine a general group ranking.
#
# Calculated features enter when all their parent groups are present. Their predictive
# value is included in the group comparisons and can be shared between parent groups
# in the average-over-orders allocation. The analysis therefore assesses groups with
# their eligible calculated features.
#
# The average group gains sum to full-model AP minus background AP on each fold.
# Small removal losses can reflect overlapping information, even when a group is
# useful alone. Fold variation and model differences describe uncertainty in the
# development results. The rankings do not establish universal importance or causality.

# %% [markdown]
# ## 17. Error analysis by customer segment
#
# We select the classifier with the highest mean development AP and use its out-of-fold
# scores for threshold and error analysis. The decision threshold maximises F1, which
# combines precision and recall. If several thresholds have the same best F1, the
# lowest threshold is selected, preserving the highest recall among the tied choices.
#
# Tenure is divided into up to 10 months, 11–12 months, 13–24 months and over 24 months.
# MonthlyRevenue is divided into three approximately equal spending groups. Missing
# tenure or revenue receives a separate unknown category. The tenure window follows
# the pattern identified during exploration.
#
# For each tenure-by-spending segment, we report customer count, churner count, missed
# churners, share of all missed churners and miss rate. The miss rate is the proportion
# of actual churners in that segment who are not flagged.

# %%
# select the learned classifier with the highest mean development AP
learned_results = pd.concat([
    logistic_results, naive_bayes_results, tree_results, forest_results, boosting_results,
], ignore_index=True)
learned_model_summary = summarize_scores(learned_results)
analysis_model = learned_model_summary['mean_ap'].idxmax()
print(f'Development model selected for segment analysis: {analysis_model}')
# retain the selected model's scores and verify their customer alignment
selected_model_rows = model_predictions['model'] == analysis_model
oof = model_predictions[selected_model_rows]
assert len(oof) == len(train) and oof.index.equals(train.index)

# choose the development threshold and convert scores to churn flags
threshold, best_f1 = select_f1_threshold(oof['y_true'], oof['churn_score'])

flagged = oof['churn_score'] >= threshold
churned = oof['y_true'] == 1
# count the four prediction outcomes
true_positives = int((flagged & churned).sum())
false_positives = int((flagged & ~churned).sum())
false_negatives = int((~flagged & churned).sum())
true_negatives = int((~flagged & ~churned).sum())

# calculate precision, recall and the proportion selected for contact
selected_precision = true_positives / (true_positives + false_positives)
selected_recall = true_positives / (true_positives + false_negatives)
flagged_share = flagged.mean()

threshold_summary = pd.DataFrame([{
    'threshold': threshold,
    'flagged_customers': int(flagged.sum()),
    'flagged_share': flagged_share,
    'precision': selected_precision,
    'recall': selected_recall,
    'f1': best_f1,
    'false_negatives': false_negatives,
}])
display(threshold_summary.round(4))

# %%
# define tenure and spending segments for the training customers
segments = train[['MonthsInService', 'MonthlyRevenue']].copy()
segments['tenure'] = pd.cut(
    segments['MonthsInService'],
    bins=[-np.inf, 10, 12, 24, np.inf],
    labels=['up to 10 months', '11-12 months', '13-24 months', 'over 24 months'],
)
# retain missing tenure as a separate segment
segments['tenure'] = segments['tenure'].cat.add_categories('unknown tenure')
segments['tenure'] = segments['tenure'].fillna('unknown tenure')
segments['spending'] = pd.qcut(
    segments['MonthlyRevenue'], q=3, labels=['low spend', 'middle spend', 'high spend'],
)
# retain missing spending as a separate segment
segments['spending'] = segments['spending'].cat.add_categories('unknown spend')
segments['spending'] = segments['spending'].fillna('unknown spend')


def missed_by_segment(flagged):
    """count customers, churners and missed churners in each tenure and spending segment"""
    # attach flags and outcomes after the row-alignment check
    table = segments.copy()
    table['flagged'] = flagged.to_numpy()
    table['churned'] = oof['y_true'].to_numpy()
    table['missed'] = (table['churned'] == 1) & ~table['flagged']
    # count customers, churners and false negatives in each segment
    segment_groups = table.groupby(['tenure', 'spending'], observed=True)
    grouped = segment_groups.agg(
        customers=('churned', 'size'),
        churners=('churned', 'sum'),
        missed=('missed', 'sum'),
    )
    grouped['missed_share'] = grouped['missed'] / grouped['missed'].sum()
    grouped['miss_rate'] = grouped['missed'] / grouped['churners']
    return grouped.sort_values('missed', ascending=False)


def missed_by_tenure(table):
    """roll the segment table up to the four tenure bands"""
    count_columns = ['customers', 'churners', 'missed']
    tenure_groups = table.groupby('tenure', observed=True)
    rollup = tenure_groups[count_columns].sum()
    rollup['missed_share'] = rollup['missed'] / rollup['missed'].sum()
    rollup['miss_rate'] = rollup['missed'] / rollup['churners']
    return rollup


missed_table = missed_by_segment(flagged)
assert missed_table['missed'].sum() == false_negatives
assert missed_table['customers'].sum() == len(train)
display(missed_table.round(3))
print('Rolled up by tenure band:')
tenure_misses = missed_by_tenure(missed_table)
display(tenure_misses.round(3))

# %% [markdown]
# A contact-budget sensitivity analysis selects the highest-scoring 20% of customers,
# rounding the customer count down. CustomerID provides a stable ordering for tied
# scores. This policy measures errors under a fixed contact capacity.

# %%
# select customers under the fixed contact budget
top_fraction = 0.20
top_flagged = flag_top_fraction(oof['churn_score'], train['CustomerID'], top_fraction)
contact_budget = int(np.floor(len(train) * top_fraction))
flagged_customer_count = int(top_flagged.sum())
assert flagged_customer_count == contact_budget
top_missed_table = missed_by_segment(top_flagged)
# calculate recall and the tenure distribution under the contact budget
top_recall = 1 - top_missed_table['missed'].sum() / churned.sum()
print(f'Top {top_fraction:.0%} rule: {flagged_customer_count} customers flagged, recall {top_recall:.3f}')
display(top_missed_table.head(5).round(3))
print('Top 20% rule, rolled up by tenure band:')
top_tenure_misses = missed_by_tenure(top_missed_table)
display(top_tenure_misses.round(3))
print('Max-F1 rule, top 5 segments for comparison:')
display(missed_table.head(5).round(3))

# %% [markdown]
# Miss counts and miss rates describe different aspects of model errors. Large
# segments can contain many missed churners even when their miss rate is relatively
# low. Comparing a segment's share of missed churners with its share of actual churners
# helps distinguish group size from a disproportionate concentration of errors.
# Small segments, particularly those with unknown spending, provide limited evidence.

# %%
# compare the distribution of missed churners with the distribution of all churners
tenure_results_table = missed_by_tenure(missed_table)
tenure_results_table['share_of_churners'] = tenure_results_table['churners'] / churned.sum()
display(tenure_results_table.round(3))
print(f'Max-F1 threshold: {threshold:.4f}; flags {flagged_share:.1%} of customers; '
      f'precision {selected_precision:.3f}; recall {selected_recall:.3f}; '
      f'F1 {best_f1:.3f}; missed churners {false_negatives:,}.')
print(f'Top-20% contact budget: {flagged_customer_count:,} customers; recall {top_recall:.3f}.')

# %% [markdown]
# The youngest tenure band has a high miss rate despite containing a relatively small
# share of all churners. The fixed-budget policy reduces recall and changes the
# distribution of missed churners across the tenure bands. Threshold choice therefore
# affects both campaign size and the pattern of observed errors.
#
# Model and threshold selection use the same development outcomes as the error tables.
# Reported F1 and miss counts may consequently be optimistic, and the size of this
# effect has not been measured. The pooled scores also come from five fitted models;
# a model refitted on all training observations may produce different scores.
#
# The segment boundaries are analytical choices, and the results describe where errors
# occur rather than why customers churn. Final evaluation requires fixed input groups,
# preprocessing, model settings and threshold before the test outcomes are examined.
