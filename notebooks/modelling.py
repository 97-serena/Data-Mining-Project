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
# We set up the shared evaluation procedure and establish three simple reference scores:
# deterministic and random dummy classifiers, and a fixed tenure rule. The preprocessing
# and retention sections compare Logistic Regression on the shared folds.
#
# We compare five learned models, all eight behavioural-group combinations, and
# missed churners by customer segment. All results below are development results.

# %% [markdown]
# ## 1. Shared imports
#
# We locate the project and import the reusable evaluation code. Like the earlier
# chapters, this text notebook works from the project root or the notebooks folder.

# %%
from pathlib import Path
import sys
import warnings

project_root = Path.cwd()
if not (project_root / 'src' / 'data.py').is_file():
    project_root = project_root.parent
if not (project_root / 'src' / 'data.py').is_file():
    raise FileNotFoundError('Open this notebook from the repository root or notebooks folder')
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

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

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# ## 2. Training data and churn prevalence
#
# We use only the committed 80% training partition and preserve its row order. Churn is
# the positive class (1). Prevalence means the proportion of customers who churn.
# Identifiers and both versions of the target are excluded from the predictor table.
# We do not load a test dataframe or change dataset/split.csv.

# %%
train = load_train()
assert train['CustomerID'].is_unique

X_train = train.drop(columns=['CustomerID', 'Churn', 'churn'])
y_train = train['churn']
training_prevalence = y_train.mean()

display(pd.DataFrame({
    'customers': [len(train)],
    'churners': [int(y_train.sum())],
    'churn_prevalence': [training_prevalence],
}))

# %% [markdown]
# ## 3. Shared evaluation protocol
#
# Five-fold cross-validation fits on four parts of the training data and evaluates on
# the fifth, rotating until each customer has been evaluated once. Stratification keeps
# the churn proportion similar across the parts. Shuffling with seed 42 reproduces the
# assignments. Every model and configuration must reuse these same row positions.
#
# Average Precision (AP) is the primary metric. It summarises precision (the share of
# flagged customers who churn) as recall (the share of all churners found) increases.
# We score churn probabilities or risk scores, then report the mean and standard
# deviation of the five AP values. Standard deviation describes variation across folds;
# it is not a confidence interval. AP is not classification accuracy.
#
# For learned models, src.evaluation.make_model_pipeline attaches the existing
# make_preprocessor to the estimator. The evaluator clones and fits the whole pipeline
# within each fold: medians, encodings, log eligibility and scaling use training rows
# only. ServiceArea target encoding also uses its existing internal cross-fitting,
# so each training row is encoded without its own label. Validation rows are only
# transformed. Never preprocess all training data into one matrix before model CV.
#
# The test partition is reserved for a single final evaluation after preprocessing,
# model settings and any decision threshold have been fixed using training data.

# %%
shared_folds = make_shared_folds(X_train, y_train)

fold_rows = []
for fold_number, (training_rows, validation_rows) in enumerate(shared_folds, start=1):
    fold_rows.append({
        'fold': fold_number,
        'training_rows': len(training_rows),
        'validation_rows': len(validation_rows),
        'training_churn_prevalence': y_train.iloc[training_rows].mean(),
        'validation_churn_prevalence': y_train.iloc[validation_rows].mean(),
    })
display(pd.DataFrame(fold_rows).round(4))

# %% [markdown]
# ## 4. Background information and behavioural groups
#
# The fixed background contains account, demographic and handset information.
# Section 16 compares every combination of three extra groups: usage, usage change
# and service quality. We also keep one addition order as an easy-to-follow example.
# A single order cannot tell us which group contributes most across other combinations.
#
# A calculated feature is included only when all its parent groups are available.
# The legacy name usage_change_retention is kept for compatibility, but the three
# retention fields stay outside the main analysis and are checked separately in section 10.

# %%
baseline_groups = [
    'account_billing',
    'demographics_region',
    'handset_lifestyle',
]
addition_order = ['usage', 'usage_change_retention', 'service_quality']
feature_configurations = build_feature_configurations(baseline_groups, addition_order)
feature_subset_configurations = build_feature_subsets(baseline_groups, addition_order)

configuration_rows = []
for name, groups in feature_subset_configurations.items():
    specification = get_feature_spec(groups)
    configuration_rows.append({
        'configuration': name,
        'groups': ', '.join(groups),
        'raw_predictors': len(specification['raw_features']),
    })
display(pd.DataFrame(configuration_rows))

# %% [markdown]
# ## 5. Dummy classifiers
#
# The deterministic prior strategy ignores customer characteristics. It estimates churn
# prevalence using only the fold's training labels and assigns every validation customer that same
# score. It learns a class proportion, but no relationship between predictors and churn.
# With constant scores, AP equals the validation fold's churn prevalence. We report both
# values so later models have an explicit reference level.
#
# The dummy needs no preprocessing because none of the predictor values affects its score.

# %%
dummy = DummyClassifier(strategy='prior')
dummy_results, dummy_predictions = evaluate_classifier(
    dummy, X_train, y_train, shared_folds, name='Dummy (prior)',
)
assert np.allclose(dummy_results['average_precision'], dummy_results['churn_prevalence'])
display(dummy_results.round(4))

# %% [markdown]
# The stratified dummy also ignores customer characteristics, but randomly assigns a
# class according to the training fold's class proportions. Its probability output is
# a one-hot random class assignment, so AP can vary between runs. We leave its random
# state unset to make it non-deterministic; the shared fold assignments stay fixed.

# %%
random_dummy = DummyClassifier(strategy='stratified')
random_dummy_results, random_dummy_predictions = evaluate_classifier(
    random_dummy, X_train, y_train, shared_folds, name='Dummy (stratified)',
)
display(random_dummy_results.round(4))

# %% [markdown]
# ## 6. Fixed tenure rule
#
# The tenure baseline gives customers at months 11-12 score 1 and everyone else score 0.
# These are two priority levels, not estimated probabilities. The window is inclusive
# and is never tuned here. Missing tenure receives score 0 because window membership is
# unknown. No imputation, encoding or scaling is needed for this rule.
#
# The window came from labelled exploratory analysis. Although this rule fits no model
# parameters, its training-CV result is not an independent confirmation of that pattern.
# It is a transparent heuristic reference; the final holdout remains the later check.

# %%
tenure_rule = TenureRule()
tenure_results, tenure_predictions = evaluate_classifier(
    tenure_rule, X_train, y_train, shared_folds, name='Fixed tenure (11-12 months)',
)
display(tenure_results.round(4))

# %% [markdown]
# ## 7. Baseline comparison
#
# We combine the per-fold results and report mean AP, its standard deviation and mean
# validation prevalence. The AP gain is an absolute difference: 0.01 means one AP point,
# not a relative percentage improvement. All three predictors use identical folds.
# Later learned models should be compared with these references using the same procedure.

# %%
baseline_fold_results = pd.concat(
    [dummy_results, random_dummy_results, tenure_results],
    ignore_index=True,
)
baseline_summary = summarize_scores(baseline_fold_results)
display(baseline_summary.round(4))

print(f'Training churn prevalence: {training_prevalence:.4f}')
for name, row in baseline_summary.iterrows():
    print(f"{name}: AP {row['mean_ap']:.4f}, fold standard deviation {row['std_ap']:.4f}")

# %% [markdown]
# ## 8. Out-of-fold predictions for later analysis
#
# Out-of-fold means each customer's score comes from a predictor fitted on the other
# four folds. The evaluator returns those scores in the original training-row order.
# We attach customer ids to make the correspondence explicit. Kaustubh can obtain the
# same output format for learned models and choose a threshold or analyse missed churners
# later. We do not choose a threshold or perform his segment analysis here.
#
# AP is summarised from the five fold scores above, not recalculated after pooling all
# out-of-fold scores. Pooling can produce a different result, especially for the dummy
# because the constant training prevalence differs slightly between folds.

# %%
baseline_predictions = pd.concat([
    dummy_predictions,
    random_dummy_predictions,
    tenure_predictions,
])
baseline_predictions = baseline_predictions.join(train[['CustomerID']])

assert baseline_predictions.groupby('model').size().eq(len(train)).all()
assert baseline_predictions['churn_score'].notna().all()
display(baseline_predictions.head())

# %% [markdown]
# ## 9. Preprocessing comparisons
#
# We compare all eight combinations on the full main information set, using the same
# five folds and the same fixed Logistic Regression reference settings: lbfgs, L2
# regularisation with C=1, and at most 2000 iterations. A convergence warning stops
# the run rather than allowing an unfinished fit into the comparison.
#
# Age keep retains zeros and adds zero flags; age median replaces zeros with the
# training-fold positive-age median. ServiceArea uses either target or one-hot encoding.
# The log option applies log1p to eligible skewed columns, chosen within each fold.
# All learned preprocessing is fitted within the training portion of each fold.
#
# We select by mean fold AP. Configurations within 0.001 AP of the best are treated as
# practically similar; among them we prefer one-hot over target encoding, no log over
# log, then median over keep for age zeros. This tolerance is a practical choice, not a
# statistical test. The fold comparisons below show how consistent the differences are.

# %%
all_groups = list(MAIN_FEATURE_GROUPS)
logistic_model = LogisticRegression(
    C=1.0,
    solver='lbfgs',
    max_iter=2000,
    random_state=RANDOM_STATE,
)

preprocessing_options = []
for age_strategy in ['keep', 'median']:
    for area_encoding in ['target', 'onehot']:
        for use_log in [True, False]:
            preprocessing_options.append({
                'age_strategy': age_strategy,
                'service_area_encoding': area_encoding,
                'log_transform': use_log,
            })

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

preprocessing_fold_results = pd.concat(preprocessing_results, ignore_index=True)
preprocessing_summary = summarize_scores(preprocessing_fold_results)
preprocessing_summary = preprocessing_summary.sort_values('mean_ap', ascending=False)
preprocessing_fold_ap = preprocessing_fold_results.pivot(
    index='model', columns='fold', values='average_precision',
)
display(preprocessing_summary.round(4))
display(preprocessing_fold_ap.round(4))

# %% [markdown]
# Each difference below compares two choices on the same validation fold, holding the
# other two preprocessing options fixed. A positive value favors the second choice.

# %%
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
# The recorded choice is median age, target encoding and log1p. On the committed
# training split, its mean AP was 0.4269 versus 0.4270 for keep/target/log1p. These
# were within the 0.001 tolerance, so the median option was preferred. One-hot and
# no-log configurations were outside the tolerance. These scores were also used to
# choose preprocessing, so they are not an independent estimate of final performance.
# The test partition remains reserved for the final evaluation.
#
# The choice is a fixed literal for downstream comparisons, not a value silently
# reselected on every notebook run. Revisit it deliberately if the data or model changes.
#
# These values are also the defaults of `make_preprocessor` and `make_model_pipeline`, so a
# model built without explicit preprocessing options still uses the selected preprocessing.
# The check below stops the notebook if the code defaults and this choice ever diverge;
# change both together.

# %%
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
    print('The recorded choice is now outside the 0.001 AP tolerance; review it before continuing')

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
# We keep the chosen preprocessing, model settings and folds fixed, and add all three
# retention fields together. The main result comes from the matching run above; only
# the retention option changes in the new run. Paired AP differences compare the same
# validation fold, with positive values favoring the retention input set.
#
# Retention stays outside the main information-group comparison because its availability
# at prediction time is uncertain. An AP gain does not establish availability, leakage,
# or a causal effect of retention offers on churn.

# %%
main_spec = get_feature_spec(all_groups)
retention_spec = get_feature_spec(all_groups, include_retention=True)
assert len(main_spec['raw_features']) == 53
assert len(retention_spec['raw_features']) == 56
assert set(retention_spec['raw_features']) - set(main_spec['raw_features']) == set(RETENTION_COLUMNS)

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
print(f"Mean AP without retention: {retention_comparison['average_precision_main'].mean():.4f}")
print(f"Mean AP with retention: {retention_comparison['average_precision_retention'].mean():.4f}")
print(f"Mean paired AP difference: {retention_comparison['ap_difference'].mean():+.4f}")

# %% [markdown]
# On the committed training split, mean AP increased from 0.4269 to 0.4313, a paired
# gain of 0.0045. Four folds improved and one declined, so the gain is not consistent
# across every fold. This is a training-data sensitivity check, not a final test result.

# %% [markdown]
# ## 11. Logistic Regression (Shiqi)
#
# **Explanation:** Logistic Regression models the log-odds of churn as a weighted sum of the
# prepared predictors. We use the fixed reference settings from section 9: the lbfgs solver,
# L2 regularisation with C=1 and at most 2000 iterations. L2 regularisation shrinks the
# weights towards zero, which keeps the many correlated and one-hot columns stable. The input
# is the selected linear preprocessing (median age, target-encoded ServiceArea, log1p on
# skewed counts, scaling of continuous values) on all six main information groups.
#
# This configuration was already evaluated in section 9, so we reuse its fold scores and
# out-of-fold predictions instead of fitting it again.

# %%
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
display(summarize_scores(logistic_comparison).round(4))

logistic_vs_tenure = paired_ap_difference(tenure_results, logistic_results)
print('Logistic Regression minus fixed tenure rule, per fold:')
display(logistic_vs_tenure.round(4))

# %% [markdown]
# **Results and interpretation:** On the committed training split, Logistic Regression
# reached a mean AP of 0.4269 (fold standard deviation 0.0098), compared with 0.2882 for the
# prior dummy, which equals churn prevalence, and 0.3133 for the fixed tenure rule. It beat
# the tenure rule on every fold, by 0.10 to 0.12 AP. Combining all six information groups
# therefore ranks churners clearly better than chance or the single tenure window.
#
# The same folds were used to choose the preprocessing, so 0.4269 is a development result,
# not an independent estimate of final performance. Only fixed reference settings were used;
# the regularisation strength C has not been tuned.

# %% [markdown]
# ## 12. Naive Bayes (Shiqi)
#
# **Explanation:** Gaussian Naive Bayes estimates, for each churn class, a mean and variance
# for every prepared column, and combines the columns as if they were independent within each
# class. It has no regularisation or depth setting to choose; we keep scikit-learn's default
# variance smoothing. The input is the same selected linear preprocessing as Logistic
# Regression, so log1p and scaling make the skewed counts closer to the bell shape the model
# assumes.
#
# The shared preprocessing returns a sparse matrix because of one-hot encoding, but
# GaussianNB needs a dense one. A small conversion step inside the model pipeline handles
# this, so it is still fitted and applied fold by fold.

# %%
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

# %% [markdown]
# Naive Bayes assumes that the predictors are independent within each churn class. The table
# below checks this for the usage counts: it lists the most strongly correlated pairs, measured
# separately among customers who stayed and among customers who churned. This is a
# descriptive check on the training data only.

# %%
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
usage_correlations = pd.DataFrame(correlation_rows).set_index('pair')
usage_correlations['strongest'] = usage_correlations.abs().min(axis=1)
usage_correlations = usage_correlations.sort_values('strongest', ascending=False)
display(usage_correlations.drop(columns='strongest').head(8).round(3))
strong_pairs = int((usage_correlations['strongest'] > 0.7).sum())
print(f'Usage pairs with |correlation| > 0.7 in both classes: {strong_pairs} of {len(usage_correlations)}')

# %% [markdown]
# **Results and interpretation:** On the committed training split, Gaussian Naive Bayes
# reached a mean AP of 0.3730 (fold standard deviation 0.0074). This is above the fixed
# tenure rule (0.3133), but 0.054 AP below Logistic Regression on average, and lower on
# every fold (by 0.045 to 0.063).
#
# A likely reason is that its independence assumption is clearly violated. Eight of the 55
# pairs of usage counts have correlations above 0.7 among both customers who stayed and those
# who churned; for example, MonthlyMinutes and ReceivedCalls correlate at about 0.82. Naive
# Bayes treats such columns as separate pieces of evidence, so the same information is
# counted several times. Logistic Regression instead learns one weight per column jointly,
# which can share the effect between correlated columns.
#
# We suggest keeping Naive Bayes in the report as a simple probabilistic model whose weaker
# score illustrates this assumption, rather than removing it.

# %% [markdown]
# ## 13. Decision Tree (Shiqi)
#
# **Explanation:** A decision tree splits customers step by step on single predictors, such
# as "MonthsInService below some value", and predicts the churn share in each final group.
# We fix the settings before looking at scores: at most 5 levels of splits (`max_depth=5`)
# and at least 50 customers in every final group (`min_samples_leaf=50`). Both limits keep the
# tree shallow and stop it from memorising small groups, so its rules stay readable. The
# input is the selected tree preprocessing: the same imputation and encoding as for Logistic
# Regression, but without log1p or scaling, because splits only depend on the order of values.

# %%
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

# %% [markdown]
# To see how much the depth limit matters, we also score depths 3 and 8 on the same folds.
# This is a descriptive check: the chosen depth stays 5, so the check does not select
# the model.

# %%
depth_results = []
for depth in [3, 5, 8]:
    if depth == 5:
        depth_results.append(tree_results)
        continue
    depth_pipeline = make_model_pipeline(
        DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=RANDOM_STATE),
        all_groups,
        model_kind='tree',
    )
    results, _ = evaluate_classifier(
        depth_pipeline, X_train, y_train, shared_folds, name=f'Decision Tree (depth {depth})',
    )
    depth_results.append(results)
display(summarize_scores(pd.concat(depth_results, ignore_index=True)).round(4))

# %% [markdown]
# To see what the tree uses, we fit it once on the training part of fold 1 and list the
# predictors with the largest importance. Importance is the share of the total split
# improvement that a predictor provides; it describes this one tree, not a causal effect.

# %%
first_training_rows = shared_folds[0][0]
inspected_tree = make_model_pipeline(decision_tree, all_groups, model_kind='tree')
inspected_tree.fit(X_train.iloc[first_training_rows], y_train.iloc[first_training_rows])
tree_feature_names = inspected_tree.named_steps['preprocess'].get_feature_names_out()
tree_importance = pd.Series(
    inspected_tree.named_steps['model'].feature_importances_,
    index=tree_feature_names,
    name='importance',
)
tree_importance = tree_importance[tree_importance > 0].sort_values(ascending=False)
display(tree_importance.head(10).round(3).to_frame())
print(f'Predictors used in at least one split: {len(tree_importance)} of {len(tree_feature_names)}')

# %% [markdown]
# **Results and interpretation:** On the committed training split, the depth-5 tree reached
# a mean AP of 0.4066 (fold standard deviation 0.0086). It is well above the tenure rule
# (0.3133) and Naive Bayes (0.3730), but 0.020 AP below Logistic Regression on average and
# lower on every fold.
#
# The depth check shows that the limit matters: depth 3 reached 0.3725 and depth 8 reached
# 0.4166. A single shallow tree cannot express many combinations of predictors, which is the
# gap that Random Forest and Gradient Boosting (sections 14 and 15) are designed to close.
# We keep depth 5 as the fixed initial setting; any tuning should be decided together with
# those models.
#
# The tree fitted on fold 1 used only 18 of 115 prepared columns. Equipment age
# (CurrentEquipmentDays) gave about a third of the total split improvement, followed by the
# engineered usage change (RelChangeMinutes), tenure (MonthsInService), the missing-change
# indicator and the 11-12 month window (ContractWindow). These importances describe one tree
# and show association with churn, not cause.

# %% [markdown]
# ### Summary of the models so far
#
# All models below use the same five folds. Out-of-fold predictions for every model are
# collected in `model_predictions` for the later threshold and segment analysis.

# %%
model_fold_results = pd.concat(
    [baseline_fold_results, logistic_results, naive_bayes_results, tree_results],
    ignore_index=True,
)
display(summarize_scores(model_fold_results).sort_values('mean_ap', ascending=False).round(4))

model_predictions = pd.concat([
    baseline_predictions.drop(columns='CustomerID'),
    logistic_predictions,
    naive_bayes_predictions,
    tree_predictions,
]).join(train[['CustomerID']])
assert model_predictions.groupby('model').size().eq(len(train)).all()

# %% [markdown]
# ## 14. Random Forest (Yan-Ru)
#
# Fit an initial Random Forest using the selected preprocessing and shared folds. A forest
# combines predictions from multiple trees to reduce dependence on any one tree.
#
# **Explanation:** A Random Forest averages many decision trees. Each tree is fitted on a
# random resample of the training customers, and at every split it may only choose from a
# random subset of the columns. The trees therefore differ from each other, and their errors
# partly cancel when the predictions are averaged. We fix the settings before looking at
# scores: 300 trees (more trees mainly add run time), at least 20 customers in every final
# group (`min_samples_leaf=20`), which stops single trees from memorising individual
# customers, and the square root of the number of columns tried at each split
# (`max_features='sqrt'`). There is no depth limit, because averaging many trees already
# reduces the instability of a single deep tree. The input is the selected tree
# preprocessing: the same imputation and encoding as for Logistic Regression, but without
# log1p or scaling.

# %%
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

# %% [markdown]
# To see what the forest uses, we fit it once on the training part of fold 1 and list the
# predictors with the largest importance. Importance is the share of the total split
# improvement that a predictor provides, averaged over all trees. It describes this one
# forest and shows association with churn, not cause.

# %%
first_training_rows = shared_folds[0][0]
inspected_forest = make_model_pipeline(random_forest, all_groups, model_kind='tree')
inspected_forest.fit(X_train.iloc[first_training_rows], y_train.iloc[first_training_rows])
forest_feature_names = inspected_forest.named_steps['preprocess'].get_feature_names_out()
forest_importance = pd.Series(
    inspected_forest.named_steps['model'].feature_importances_,
    index=forest_feature_names,
    name='importance',
).sort_values(ascending=False)
display(forest_importance.head(10).round(3).to_frame())
print(f'Predictors with importance above zero: {int((forest_importance > 0).sum())} of {len(forest_importance)}')

# %% [markdown]
# **Results and interpretation:** Compare the forest with Logistic Regression using
# the paired fold table above. A higher AP means better ranking of churners. The settings
# are fixed starting values; we have not tuned them. These are development scores.
#
# The importance table describes splits in one fitted forest. Columns with many distinct
# values can receive more importance, so positive importance does not prove that a column
# helps on new customers. It also does not show cause and effect.

# %% [markdown]
# ## 15. Gradient Boosting (Yan-Ru)
#
# Gradient Boosting builds small trees in sequence. Each new tree tries to improve the
# predictions made so far. We keep the submitted settings: learning rate 0.05, at most
# 300 rounds, at most 15 leaves per tree, and at least 50 customers per leaf.
#
# Early stopping uses a separate 10% sample from each outer training fold and stops
# after 20 rounds without improvement in log loss. The stopping sample is separated
# BEFORE any preprocessing is fitted. Its labels cannot affect target encoding, medians
# or other fitted preparation steps. The outer validation fold is used only for scoring.
# The wrapper also converts the prepared matrices to the dense format this model needs.
# We retain the internal fitting model after stopping; we do not refit it on the 10% sample.

# %%
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

# %% [markdown]
# Inspect early stopping on the training part of fold 1. The fitting and stopping
# rows below are both inside that outer training fold.

# %%
first_training_rows = shared_folds[0][0]
inspected_boosting = make_early_stopping_pipeline(gradient_boosting, all_groups)
inspected_boosting.fit(X_train.iloc[first_training_rows], y_train.iloc[first_training_rows])
used_rounds = inspected_boosting.estimator_.n_iter_
print(f'Boosting rounds used on fold 1 (maximum 300): {used_rounds}')
print(f'Internal fitting rows: {len(inspected_boosting.fitting_rows_)}; '
      f'separate stopping rows: {len(inspected_boosting.stopping_rows_)}')

# %% [markdown]
# **Results and interpretation:** The tables below show the current run. Small differences
# between the two tree ensembles should be interpreted carefully. The same development
# folds have helped us choose preprocessing and compare models. They do not provide a
# final independent performance estimate. The final test set remains reserved.

# %%
model_fold_results = pd.concat(
    [baseline_fold_results, logistic_results, naive_bayes_results, tree_results,
     forest_results, boosting_results], ignore_index=True,
)
model_summary = summarize_scores(model_fold_results).sort_values('mean_ap', ascending=False)
display(model_summary.round(4))
print(f"Random Forest mean AP: {model_summary.loc['Random Forest (300 trees)', 'mean_ap']:.4f}")
print(f"Gradient Boosting mean AP: {model_summary.loc['Gradient Boosting (HistGB)', 'mean_ap']:.4f}")

model_predictions = pd.concat([
    baseline_predictions.drop(columns='CustomerID'),
    logistic_predictions, naive_bayes_predictions, tree_predictions,
    forest_predictions, boosting_predictions,
]).join(train[['CustomerID']])
assert model_predictions.groupby('model').size().eq(len(train)).all()
assert model_predictions['churn_score'].notna().all()

# %% [markdown]
# ## 16. Information-group comparison (Kaustubh)
#
# We keep the same background information and compare ALL EIGHT combinations of usage,
# usage change and service quality. Every combination uses the same five outer folds.
# Gradient Boosting is the nonlinear model; Logistic Regression is a second check.
# Each model keeps its settings and preprocessing rules fixed across combinations.
# The full-model result is reused from above, so each other combination is fitted once.
#
# We ask three different questions for each group:
# - Add to baseline: how much does it help when no other behavioural group is present?
# - Remove from full: how much AP do we lose when it is removed and the model is retrained?
# - Mean over orders: how much does it add on average across all six possible orders?
#
# The last calculation is a Shapley allocation of the full-minus-baseline AP gain.
# It reuses the eight scores: it does not fit six more sets of models. The order-gain
# table also shows whether a group's contribution changes with the groups already present.
# The original fixed order remains below as an illustration, not as a general ranking.
#
# These questions differ: a group may be useful alone but replaceable by other groups.
# Groups may also be more useful together. This compares only the three extra groups;
# it does not rank the fixed account, demographic or handset groups.

# %%
group_labels = {
    'usage': 'usage',
    'usage_change_retention': 'usage change',
    'service_quality': 'service quality',
}


def run_group_comparison(pipeline_factory, full_results):
    fold_results = []
    for name, groups in feature_subset_configurations.items():
        if name == 'full':
            results = full_results.copy()
            results['model'] = name
        else:
            results, _ = evaluate_classifier(
                pipeline_factory(groups), X_train, y_train, shared_folds, name=name,
            )
        fold_results.append(results)
    fold_results = pd.concat(fold_results, ignore_index=True)
    contributions, order_gains = group_contributions(
        fold_results, feature_subset_configurations, baseline_groups, addition_order,
    )
    return fold_results, contributions, order_gains


def fixed_order_gains(fold_results):
    ap = fold_results.pivot(index='fold', columns='model', values='average_precision')
    subset_names = {frozenset(groups): name
                    for name, groups in feature_subset_configurations.items()}
    names = [subset_names[frozenset(groups)] for groups in feature_configurations.values()]
    gains = ap[names].diff(axis=1).iloc[:, 1:]
    gains.columns = [group_labels[group] for group in addition_order]
    return gains


def show_group_comparison(fold_results, contributions, order_gains):
    summary = summarize_scores(fold_results).reindex(feature_subset_configurations)
    display(summary.round(4))
    print('Paired AP contributions (means across five folds):')
    contribution_summary = summarize_contributions(contributions).rename(index=group_labels)
    display(contribution_summary.round(4))
    print('Original fixed order: usage -> usage change -> service quality')
    display(fixed_order_gains(fold_results).round(4))
    order_means = order_gains.groupby(['order', 'group'], sort=False)['ap_gain'].mean().unstack()
    print('Mean AP gain in each of the six addition orders:')
    display(order_means.rename(columns=group_labels).round(4))
    strongest = contribution_summary['mean_over_orders'].idxmax()
    print(f'Largest average contribution in this run: {strongest}.')
    print('This is conditional on the background, model, metric and feature definitions.')
    return summary, contribution_summary


boosting_group_results, boosting_contributions, boosting_order_gains = run_group_comparison(
    lambda groups: make_early_stopping_pipeline(gradient_boosting, groups), boosting_results,
)
boosting_summary, boosting_contribution_summary = show_group_comparison(
    boosting_group_results, boosting_contributions, boosting_order_gains,
)

# %% [markdown]
# Repeat the same eight combinations with Logistic Regression. Agreement between two
# different model types makes a result more convincing; disagreement is also useful to report.

# %%
logistic_group_results, logistic_contributions, logistic_order_gains = run_group_comparison(
    lambda groups: make_model_pipeline(logistic_model, groups, model_kind='linear'),
    logistic_results,
)
logistic_summary, logistic_contribution_summary = show_group_comparison(
    logistic_group_results, logistic_contributions, logistic_order_gains,
)

# %% [markdown]
# **Results and interpretation:** Read the generated tables above rather than a copied
# score from an older run. Compare each group's gain alone, its removal loss and its
# average contribution. A small removal loss can mean another group provides similar
# information; it does not establish that the removed group contains no useful information.
#
# Calculated features enter only when all parent groups are present. Their benefit is
# therefore part of the group comparison and can be shared between their parent groups
# in the average-over-orders calculation. This is not a comparison of raw columns alone.
#
# The average gains add up to the full-model AP minus baseline AP on every fold.
# Fold variation and model disagreement should be reported. Standard deviations are
# descriptive, not confidence intervals or significance tests. A ranking on this dataset
# is not a universal ranking, and predictive improvement does not prove cause and effect.
#
# If a combination is chosen because it scores best here, its score is a development
# score. Fix the final choices before using the reserved test set. Do not search for an
# addition order that makes a preferred group look strongest.
#
# Method background: [Grouped Feature Importance](https://arxiv.org/abs/2104.11688).

# %% [markdown]
# ## 17. Error analysis by customer segment (Kaustubh)
#
# Choose a decision threshold using out-of-fold training predictions. A threshold converts
# risk scores into churn flags. Then count false negatives, meaning actual churners who were
# not flagged, within each tenure and spending segment. Keep test outcomes reserved.
#
# **Explanation:** We use the learned model with the highest mean development AP.
# Each customer has a score from an outer-fold model that was not fitted on that customer.
# We choose the threshold with the highest F1, which balances precision and recall.
# Equal best F1 values choose the lower threshold, which finds at least as many churners.
# This threshold is chosen and described on the same outcomes, so this is exploratory.
# A separate sensitivity check flags at most 20% of customers by score. Equal scores
# are ordered by CustomerID so that the contact budget is respected.
#
# The segments combine four tenure bands (up to 10 months, 11-12 months, 13-24 months, over
# 24 months) with three spending groups, defined as terciles of MonthlyRevenue (low, middle,
# high). Customers with missing MonthlyRevenue form a separate group. The 11-12 month band
# follows the window found in the exploration and used in the tenure rule. The segments were
# fixed before the false negatives were counted. For each segment we report its size, its
# churners, its missed churners, its share of all missed churners, and its miss rate, which
# is the missed churners divided by the churners in the segment.

# %%
learned_results = pd.concat([
    logistic_results, naive_bayes_results, tree_results, forest_results, boosting_results,
], ignore_index=True)
analysis_model = summarize_scores(learned_results)['mean_ap'].idxmax()
print(f'Development model selected for segment analysis: {analysis_model}')
oof = model_predictions[model_predictions['model'] == analysis_model]
assert len(oof) == len(train) and oof.index.equals(train.index)

threshold, best_f1 = select_f1_threshold(oof['y_true'], oof['churn_score'])

flagged = oof['churn_score'] >= threshold
churned = oof['y_true'] == 1
true_positives = int((flagged & churned).sum())
false_positives = int((flagged & ~churned).sum())
false_negatives = int((~flagged & churned).sum())
true_negatives = int((~flagged & ~churned).sum())

threshold_summary = pd.DataFrame([{
    'threshold': threshold,
    'flagged_customers': int(flagged.sum()),
    'flagged_share': flagged.mean(),
    'precision': true_positives / (true_positives + false_positives),
    'recall': true_positives / (true_positives + false_negatives),
    'f1': best_f1,
    'false_negatives': false_negatives,
}])
display(threshold_summary.round(4))

# %%
segments = train[['MonthsInService', 'MonthlyRevenue']].copy()
segments['tenure'] = pd.cut(
    segments['MonthsInService'],
    bins=[-np.inf, 10, 12, 24, np.inf],
    labels=['up to 10 months', '11-12 months', '13-24 months', 'over 24 months'],
)
segments['tenure'] = segments['tenure'].cat.add_categories('unknown tenure').fillna('unknown tenure')
segments['spending'] = pd.qcut(
    segments['MonthlyRevenue'], q=3, labels=['low spend', 'middle spend', 'high spend'],
)
segments['spending'] = segments['spending'].cat.add_categories('unknown spend').fillna('unknown spend')


def missed_by_segment(flagged):
    """count customers, churners and missed churners in each tenure and spending segment"""
    table = segments.assign(flagged=flagged.to_numpy(), churned=oof['y_true'].to_numpy())
    table['missed'] = (table['churned'] == 1) & ~table['flagged']
    grouped = table.groupby(['tenure', 'spending'], observed=True).agg(
        customers=('churned', 'size'),
        churners=('churned', 'sum'),
        missed=('missed', 'sum'),
    )
    grouped['missed_share'] = grouped['missed'] / grouped['missed'].sum()
    grouped['miss_rate'] = grouped['missed'] / grouped['churners']
    return grouped.sort_values('missed', ascending=False)


def missed_by_tenure(table):
    """roll the segment table up to the four tenure bands"""
    rollup = table.groupby('tenure', observed=True)[['customers', 'churners', 'missed']].sum()
    rollup['missed_share'] = rollup['missed'] / rollup['missed'].sum()
    rollup['miss_rate'] = rollup['missed'] / rollup['churners']
    return rollup


missed_table = missed_by_segment(flagged)
assert missed_table['missed'].sum() == false_negatives
assert missed_table['customers'].sum() == len(train)
display(missed_table.round(3))
print('Rolled up by tenure band:')
display(missed_by_tenure(missed_table).round(3))

# %% [markdown]
# Sensitivity check: select floor(20% * customer count) customers with the highest
# scores. CustomerID breaks ties. This respects a strict contact budget.

# %%
top_fraction = 0.20
top_flagged = flag_top_fraction(oof['churn_score'], train['CustomerID'], top_fraction)
assert int(top_flagged.sum()) == int(np.floor(len(train) * top_fraction))
top_missed_table = missed_by_segment(top_flagged)
top_recall = 1 - top_missed_table['missed'].sum() / churned.sum()
print(f'Top {top_fraction:.0%} rule: {int(top_flagged.sum())} customers flagged, recall {top_recall:.3f}')
display(top_missed_table.head(5).round(3))
print('Top 20% rule, rolled up by tenure band:')
display(missed_by_tenure(top_missed_table).round(3))
print('Max-F1 rule, top 5 segments for comparison:')
display(missed_table.head(5).round(3))

# %% [markdown]
# **Results and interpretation:** The generated tables show the current threshold,
# contact rate and missed churners. The largest number of misses and the highest miss
# rate answer different questions: large groups can contain many misses even when their
# miss rate is low. Compare a group's share of misses with its share of actual churners.
# Small cells, especially unknown-spending groups, may be too small for firm conclusions.

# %%
tenure_results_table = missed_by_tenure(missed_table)
tenure_results_table['share_of_churners'] = tenure_results_table['churners'] / churned.sum()
display(tenure_results_table.round(3))
print(f'Max-F1 threshold: {threshold:.4f}; flags {flagged.mean():.1%} of customers; '
      f'precision {threshold_summary.iloc[0]["precision"]:.3f}; '
      f'recall {threshold_summary.iloc[0]["recall"]:.3f}; '
      f'F1 {best_f1:.3f}; missed churners {false_negatives:,}.')
print(f'Top-20% contact budget: {int(top_flagged.sum()):,} customers; recall {top_recall:.3f}.')

# %% [markdown]
# The choice of threshold can change which groups contain the most misses. A high F1
# does not guarantee that a campaign's contact rate is affordable. A real campaign needs
# a contact budget or a cost-based objective as well as a predictive model.
#
# Limitations: model choice and threshold choice both used these development outcomes.
# The reported F1 and error counts may be optimistic; we have not measured the size of
# that effect. Scores are pooled from five models, and a later model fitted on all
# training data may behave differently. Segment boundaries also affect the picture.
# This analysis shows where churners are missed, not why they churn. Before final test
# evaluation, fix the model, input groups and threshold. The test partition is unused here.
