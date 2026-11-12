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
# Separate sections below provide space for the team's remaining model code,
# explanations and interpretation of results.

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
    get_feature_spec,
)

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
# ## 4. Information baseline and addition order
#
# An information baseline is a starting set of predictors, not a simple predictor like
# the dummy. We fix the following order before comparing scores: background account,
# customer and handset/lifestyle information first, then usage, usage change and service
# quality. This directly addresses whether behavioural information adds predictive value.
#
# Each gain is conditional on the groups already present. Engineered features enter only
# once all parent groups are available, so a gain can include those eligible derivatives.
# The legacy group name usage_change_retention contains no retention fields in the main
# representation; all three retention fields remain a separate sensitivity comparison.

# %%
baseline_groups = [
    'account_billing',
    'demographics_region',
    'handset_lifestyle',
]
addition_order = ['usage', 'usage_change_retention', 'service_quality']
feature_configurations = build_feature_configurations(baseline_groups, addition_order)

configuration_rows = []
for name, groups in feature_configurations.items():
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
# five folds and the same Logistic Regression settings. The team's model implementation
# is not available yet, so this comparison uses a fixed reference model: lbfgs, L2
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
# The fixed reference model and selected preprocessing were evaluated above. Reuse the
# existing fold scores and out-of-fold predictions instead of fitting them again. The
# team's eventual model settings may differ and should be recorded when available.

# %%
logistic_results = main_results.copy()
logistic_predictions = preprocessing_predictions[selected_name].copy()
logistic_comparison = pd.concat(
    [baseline_fold_results, logistic_results],
    ignore_index=True,
)
display(summarize_scores(logistic_comparison).round(4))

# %% [markdown]
# The Logistic Regression score above uses the same folds that selected preprocessing;
# compare it with the baselines as a development result, not an independent final score.

# %% [markdown]
# ## 12. Naive Bayes (Shiqi)
#
# Fit Gaussian Naive Bayes using the linear preprocessing with log1p and scaling, and
# evaluate its AP using the shared folds. Discuss correlated usage counts, which violate
# its assumption that predictors are independent within each churn class.
#
# **Explanation:** Add the model settings and any required input-format handling here.

# %%
# Add the Naive Bayes pipeline and shared-fold evaluation here.

# %% [markdown]
# **Results and interpretation:** Add the AP summary, assumption limitations and whether
# this model should be retained in the final report.

# %% [markdown]
# ## 13. Decision Tree (Shiqi)
#
# Fit an initial shallow Decision Tree using the selected preprocessing and shared folds.
# A shallow tree limits the number of successive splits used to make a prediction.
#
# **Explanation:** Add the depth limit, other model settings and their rationale here.

# %%
# Add the Decision Tree pipeline and shared-fold evaluation here.

# %% [markdown]
# **Results and interpretation:** Add the AP summary and comparison with the baselines here.

# %% [markdown]
# ## 14. Random Forest (Yan-Ru)
#
# Fit an initial Random Forest using the selected preprocessing and shared folds.
# A forest combines predictions from multiple trees to reduce dependence on any one tree.
#
# **Explanation:** Add the model settings and reasons for choosing them here.

# %%
# Add the Random Forest pipeline and shared-fold evaluation here.

# %% [markdown]
# **Results and interpretation:** Add the AP summary and comparison with the baselines here.

# %% [markdown]
# ## 15. Gradient Boosting (Yan-Ru)
#
# Fit an initial HistGradientBoostingClassifier using the selected preprocessing and
# shared folds. Boosting builds trees successively to improve the combined predictions.
#
# **Explanation:** Add the model settings and any required input-format handling here.

# %%
# Add the Gradient Boosting pipeline and shared-fold evaluation here.

# %% [markdown]
# **Results and interpretation:** Add the AP summary and comparison with the baselines here.

# %% [markdown]
# ## 16. First model and information-group evaluation (Kaustubh)
#
# Compare model AP scores, then run an initial incremental information-group comparison
# using feature_configurations from section 4. Keep preprocessing decisions fixed and
# reuse shared_folds. Each group's gain is conditional on the groups already included.
#
# **Explanation:** Add which model is used for the group comparison and why here.

# %%
# Add the model-score comparison and incremental information-group evaluation here.

# %% [markdown]
# **Results and interpretation:** Add AP gains, limitations and questions for feedback here.

# %% [markdown]
# ## 17. Error analysis by customer segment (Kaustubh)
#
# Choose a decision threshold using out-of-fold training predictions. A threshold converts
# risk scores into churn flags. Then count false negatives, meaning actual churners who
# were not flagged, within each tenure and spending segment. Keep test outcomes reserved.
#
# **Explanation:** Add the threshold-selection rule and segment definitions here.

# %%
# Add threshold selection and the tenure-by-spending false-negative analysis here.

# %% [markdown]
# **Results and interpretation:** Add which segments contain the most missed churners,
# their sizes and the limitations of the chosen threshold here.
