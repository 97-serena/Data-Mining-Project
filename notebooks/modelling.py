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
# and retention sections prepare comparisons using Shiqi's Logistic Regression implementation.
#
# Separate sections below provide space for the team's model code, explanations and
# interpretation of results. At present, only the three baselines are evaluated.

# %% [markdown]
# ## 1. Shared imports
#
# We locate the project and import the reusable evaluation code. Like the earlier
# chapters, this text notebook works from the project root or the notebooks folder.

# %%
from pathlib import Path
import sys

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

from src.baselines import TenureRule
from src.data import load_train
from src.evaluation import evaluate_classifier, make_shared_folds, summarize_scores
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
# All switches already exist in make_preprocessor. The next experiment will compare
# eight combinations on the full main information set, using the shared folds and
# Shiqi's fixed Logistic Regression implementation. Regularisation penalises overly
# large model weights; its settings must stay the same across these comparisons so
# we measure preprocessing changes rather than model-setting changes.
#
# - Age keep: preserve recorded zeros and their two zero flags. Age median: replace
#   zeros with training-fold positive-age medians and omit those flags. Both options
#   still impute genuine missing ages from the training fold.
# - One-hot ServiceArea: a separate 0/1 column for each area. Target encoding: a compact
#   churn-rate estimate stabilised for rare areas, using the existing cross-fitting.
# - log1p: log(1 + x) compresses large nonnegative measurements. The existing pipeline
#   chooses eligible skewed columns inside each fold; signed values are not logged.
#
# We list the configurations now without fitting Logistic Regression or choosing a
# winner. After evaluation, inspect mean AP and fold-by-fold differences, prefer a
# simpler representation when gains are small and inconsistent, and record the chosen
# age, area and log options here. Freeze those choices before information-group scoring.

# %%
preprocessing_options = []
for age_strategy in ['keep', 'median']:
    for area_encoding in ['target', 'onehot']:
        for use_log in [True, False]:
            preprocessing_options.append({
                'age_strategy': age_strategy,
                'service_area_encoding': area_encoding,
                'log_transform': use_log,
                'include_retention': False,
            })

assert len(preprocessing_options) == 8
display(pd.DataFrame(preprocessing_options))

# %% [markdown]
# ## 10. Retention sensitivity
#
# A sensitivity analysis asks whether results change under an alternative input set.
# After fixing preprocessing, repeat the Logistic Regression evaluation with the three
# original retention fields added together. Use the same full main groups, model settings
# and folds; only include_retention changes. Report AP with and without retention and
# the per-fold differences. Do not retune preprocessing separately for this comparison.
#
# Keep retention outside the main information-group comparison because its availability
# at prediction time is uncertain. An AP gain would not establish leakage, availability
# before prediction, or a causal effect of retention offers on churn.
#
# The following table checks the two raw input sets. It does not evaluate a model.

# %%
all_groups = list(MAIN_FEATURE_GROUPS)
main_spec = get_feature_spec(all_groups)
retention_spec = get_feature_spec(all_groups, include_retention=True)
assert len(main_spec['raw_features']) == 53
assert len(retention_spec['raw_features']) == 56
assert set(retention_spec['raw_features']) - set(main_spec['raw_features']) == set(RETENTION_COLUMNS)

display(pd.DataFrame({
    'representation': ['main', 'retention sensitivity'],
    'raw_predictors': [53, 56],
    'include_retention': [False, True],
}))

# %% [markdown]
# ## 11. Logistic Regression (Shiqi)
#
# Fit an initial regularised Logistic Regression model using the selected preprocessing
# and the shared evaluation procedure. Report AP on the same five folds as the baselines.
#
# **Explanation:** Add the model settings and reasons for choosing them here.

# %%
# Add the Logistic Regression pipeline and shared-fold evaluation here.

# %% [markdown]
# **Results and interpretation:** Add the AP summary and comparison with the baselines here.

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
