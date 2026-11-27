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
# using each estimator's default training hyperparameters, with a higher
# iteration cap for logistic regression. We fix random seeds where applicable
# and fit each option inside each fold.
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
# | [CatBoost](catboost_classifier.py) | Native | Native missing values |
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
# We report AP and ROC AUC, and precision, recall, lift, the confusion matrix
# and the savings from outreach at our contact budget.
#
# The settings below must be fixed from training results only. The test cells
# stop with an error until we set `RUN_TEST_EVALUATION = True`, so nobody
# opens the test set by accident. We run them once and do not change any
# decision afterward.

# %%
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from IPython.display import display
from sklearn import base as sk_base
from sklearn import compose as sk_compose
from sklearn import dummy as sk_dummy
from sklearn import ensemble as sk_ensemble
from sklearn import linear_model as sk_linear
from sklearn import metrics as sk_metrics
from sklearn import model_selection as sk_selection
from sklearn import naive_bayes as sk_naive_bayes
from sklearn import pipeline as sk_pipeline
from sklearn import preprocessing as sk_preprocessing
from sklearn import tree as sk_tree

project_root = Path.cwd()
if not (project_root / "src" / "data.py").is_file():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src import baselines, data, evaluation, preprocessing
from src.features import CATEGORICAL_COLUMNS, SERVICE_AREA

train = data.load_train()
X = preprocessing.prepare_predictors(train)
y = train["churn"]
N_JOBS = 4

splitter = sk_selection.RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=data.RANDOM_STATE,
)

# %% [markdown]
# ## Decisions to confirm before the test set
#
# The contact budget, the cost assumptions and the preferred model are still
# open. The values below are placeholders until the team agrees on them.

# %%
CONTACT_SHARE = 0.10  # TO CONFIRM: share of customers we contact
CONTACT_SHARES = [0.05, 0.10, 0.20]  # smaller and larger budgets for comparison
CONTACT_COST = 20  # TO CONFIRM: cost of one contact
CHURN_LOSS = 12 * train["MonthlyRevenue"].median()  # TO CONFIRM: loss per churner
SAVE_RATE = 0.20  # TO CONFIRM: share of contacted churners who stay
PREFERRED_MODEL = "Histogram gradient boosting"  # TO CONFIRM from validation results

RUN_TEST_EVALUATION = False

# %% [markdown]
# ## Final pipelines
#
# Each pipeline uses the preprocessing and hyperparameters chosen in its model
# notebook. CatBoost carries forward its default training settings after
# comparing them with the tuning grid. We fit the pipelines on the full
# training partition.

# %%
final_pipelines = {
    "Logistic regression": sk_pipeline.make_pipeline(
        preprocessing.combine_feature_steps(
            numeric=sk_pipeline.make_pipeline(
                preprocessing.LogSkewedNumeric(),
                preprocessing.numeric_median_imputer(),
                sk_preprocessing.StandardScaler(),
            ),
            service_area=preprocessing.target_encode_service_area(),
        ),
        sk_linear.LogisticRegression(max_iter=2000, C=0.0015),
    ),
    "Gaussian naive Bayes": sk_pipeline.make_pipeline(
        preprocessing.combine_feature_steps(
            numeric=sk_pipeline.make_pipeline(
                preprocessing.LogSkewedNumeric(),
                preprocessing.numeric_median_imputer(),
            ),
            service_area=preprocessing.ServiceAreaFrequencyEncoder(),
        ).set_params(sparse_threshold=0.0),
        sk_naive_bayes.GaussianNB(var_smoothing=5e-7),
    ),
    "Decision tree": sk_pipeline.make_pipeline(
        preprocessing.combine_feature_steps(
            numeric=preprocessing.numeric_median_imputer(),
            service_area=preprocessing.ServiceAreaFrequencyEncoder(),
        ),
        sk_tree.DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=250, random_state=data.RANDOM_STATE
        ),
    ),
    "Random forest": sk_pipeline.make_pipeline(
        preprocessing.combine_feature_steps(
            numeric=preprocessing.numeric_median_imputer(),
            service_area=preprocessing.ServiceAreaFrequencyEncoder(),
        ).set_params(sparse_threshold=0.0),
        sk_ensemble.RandomForestClassifier(
            max_features=0.2,
            min_samples_leaf=20,
            random_state=data.RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
    ),
    "Histogram gradient boosting": sk_pipeline.make_pipeline(
        preprocessing.combine_feature_steps(
            numeric="passthrough",
            service_area=preprocessing.ServiceAreaFrequencyEncoder(),
        ).set_params(sparse_threshold=0.0),
        sk_ensemble.HistGradientBoostingClassifier(
            learning_rate=0.10,
            max_iter=200,
            max_leaf_nodes=15,
            early_stopping=False,
            random_state=data.RANDOM_STATE,
        ),
    ),
    "CatBoost": sk_pipeline.make_pipeline(
        # Match the CatBoost notebook's feature order and keep column names.
        sk_compose.ColumnTransformer(
            [("service_area", "passthrough", [SERVICE_AREA])],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas"),
        CatBoostClassifier(
            cat_features=(*CATEGORICAL_COLUMNS, SERVICE_AREA),
            random_seed=data.RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=N_JOBS,
        ),
    ),
}
reference_baselines = {
    "Constant churn rate": sk_dummy.DummyClassifier(strategy="prior"),
    "Fixed tenure (11–12 months)": baselines.TenureRule(),
}

for pipeline in [*final_pipelines.values(), *reference_baselines.values()]:
    pipeline.fit(X, y)


def churn_scores(estimator, features):
    """Return scores that rank customers by churn risk."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    return estimator.decision_function(features)


def score_customers(y_true, scores):
    """Return the ranking and outreach scores we compare across data sets."""
    outreach_at_budget = evaluation.outreach_scores(y_true, scores, CONTACT_SHARE)
    return {
        "AP": sk_metrics.average_precision_score(y_true, scores),
        "ROC AUC": sk_metrics.roc_auc_score(y_true, scores),
        "Precision": outreach_at_budget["Precision"],
        "Recall": outreach_at_budget["Recall"],
        "Savings per 1,000 customers": evaluation.savings_per_1000(
            outreach_at_budget, cost_matrix
        ),
    }


cost_matrix = evaluation.cost_matrix(CONTACT_COST, CHURN_LOSS, SAVE_RATE)

# %% [markdown]
# ## Validation scores
#
# We score every final pipeline on the same ten validation folds used in the
# model notebooks. These scores use only the training partition, so we use
# them to choose the preferred model. Precision, recall and savings use the
# contact budget above.

# %%
validation_rows = []
for fold, (fit_rows, valid_rows) in enumerate(splitter.split(X, y)):
    for name, pipeline in final_pipelines.items():
        fold_pipeline = sk_base.clone(pipeline).fit(X.iloc[fit_rows], y.iloc[fit_rows])
        scores = churn_scores(fold_pipeline, X.iloc[valid_rows])
        validation_rows.append(
            {"Model": name, "Fold": fold, **score_customers(y.iloc[valid_rows], scores)}
        )

validation = pd.DataFrame(validation_rows).groupby("Model", sort=False).mean().drop(columns="Fold")
print(f"Mean over ten validation folds, contacting the top {CONTACT_SHARE:.0%}")
display(validation.sort_values("AP", ascending=False).round(4))

# %% [markdown]
# ## Test set

# %%
if RUN_TEST_EVALUATION:
    raise RuntimeError(
        "Confirm the decisions above, then set RUN_TEST_EVALUATION = True to use the test set once."
    )

test = data.load_test()
X_test = preprocessing.prepare_predictors(test)
y_test = test["churn"]
test_scores = {
    name: churn_scores(estimator, X_test)
    for name, estimator in {**final_pipelines, **reference_baselines}.items()
}
print(f"{len(y_test):,} test customers, churn rate {y_test.mean():.1%}")

# %% [markdown]
# ### Ranking quality

# %%
ranking = pd.DataFrame(
    {
        name: {
            "AP": sk_metrics.average_precision_score(y_test, scores),
            "ROC AUC": sk_metrics.roc_auc_score(y_test, scores),
        }
        for name, scores in test_scores.items()
    }
).T
display(ranking.round(4))

# %% [markdown]
# ### Outreach at the contact budget
#
# We contact the given share of test customers with the highest scores.
# Random contact reaches the test churn rate as precision, so its lift is 1.

# %%
print("Cost per customer")
display(cost_matrix.round(2))

outreach = pd.DataFrame(
    [
        {
            "Model": name,
            "Contact share": share,
            **evaluation.outreach_scores(y_test, test_scores[name], share),
        }
        for name in final_pipelines
        for share in CONTACT_SHARES
    ]
)
outreach["Savings per 1,000 customers"] = evaluation.savings_per_1000(outreach, cost_matrix)

columns = ["Precision", "Recall", "Lift", "Savings per 1,000 customers"]
at_budget = outreach.loc[outreach["Contact share"].eq(CONTACT_SHARE)].set_index("Model")[columns]
at_budget.loc["Random contact"] = [y_test.mean(), CONTACT_SHARE, 1.0, None]
print(f"Contacting the top {CONTACT_SHARE:.0%}")
display(at_budget.round(4))

print("Precision by contact share")
display(outreach.pivot(index="Model", columns="Contact share", values="Precision").round(4))

# %%
contacted = evaluation.contact_top(test_scores[PREFERRED_MODEL], CONTACT_SHARE)
confusion = pd.crosstab(
    y_test.map({1: "Churned", 0: "Stayed"}).rename("Actual"),
    pd.Series(contacted, index=y_test.index)
    .map({True: "Contacted", False: "Not contacted"})
    .rename("Outreach"),
)
print(f"{PREFERRED_MODEL}, top {CONTACT_SHARE:.0%} of test customers")
display(confusion)

# %% [markdown]
# ### Validation and test scores
#
# We place each model's mean validation score next to its test score. Similar
# values mean the validation results carried over to new customers. A clearly
# lower test score means our choices fit the validation folds too closely. We
# report the gap and do not change any decision because of it.

# %%
test_results = pd.DataFrame(
    {name: score_customers(y_test, test_scores[name]) for name in final_pipelines}
).T
for metric in ["AP", "ROC AUC", "Precision"]:
    gap = pd.DataFrame(
        {
            "Validation": validation[metric],
            "Test": test_results[metric],
        }
    )
    gap["Test − validation"] = gap["Test"] - gap["Validation"]
    print(metric if metric != "Precision" else f"Precision in the top {CONTACT_SHARE:.0%}")
    display(gap.round(4))

# %%
