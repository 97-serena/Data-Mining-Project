# Predicting telecom customer churn

We estimate which customers may leave so we can combine churn risk with customer value when deciding whom to contact. We use the Cell2Cell dataset, which contains 51,047 labelled customers. About 29% churned.

## Run the notebooks

From the repository root, we use [uv](https://docs.astral.sh/uv/getting-started/installation/) to set up the shared Python version and dependencies.

```bash
uv venv
uv pip sync requirements.txt
source .venv/bin/activate
jupyter lab
```

On Windows PowerShell, we activate with `.\.venv\Scripts\Activate.ps1`. In JupyterLab, we open a `.py` file in `notebooks/` with **Open With → Notebook** (or **Jupytext Notebook**). In VS Code, we use the Python and Jupyter extensions, select the `.venv` kernel, and choose **Run All Cells**.

We use Jupytext’s `py:percent` format to store notebooks as plain Python files. `# %%` marks a code cell and `# %% [markdown]` marks a text cell. Running a model notebook recreates its results table.

## Notebook guide

- [Data exploration](notebooks/data_exploration.py) checks the prepared model inputs, measures linear correlations, and compares simple tenure rules on the training data.
- [Reference baselines](notebooks/baselines.py) compares constant churn-rate scores, stratified random guesses, and a fixed tenure rule on our shared validation splits.
- [Logistic regression](notebooks/logistic_regression.py), [Gaussian naive Bayes](notebooks/gaussian_naive_bayes.py), [decision tree](notebooks/decision_tree.py), [random forest](notebooks/random_forest.py), and [histogram gradient boosting](notebooks/histogram_gradient_boosting.py) each compare preprocessing options and explain which one we carry forward to tuning.
- [Model comparison](notebooks/model_comparison.py) outlines how we’ll evaluate the chosen pipelines together once we’ve agreed on their settings.

## How we choose a model

1. **Explore the training data.** We use `data.load_train()` and `preprocessing.prepare_predictors()` from our `src` package. Churn is the target. Customer IDs, both churn labels, and retention call and offer fields whose timing is unknown are excluded from the inputs.
2. **Compare preprocessing.** We hold the model settings fixed at scikit-learn defaults, with fixed random seeds and a higher iteration cap for logistic regression to let the solver converge. We use `sk_selection.cross_validate()` with five stratified folds repeated twice. Each pipeline fits its preprocessing and model on the training rows of each fold, then scores the validation rows.
3. **Lock one preprocessing option per model.** We compare mean validation average precision (AP) and use ROC AUC as a secondary measure. If mean AP is within 0.002 of the best option (0.2 percentage points), we treat the options as practically close. We compare AP on matching folds to see how often and by how much one option leads, then favor a simpler choice when scores are close. Fold standard deviations describe variation across overlapping splits.
4. **Tune the models later.** We keep each preprocessing choice fixed and compare model hyperparameters on the training partition. The preprocessing is fitted again inside each tuning fold.
5. **Evaluate the final choices.** We reserve the fixed 20% test split in `dataset/split.csv`. Once the model settings and outreach budget are fixed, we report test AP and ROC AUC, plus precision and recall at that budget.

About 28.8% of the training customers churn. We use AP because outreach starts at the top of a ranked list. AP summarizes how well we keep the share of true churners among flagged customers high as we capture more of the churners. A constant score gives AP of about 0.288. We consider the contact budget and customer value separately when deciding whom to contact.
