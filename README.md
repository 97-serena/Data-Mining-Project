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

- [Data exploration](notebooks/data_exploration.py) describes the data and the inputs available to our models.
- [Reference baselines](notebooks/baselines.py) compares constant churn-rate scores, stratified random guesses, and a fixed tenure rule on our shared validation splits.
- [Logistic regression](notebooks/logistic_regression.py), [Gaussian naive Bayes](notebooks/gaussian_naive_bayes.py), [decision tree](notebooks/decision_tree.py), [random forest](notebooks/random_forest.py), and [histogram gradient boosting](notebooks/histogram_gradient_boosting.py) each compare preprocessing options at fixed starting model settings. We leave tuning for the team to complete in each notebook.
- [Model comparison](notebooks/model_comparison.py) outlines how we’ll evaluate the chosen pipelines together once we’ve agreed on their settings.

## How we compare preprocessing

We use churn as the prediction target and exclude customer IDs and both versions of the churn label from the inputs. The dataset has no event timestamps, so we also exclude retention call and offer fields whose availability at the time of outreach is unclear. We reserve the fixed 20% test split in `dataset/split.csv` for the final evaluation.

We use repeated stratified cross-validation on the training partition. We repeatedly divide it into folds with similar churn rates, take turns using each fold for validation, and fit preprocessing and the model on the other folds. All model notebooks use the same splits. We report the mean and standard deviation across folds for average precision (AP), which summarizes how well we find churners while avoiding false alarms at different cutoffs, and ROC AUC, which measures how well we rank churners ahead of customers who stay.

We load the training rows with `load_train()`, prepare the eligible inputs with `prepare_predictors()`, and take the churn labels from those same rows. In each model notebook we choose the numeric steps and the service-area encoder explicitly, then pass the assembled options to `compare_preprocessing()`. We also use `prepare_predictors()` on raw records before scoring.
