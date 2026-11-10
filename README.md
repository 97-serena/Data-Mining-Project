# Data Mining Project: Telecom Customer Churn

Predict whether a telecom customer will churn, and investigate:
1. Do usage, usage-change and service-quality features improve prediction beyond customer and billing features?
2. Which tenure/spending groups contain the most missed churners (false negatives)?

**Data:** Cell2Cell telecom churn dataset, `cell2celltrain.csv`: 51,047 labelled customers × 58 columns, churn rate 28.8%. The original holdout file has no labels and is not used.

## Repository structure

```
dataset/
  cell2celltrain.csv   raw data with all customers
  split.csv            no customer data, only which customers are in the 80% training set
                       and which in the 20% test set, so everyone uses the same split; do not change
src/
  data.py              load_train(): the 80% training part used for all analysis
  preprocessing.py     shared preprocessing steps, reused by the pipelines and the modeling
  features.py          EDA feature groups, engineered features and configurable group additions
notebooks/
  data_exploration.py   shared analysis as a Jupytext notebook
  preprocessing.py     data preparation and pipeline validation as a Jupytext chapter
requirements.in        direct dependencies
requirements.txt       pinned shared dependencies
.python-version        shared Python version
```

`main` only contains the agreed, final state of the project. Work in progress happens on other branches.

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. Use the
repository's `.python-version` and pinned `requirements.txt` so the team uses the same
Python version and dependency versions. Each person keeps their own `.venv`.

From a clone containing these files:

```bash
uv venv
uv pip sync requirements.txt
```

`uv` downloads the pinned Python version if it is not installed. To open JupyterLab:

```bash
source .venv/bin/activate
jupyter lab
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. Run these
commands from the repository root. In JupyterLab, right-click a file under `notebooks/`
and select **Open With → Notebook** (or **Jupytext Notebook**). If JupyterLab was already
running when Jupytext was installed, restart it.

### Text notebooks

- `notebooks/data_exploration.py`: analysis and agreed feature decisions
- `notebooks/preprocessing.py`: data preparation, feature sets and pipeline validation

These are Jupytext percent-format notebooks (`# %%` code cells and `# %% [markdown]`
text cells). Their file headers select **only** `py:percent`: saving in JupyterLab keeps
the `.py` source and does not create a paired `.ipynb`. Outputs are available during the
session but are not stored in the text file; rerun the cells to reproduce them.

For **VS Code**, install Microsoft's **Python** and **Jupyter** extensions yourself.
Open the repository folder, select its `.venv` interpreter/kernel, open a text notebook,
and use **Run Cell** or **Run All Cells**. JupyterLab and VS Code use the same source files,
environment and shared `src/` modules. The notebooks work when the kernel starts in the
repository root or `notebooks/`; no shared helper changes are needed.

### Reusing preprocessing in modeling

`src.features.get_feature_spec(groups)` describes the raw inputs and eligible engineered
features. `build_feature_configurations(baseline_groups, addition_order)` constructs nested
group lists; the modeling notebook supplies the baseline and order explicitly.

`src.preprocessing.make_preprocessor(groups, ...)` builds an unfitted pipeline. Its options
cover linear/tree preparation, both age strategies, target/one-hot ServiceArea encoding,
log/no-log numerical preparation for linear models, and the retention sensitivity set.
Use it inside the model pipeline so each cross-validation fold fits its own preprocessing.
Training data must pass through `fit_transform(X_fit, y_fit)` for internally cross-fitted
target encoding; use `transform(X_check)` for held-out rows. The pinned scikit-learn version
supports passing the reproducible inner-fold splitter directly to `TargetEncoder`.

The preprocessing notebook runs transformation checks across the five training folds. It
does not choose the group-addition order, compare model scores, or inspect test outcomes.

### Updating dependencies

Edit direct dependencies in `requirements.in`, then regenerate the shared lock from
the repository root:

```bash
uv pip compile --universal --python-version 3.12.16 requirements.in -o requirements.txt
uv pip sync requirements.txt
```

Commit both requirement files together. The universal resolution includes platform
markers for macOS, Linux and Windows; optional platform dependencies can differ while
shared packages stay pinned. Do not commit `.venv`, notebook outputs or local exports.

## Project phases

| Phase | Branch | Main files | Status |
|---|---|---|---|
| 1. Data exploration | `exploration` (archive) | `notebooks/data_exploration.py` | done |
| 2. Preprocessing | `preprocessing` | `notebooks/preprocessing.py`, `src/preprocessing.py`, `src/features.py` | done, waiting for team review before merging into `main` |
| 3. Modelling and analysis | `modelling` (based on `preprocessing`) | to be added | **current** |

`main` only receives a phase after the team has reviewed and agreed on it.

**Phase 1:** the final exploration is `notebooks/data_exploration.py`. The individual notebooks and the original Colab version are kept on the [`exploration` branch](https://github.com/97-serena/Data-Mining-Project/tree/exploration/eda) as an archive.

**Phase 2:** the preprocessing chapter, reusable transformations and validation checks are complete on `preprocessing`. If a preprocessing problem turns up during modelling, fix it on `preprocessing` and then merge it into `modelling`, so both branches keep the same preprocessing code.

**Phase 3:** we train and compare the models on `modelling`, following the team's model task list:

- baselines: a Dummy Classifier and a fixed tenure rule, neither learned from the data
- comparisons of the preprocessing options and the retention sensitivity set, settled before the model comparison
- candidate models: Logistic Regression, Naive Bayes, Decision Tree, Random Forest and Gradient Boosting
- the incremental information-group comparison and the error analysis by tenure × spending segment

All models and configurations use the same stratified five-fold splits (`StratifiedKFold(5, shuffle=True, random_state=42)`) and build their preprocessing with `make_preprocessor(...)`, so each fold fits its own preprocessing. Average Precision is the primary metric. The test set is used only once, for the final evaluation.

## Ground rules

- Use only the training split (`load_train()`); never look at the test split.
- Only edit your own files. Shared code lives in `src/`; tell the team before changing it.
- Clear notebook outputs before committing.
- Use `random_state=42` everywhere.
