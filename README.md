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
data_exploration.py    final exploration as a Jupytext notebook
preprocessing.py       preprocessing steps as a Jupytext notebook, one section per task
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
commands from the repository root. In JupyterLab, right-click `data_exploration.py` or
`preprocessing.py` and select **Open With → Notebook** (or **Jupytext Notebook**). If JupyterLab was already
running when Jupytext was installed, restart it.

### Text notebooks

- `data_exploration.py`: final exploration and feature decisions
- `preprocessing.py`: the preprocessing tasks, one section per task with its owner

These are Jupytext percent-format notebooks (`# %%` code cells and `# %% [markdown]`
text cells). Their file headers select **only** `py:percent`: saving in JupyterLab keeps
the `.py` source and does not create a paired `.ipynb`. Outputs are available during the
session but are not stored in the text file; rerun the cells to reproduce them.

For **VS Code**, install Microsoft's **Python** and **Jupyter** extensions yourself.
Open the repository folder, select its `.venv` interpreter/kernel, open a text notebook,
and use **Run Cell** or **Run All Cells**. JupyterLab and VS Code use the same source files,
environment and shared `src/` modules. Start the kernel in the repository root.

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

| Phase | Branch | Status |
|---|---|---|
| 1. Data exploration | `exploration` (archive), final version in `data_exploration.py` | done |
| 2. Agreed preprocessing | `preprocessing` → merged into `main` | **current** |
| 3. Modelling and analysis | to be decided | later |

**Phase 1:** the final exploration is `data_exploration.py`. The individual notebooks and the original Colab version stay on the [`exploration` branch](https://github.com/97-serena/Data-Mining-Project/tree/exploration/eda) as an archive.

**Phase 2:** we work through the team's preprocessing task list in `preprocessing.py` on `preprocessing`, and then merge the result into `main`. Section 8 of `data_exploration.py` holds the detailed decisions; differences from the task list are marked as open questions in `preprocessing.py`.

## Ground rules

- Use only the training split (`load_train()`); never look at the test split.
- Only edit your own files. Shared code lives in `src/`; tell the team before changing it.
- Clear notebook outputs before committing.
- Use `random_state=42` everywhere.
