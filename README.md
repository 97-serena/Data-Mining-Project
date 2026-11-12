# Telecom Customer Churn

This project uses data mining to predict customer churn in the Cell2Cell dataset. It also looks at how different groups of information affect prediction and which churners are missed across tenure and spending levels. The dataset contains 51,047 labelled customers, 28.8% of whom churned.

## Getting started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run the following from the project root:

```bash
uv venv
uv pip sync requirements.txt
source .venv/bin/activate
jupyter lab
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead.

The notebooks are stored as `.py` files. Open them as Jupytext notebooks in JupyterLab, or run the cells in VS Code with the Python and Jupyter extensions. Notebook outputs and `.ipynb` exports are not included.

## Project structure

| Path | Description |
| --- | --- |
| `dataset/split.csv` | Fixed 80/20 train/test split. |
| `src/data.py` | Data loading, including `load_train()` for the training partition. |
| `src/preprocessing.py`, `src/features.py` | Preprocessing and feature groups. |
| `src/evaluation.py`, `src/baselines.py` | Five-fold evaluation and the tenure baseline. |
| `notebooks/data_exploration.py` | Exploratory analysis. |
| `notebooks/preprocessing.py` | Preprocessing notes and transformation checks. |
| `notebooks/modelling.py` | Evaluation, baselines, modelling and analysis. |

## Evaluation

Models are evaluated using stratified five-fold cross-validation with shuffling and seed 42. Average Precision (AP) is the primary metric. Preprocessing is fitted within each fold to avoid data leakage. The information baseline and the order in which feature groups are added are defined in `notebooks/modelling.py`.

The example below evaluates a dummy classifier on the training data:

```python
from sklearn.dummy import DummyClassifier
from src.data import load_train
from src.evaluation import make_shared_folds, evaluate_classifier, summarize_scores

train = load_train()
X = train.drop(columns=['CustomerID', 'Churn', 'churn'])
y = train['churn']
folds = make_shared_folds(X, y)
scores, predictions = evaluate_classifier(
    DummyClassifier(strategy='prior'), X, y, folds, name='Dummy (prior)',
)
print(summarize_scores(scores))
```

For other estimators, `make_model_pipeline(estimator, groups, **options)` in `src.evaluation` creates a pipeline that can be passed to `evaluate_classifier`. The test set is reserved for final evaluation.

## Tests and dependencies

Run the tests with:

```bash
python -m unittest discover -s tests
```

To update and sync the dependency lock:

```bash
uv pip compile --universal --python-version 3.12.16 requirements.in -o requirements.txt
uv pip sync requirements.txt
```
