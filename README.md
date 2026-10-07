# Data Mining Project: Telecom Customer Churn

Predict whether a telecom customer will churn, and investigate:
1. Do usage, usage-change and service-quality features improve prediction beyond customer and billing features?
2. Which tenure/spending groups contain the most missed churners (false negatives)?

**Data:** Cell2Cell telecom churn dataset, `cell2celltrain.csv`: 51,047 labelled customers × 58 columns, churn rate 28.8%. The original holdout file has no labels and is not used.

The data file is not stored in this repository. Download `cell2celltrain.csv` from our [Google Drive folder](https://drive.google.com/drive/folders/1oVlHRakuFTE_EgKDbQ75KSdIqoTyB_IR) and put it into the `dataset/` folder.

## Repository structure (main)

```
dataset/
  cell2celltrain.csv   raw data (download from Google Drive, not in the repo)
  split.csv            fixed 80/20 train/test split (CustomerID -> train/test), do not change
src/
  data.py              load_train(): the 80% training part used for all analysis
requirements.txt
```

`main` only contains the agreed, final state of the project. Work in progress happens on other branches.

## Setup

```bash
git clone https://github.com/97-serena/Data-Mining-Project.git
cd Data-Mining-Project
pip install -r requirements.txt
```

Then download `cell2celltrain.csv` from Google Drive into `dataset/`.

## Project phases and branches

| Phase | Branch | Status |
|---|---|---|
| 1. Individual data exploration | `exploration` | **current** |
| 2. Agreed preprocessing | `preprocessing` → merged into `main` | next |
| 3. Modelling and analysis | to be decided | later |

### Phase 1: individual data exploration (current)

This phase is our preparatory exploration of the dataset. Everyone goes through **all 56 features** independently, so that we all know the data before writing the proposal.

The `exploration` branch contains the material for this phase:
- `eda/eda_template.ipynb`: template notebook (feature groups, cross-group checks, decision table)
- `src/eda_utils.py`: helper functions used by the template
- `eda/summaries/`: one decision table (CSV) per person

Everyone works directly on the `exploration` branch. This does not cause conflicts because each person only adds their own files.

Steps:
1. `git checkout exploration`
2. Copy `eda/eda_template.ipynb` to `eda/eda_<yourname>.ipynb` and work only in your copy.
3. Run it section by section and write your findings under each section.
4. Fill in the decision table at the end; it is saved to `eda/summaries/eda_summary_<yourname>.csv`.
5. Clear outputs, then upload your two files:
   ```bash
   git pull
   git add eda/eda_<yourname>.ipynb eda/summaries/eda_summary_<yourname>.csv
   git commit -m "EDA <yourname>"
   git push
   ```
   Always `git pull` before `git push`, otherwise the push is rejected when someone else pushed first.

Do not edit the template, `src/` or other people's files. The `exploration` branch is **not** merged into `main`.

### Phase 2: agreed preprocessing (next)

After everyone has finished, we compare the decision tables and agree on one preprocessing plan. It is implemented once on a `preprocessing` branch and merged into `main`. After that, the `exploration` branch is deleted.

## Ground rules

- Use only the training split (`load_train()`); never look at the test split.
- Only edit your own notebook. Shared code lives in `src/`; tell the team before changing it.
- Clear notebook outputs before committing.
- Use `random_state=42` everywhere.
