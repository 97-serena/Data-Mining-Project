# Data Mining Project: Telecom Customer Churn

Predict whether a telecom customer will churn, and investigate:
1. Do usage, usage-change and service-quality features improve prediction beyond customer and billing features?
2. Which tenure/spending groups contain the most missed churners (false negatives)?

**Data:** Cell2Cell telecom churn dataset, `cell2celltrain.csv`: 51,047 labelled customers × 58 columns, churn rate 28.8%. The original holdout file has no labels and is not used. The data file is not stored in this repository; download it from our [Google Drive folder](https://drive.google.com/drive/folders/1oVlHRakuFTE_EgKDbQ75KSdIqoTyB_IR).

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

Then put `cell2celltrain.csv` from Google Drive into `dataset/`.

## Project phases

| Phase | Branch | Status |
|---|---|---|
| 1. Individual data exploration | `exploration` | **current** (deadline: 11 October, 10:00 German time) |
| 2. Agreed preprocessing | `preprocessing` → merged into `main` | next |
| 3. Modelling and analysis | to be decided | later |

**Phase 1:** everyone explores all 56 features independently and fills in a decision table. Instructions: [`eda/README.md` on the `exploration` branch](https://github.com/97-serena/Data-Mining-Project/tree/exploration/eda).

**Phase 2:** we compare the decision tables, agree on one preprocessing plan, implement it on `preprocessing` and merge it into `main`. The `exploration` branch is then deleted.

## Ground rules

- Use only the training split (`load_train()`); never look at the test split.
- Only edit your own files. Shared code lives in `src/`; tell the team before changing it.
- Clear notebook outputs before committing.
- Use `random_state=42` everywhere.
