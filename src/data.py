"""Read Cell2Cell records and the project's fixed train/test assignments."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "cell2celltrain.csv"
SPLIT_PATH = ROOT / "dataset" / "split.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw():
    """Load all labelled customers with the original and binary churn labels."""
    return _with_churn_label(pd.read_csv(DATA_PATH))


def _with_churn_label(df):
    df["churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    if df["churn"].isna().any():
        raise ValueError("Churn must contain only Yes and No labels")
    return df


def make_split():
    """Overwrite the split file with reproducible, stratified customer assignments."""
    df = load_raw()
    train_ids, test_ids = train_test_split(
        df["CustomerID"],
        test_size=TEST_SIZE,
        stratify=df["churn"],
        random_state=RANDOM_STATE,
    )
    train_split = pd.DataFrame({"CustomerID": train_ids, "split": "train"})
    test_split = pd.DataFrame({"CustomerID": test_ids, "split": "test"})
    split = pd.concat([train_split, test_split])
    split = split.sort_values("CustomerID")
    split.to_csv(SPLIT_PATH, index=False)


def load_train():
    """Load only training customers in dataset order, with both churn labels."""
    split = pd.read_csv(SPLIT_PATH)
    training_ids = split.loc[split["split"] == "train", "CustomerID"]
    raw = pd.read_csv(DATA_PATH)
    train = raw.loc[raw["CustomerID"].isin(training_ids)].reset_index(drop=True)
    return _with_churn_label(train)


def load_training_data():
    """Return eligible predictors and binary churn labels from the fixed training partition."""
    from src.features import PREDICTOR_COLUMNS

    train = load_train()
    return train.loc[:, list(PREDICTOR_COLUMNS)], train["churn"]


if __name__ == "__main__":
    make_split()
