"""Shared data loading and the fixed train/test split used by the whole team."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "cell2celltrain.csv"
SPLIT_PATH = ROOT / "dataset" / "split.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw():
    """All 51,047 labelled customers, plus a 0/1 target column `churn`."""
    df = pd.read_csv(DATA_PATH)
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def make_split():
    """Create dataset/split.csv (CustomerID -> train/test). Run once; the file is committed."""
    df = load_raw()
    train_ids, test_ids = train_test_split(
        df["CustomerID"], test_size=TEST_SIZE, stratify=df["churn"], random_state=RANDOM_STATE)
    split = pd.concat([pd.DataFrame({"CustomerID": train_ids, "split": "train"}),
                       pd.DataFrame({"CustomerID": test_ids, "split": "test"})])
    split.sort_values("CustomerID").to_csv(SPLIT_PATH, index=False)


def load_train():
    """The 80% training part. Use only this for exploration and preprocessing decisions."""
    df = load_raw().merge(pd.read_csv(SPLIT_PATH), on="CustomerID")
    return df[df["split"] == "train"].drop(columns="split").reset_index(drop=True)


if __name__ == "__main__":
    make_split()
