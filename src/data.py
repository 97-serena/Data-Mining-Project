"""load data and create the fixed train/test split"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "cell2celltrain.csv"
SPLIT_PATH = ROOT / "dataset" / "split.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw():
    """load all labelled customers with a binary churn column"""
    df = pd.read_csv(DATA_PATH)
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def make_split():
    """save stratified customer assignments to the split file

    running this again overwrites the existing split
    """
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
    """load training customers with the original and binary churn labels"""
    df = load_raw()
    split = pd.read_csv(SPLIT_PATH)
    df = df.merge(split, on="CustomerID")
    train = df[df["split"] == "train"]
    train = train.drop(columns="split")
    return train.reset_index(drop=True)


if __name__ == "__main__":
    make_split()
