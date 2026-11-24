"""Read Cell2Cell records and the project's fixed train/test assignments."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "cell2celltrain.csv"
SPLIT_PATH = ROOT / "dataset" / "split.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw() -> pd.DataFrame:
    """Load all labelled customers with the original and binary churn labels."""
    return _with_churn_label(pd.read_csv(DATA_PATH))


def _with_churn_label(customers: pd.DataFrame) -> pd.DataFrame:
    churn = customers["Churn"].map({"No": 0, "Yes": 1})
    if churn.isna().any():
        raise ValueError("Churn must contain only Yes and No labels")
    return customers.assign(churn=churn)


def make_split() -> None:
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
    assignments = pd.concat([train_split, test_split]).sort_values("CustomerID")
    assignments.to_csv(SPLIT_PATH, index=False)


def load_train() -> pd.DataFrame:
    """Return labelled training rows in dataset order.

    The committed split must assign every customer to exactly one partition.
    Holding the row order fixed keeps validation splits reproducible.
    """
    return _load_partition("train")


def load_test() -> pd.DataFrame:
    """Return the reserved test rows in dataset order.

    Use this only once, for the final evaluation of the frozen pipelines.
    """
    return _load_partition("test")


def _load_partition(partition: str) -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)
    assignments = pd.read_csv(SPLIT_PATH, usecols=["CustomerID", "split"])
    if raw["CustomerID"].isna().any() or assignments["CustomerID"].isna().any():
        raise ValueError("Customer IDs must be present in both the dataset and the split file")

    customers = raw.merge(assignments, on="CustomerID", how="left", validate="one_to_one")
    if len(assignments) != len(raw) or set(customers["split"]) != {"train", "test"}:
        raise ValueError("The split file must assign every customer to train or test")

    rows = customers.loc[customers["split"] == partition, raw.columns].reset_index(drop=True)
    return _with_churn_label(rows)


if __name__ == "__main__":
    make_split()
