"""Load the labeled data and make the single reproducible validation split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TRAIN_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.csv"
RANDOM_STATE = 42


def load_train(path: Path = TRAIN_PATH) -> pd.DataFrame:
    """Read the original labeled CSV without changing its values."""
    return pd.read_csv(path)


def split_train_validation(frame: pd.DataFrame):
    """Return X_train, X_valid, y_train, y_valid from an 80/20 stratified split."""
    predictors = frame.drop(columns=["id", "Response"])
    target = frame["Response"]
    return train_test_split(
        predictors,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
