from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

TARGET_COLUMN = "y"
LEAKAGE_COLUMNS = ["duration"]

EXPECTED_COLUMNS = {
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
}


@dataclass
class SplitData:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_dataset(path: str, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


def validate_dataset(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    validate_dataset(df)
    work_df = df.copy()

    work_df[TARGET_COLUMN] = work_df[TARGET_COLUMN].map({"no": 0, "yes": 1})
    if work_df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains values outside {no, yes}")

    x = work_df.drop(columns=[TARGET_COLUMN] + LEAKAGE_COLUMNS)
    y = work_df[TARGET_COLUMN].astype(int)
    return x, y


def time_ordered_split(
    x: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SplitData:
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of rows")

    n = len(x)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return SplitData(
        x_train=x.iloc[:train_end].copy(),
        x_val=x.iloc[train_end:val_end].copy(),
        x_test=x.iloc[val_end:].copy(),
        y_train=y.iloc[:train_end].copy(),
        y_val=y.iloc[train_end:val_end].copy(),
        y_test=y.iloc[val_end:].copy(),
    )
