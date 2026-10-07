"""Lending Club loans as a time-ordered stream.

Source: Zenodo record 11295916 (CC-BY-4.0), a cleaned cut of the Kaggle
"wordsforthewise/lending-club" accepted-loans file that keeps only fields known at
application time and only loans that reached a final state (Fully Paid / Charged Off).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ZENODO_URL = "https://zenodo.org/api/records/11295916/files/LC_loans_granting_model_dataset.csv/content"

TARGET_COLUMN = "default"
MONTH_COLUMN = "issue_month"

NUMERIC_FEATURES = ["annual_income", "dti", "loan_amnt", "fico"]
CATEGORICAL_FEATURES = ["emp_length", "purpose", "home_ownership", "addr_state"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_RENAME = {
    "revenue": "annual_income",
    "dti_n": "dti",
    "fico_n": "fico",
    "home_ownership_n": "home_ownership",
    "Default": TARGET_COLUMN,
}


def load_lending_club(path: str) -> pd.DataFrame:
    """Load the raw CSV and return features, target and an issue-month period column.

    Dropped on purpose: `zip_code` (high-cardinality proxy for `addr_state`), `title` and
    `desc` (free text), `experience_c` (constant: 25 of 1.35M rows differ). Rows with
    DTI above 100 are removed: 533 rows, including a 999 sentinel used for missing values.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Download it with: curl -L -o {path} '{ZENODO_URL}'")

    raw = pd.read_csv(
        path,
        usecols=["issue_d", "revenue", "dti_n", "loan_amnt", "fico_n", "emp_length", "purpose",
                 "home_ownership_n", "addr_state", "Default"],
    )
    df = raw.rename(columns=_RENAME)
    df = df[df["dti"] <= 100].copy()
    df[MONTH_COLUMN] = pd.to_datetime(df.pop("issue_d"), format="%b-%Y").dt.to_period("M")
    df["loan_amnt"] = df["loan_amnt"].astype(float)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    return df.sort_values(MONTH_COLUMN, kind="stable").reset_index(drop=True)


def monthly_stream(
    df: pd.DataFrame,
    start: str,
    end: str,
    max_rows_per_month: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Restrict to [start, end] (inclusive, 'YYYY-MM') and optionally cap each month by random sampling.

    Capping keeps months comparable: monthly volume grows ~25x from 2011 to 2015 and
    drift-test power grows with sample size, so uncapped months would confound
    "more drift" with "more rows".
    """
    lo, hi = pd.Period(start, "M"), pd.Period(end, "M")
    out = df[(df[MONTH_COLUMN] >= lo) & (df[MONTH_COLUMN] <= hi)]
    if max_rows_per_month:
        rng = np.random.default_rng(seed)
        parts = []
        for _, group in out.groupby(MONTH_COLUMN, sort=True):
            if len(group) > max_rows_per_month:
                group = group.iloc[np.sort(rng.choice(len(group), max_rows_per_month, replace=False))]
            parts.append(group)
        out = pd.concat(parts)
    return out.reset_index(drop=True)


def months_between(start: str, end: str) -> List[pd.Period]:
    return list(pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M"))
