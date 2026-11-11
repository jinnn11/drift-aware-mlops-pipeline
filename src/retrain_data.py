"""Which rows a retrained candidate trains on, and which rows it is tuned and scored on.

The candidate trains on the first 70% of the historical rows (the rows the baseline in
train.py trains on) plus the older part of the newly labeled data, with the new rows
upweighted. It never trains on:

  - the last 15% of the historical rows: the historical test slice, where the candidate and
    the current model are compared on older data;
  - the newest slice of the new data: the evaluation slice, where they are compared on the
    shifted data;
  - the slice just before it: the validation slice, used only to pick the decision threshold.

So every number the promotion gate compares comes from rows neither model was fitted or tuned on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.data import time_ordered_frame_split


@dataclass
class RetrainData:
    train_df: pd.DataFrame
    sample_weights: np.ndarray
    training_mode: str
    historical_train_rows: int
    recent_train_rows: int
    historical_val: pd.DataFrame
    historical_test: pd.DataFrame
    recent_val: Optional[pd.DataFrame]
    recent_eval: Optional[pd.DataFrame]
    new_labeled_rows: int


def split_recent_labeled_data(
    df: pd.DataFrame,
    holdout_ratio: float,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered split of newly labeled rows into train | validation | evaluation."""
    eval_rows = max(300, int(len(df) * holdout_ratio))
    val_rows = max(200, int(len(df) * val_ratio))
    if len(df) <= eval_rows + val_rows:
        raise ValueError("Recent labeled dataset is too small for the requested validation and holdout sizes")
    train_end = len(df) - eval_rows - val_rows
    return df.iloc[:train_end].copy(), df.iloc[train_end:-eval_rows].copy(), df.iloc[-eval_rows:].copy()


def combine_historical_and_recent(
    historical_df: pd.DataFrame,
    recent_train_df: pd.DataFrame,
    recent_weight: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    combined_df = pd.concat([historical_df, recent_train_df], axis=0, ignore_index=True)
    weights = np.ones(len(combined_df), dtype=float)
    weights[len(historical_df):] = float(recent_weight)
    return combined_df, weights


def build_retrain_data(
    historical_df: pd.DataFrame,
    recent_df: Optional[pd.DataFrame],
    holdout_ratio: float = 0.2,
    val_ratio: float = 0.1,
    recent_weight: float = 3.0,
) -> RetrainData:
    hist_train, hist_val, hist_test = time_ordered_frame_split(historical_df)
    if recent_df is None:
        return RetrainData(
            train_df=hist_train.reset_index(drop=True),
            sample_weights=np.ones(len(hist_train), dtype=float),
            training_mode="historical_only",
            historical_train_rows=len(hist_train),
            recent_train_rows=0,
            historical_val=hist_val,
            historical_test=hist_test,
            recent_val=None,
            recent_eval=None,
            new_labeled_rows=0,
        )

    recent_train, recent_val, recent_eval = split_recent_labeled_data(recent_df, holdout_ratio, val_ratio)
    train_df, weights = combine_historical_and_recent(hist_train, recent_train, recent_weight)
    return RetrainData(
        train_df=train_df,
        sample_weights=weights,
        training_mode="historical_plus_recent_weighted",
        historical_train_rows=len(hist_train),
        recent_train_rows=len(recent_train),
        historical_val=hist_val,
        historical_test=hist_test,
        recent_val=recent_val,
        recent_eval=recent_eval,
        new_labeled_rows=len(recent_df),
    )
