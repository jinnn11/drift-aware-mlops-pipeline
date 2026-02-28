from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


DEFAULT_PARAM_GRID = {
    "model__n_estimators": [200, 350],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}



def build_xgboost_pipeline(preprocessor) -> Pipeline:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_with_cv(
    x_train: pd.DataFrame,
    y_train,
    preprocessor,
    param_grid: Dict | None = None,
    cv_splits: int = 3,
    sample_weight=None,
) -> Tuple[Pipeline, Dict, float]:
    pipeline = build_xgboost_pipeline(preprocessor)
    grid = param_grid or DEFAULT_PARAM_GRID
    cv = TimeSeriesSplit(n_splits=cv_splits)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    search.fit(x_train, y_train, **fit_kwargs)
    return search.best_estimator_, search.best_params_, float(search.best_score_)
