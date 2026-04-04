from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold(y_true, y_proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(thresholds, dtype=float),
        where=(precision + recall) > 0,
    )
    return float(thresholds[int(np.argmax(f1_scores))])


def evaluate_probabilistic_classifier(model, x, y, threshold: float = 0.5) -> Dict[str, Any]:
    y_proba = model.predict_proba(x)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "brier": float(brier_score_loss(y, y_proba)),
        "log_loss": float(log_loss(y, y_proba, labels=[0, 1])),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
    }
