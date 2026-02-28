from __future__ import annotations

from typing import Dict

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score



def evaluate_probabilistic_classifier(model, x, y) -> Dict[str, float]:
    y_proba = model.predict_proba(x)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "brier": float(brier_score_loss(y, y_proba)),
    }
