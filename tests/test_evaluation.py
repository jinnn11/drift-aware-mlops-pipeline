import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.evaluation import evaluate_probabilistic_classifier, find_optimal_threshold


class DummyModel:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, x):
        n = len(x)
        probs = self._probs[:n]
        return np.column_stack([1.0 - probs, probs])


def test_find_optimal_threshold_maximizes_f1_on_pr_thresholds():
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.35, 0.8])

    threshold = find_optimal_threshold(y_true, y_proba)

    assert threshold == pytest.approx(0.35)


def test_evaluate_probabilistic_classifier_returns_extended_metrics():
    y_true = pd.Series([0, 0, 1, 1])
    x = pd.DataFrame({"feature": [1, 2, 3, 4]})
    model = DummyModel([0.1, 0.9, 0.8, 0.2])

    metrics = evaluate_probabilistic_classifier(model, x, y_true, threshold=0.5)

    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, [0.1, 0.9, 0.8, 0.2]))
    assert metrics["pr_auc"] == pytest.approx(average_precision_score(y_true, [0.1, 0.9, 0.8, 0.2]))
    assert metrics["brier"] == pytest.approx(brier_score_loss(y_true, [0.1, 0.9, 0.8, 0.2]))
    assert metrics["log_loss"] == pytest.approx(log_loss(y_true, [0.1, 0.9, 0.8, 0.2], labels=[0, 1]))
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["threshold"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_compute_scale_pos_weight_uses_negative_to_positive_ratio():
    pytest.importorskip("xgboost")
    from src.modeling import compute_scale_pos_weight

    y_train = pd.Series([0, 0, 0, 1, 1])

    assert compute_scale_pos_weight(y_train) == pytest.approx(1.5)
