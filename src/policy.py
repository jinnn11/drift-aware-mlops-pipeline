from __future__ import annotations


def should_retrain(drifted_features: int, total_features: int, threshold: float) -> bool:
    ratio = drifted_features / max(total_features, 1)
    return ratio >= threshold
