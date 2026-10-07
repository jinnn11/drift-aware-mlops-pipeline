import numpy as np
import pandas as pd
import pytest

from src.detectors import (
    CalibrationDetector,
    DomainClassifierDetector,
    KSChi2Detector,
    PSIDetector,
    population_stability_index,
)

NUMERIC, CATEGORICAL = ["a", "b"], ["c"]


def _frame(n, rng, shift=0.0, cat_p=(0.5, 0.3, 0.2)):
    return pd.DataFrame({
        "a": rng.normal(shift, 1.0, n),
        "b": rng.normal(0.0, 1.0, n),
        "c": rng.choice(["x", "y", "z"], n, p=cat_p),
    })


@pytest.mark.parametrize("factory", [
    lambda: KSChi2Detector(NUMERIC, CATEGORICAL),
    lambda: PSIDetector(NUMERIC, CATEGORICAL, threshold=0.1),
    lambda: DomainClassifierDetector(NUMERIC, CATEGORICAL, seed=0),
])
def test_input_detectors_quiet_on_same_distribution_and_fire_on_shift(factory):
    rng = np.random.default_rng(0)
    reference = _frame(3000, rng)
    detector = factory().fit(reference)

    assert not detector.test(_frame(3000, rng)).alarm
    assert detector.test(_frame(3000, rng, shift=0.5)).alarm


def test_ks_chi2_flags_categorical_shift():
    rng = np.random.default_rng(1)
    detector = KSChi2Detector(NUMERIC, CATEGORICAL).fit(_frame(3000, rng))

    result = detector.test(_frame(3000, rng, cat_p=(0.3, 0.3, 0.4)))

    assert result.alarm
    assert result.details["p_c"] < 1e-6


def test_psi_of_identical_distributions_is_zero():
    p = np.array([0.2, 0.3, 0.5])

    assert population_stability_index(p, p) == pytest.approx(0.0)


class _ConstantModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1 - self.p), np.full(len(x), self.p)])


def test_calibration_detector_needs_labels_to_see_concept_drift():
    rng = np.random.default_rng(2)
    frame = _frame(4000, rng)
    detector = CalibrationDetector(_ConstantModel(0.2), NUMERIC + CATEGORICAL, "y")

    calibrated = frame.assign(y=rng.binomial(1, 0.2, len(frame)))
    drifted = frame.assign(y=rng.binomial(1, 0.3, len(frame)))

    assert not detector.test(calibrated).alarm
    assert detector.test(drifted).alarm


def test_domain_classifier_false_alarm_rate_is_near_alpha_without_drift():
    rng = np.random.default_rng(3)
    reference = _frame(1500, rng)
    alarms = [
        DomainClassifierDetector(NUMERIC, CATEGORICAL, seed=i).fit(reference).test(_frame(1500, rng)).alarm
        for i in range(40)
    ]

    assert np.mean(alarms) <= 0.15
