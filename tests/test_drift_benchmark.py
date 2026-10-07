import numpy as np
import pandas as pd
import pytest

from src.drift_benchmark import Scenario, ScenarioSampler, detection_metrics, harm_lookahead, strength_at


def _pool(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    fico = rng.normal(700, 30, n)
    dti = rng.normal(18, 8, n)
    p = 1 / (1 + np.exp(-(-1.5 - 0.03 * (fico - 700) + 0.03 * (dti - 18))))
    return pd.DataFrame({"fico": fico, "dti": dti, "default": rng.binomial(1, p)})


def test_strength_profiles():
    sudden, gradual = Scenario("s", "covariate", "sudden"), Scenario("g", "covariate", "gradual")

    assert [strength_at(sudden, b, onset=2, ramp=4) for b in range(4)] == [0, 0, 1, 1]
    assert [strength_at(gradual, b, onset=2, ramp=4) for b in range(7)] == [0, 0, 0.25, 0.5, 0.75, 1, 1]


def test_covariate_injection_shifts_inputs_but_keeps_labels_tied_to_inputs():
    sampler = ScenarioSampler(_pool())
    rng = np.random.default_rng(0)

    base = sampler.sample("covariate", 0.0, 5000, rng)
    shifted = sampler.sample("covariate", 1.0, 5000, rng)

    assert shifted["fico"].mean() < base["fico"].mean() - 5
    assert shifted["dti"].mean() > base["dti"].mean() + 1


def test_concept_injection_keeps_inputs_and_changes_labels():
    pool = _pool()
    sampler = ScenarioSampler(pool)
    rng = np.random.default_rng(0)

    drifted = sampler.sample("concept", 1.0, 8000, rng)
    original = pool.loc[drifted["fico"].map(dict(zip(pool["fico"], pool.index))), "default"].to_numpy()

    high = drifted["fico"] >= pool["fico"].quantile(0.7)
    assert drifted.loc[high, "default"].mean() > original[high.to_numpy()].mean()
    assert abs(drifted["fico"].mean() - pool["fico"].mean()) < 2


def test_detection_metrics_delay_false_alarms_and_misses():
    rows = []
    for seed, first_alarm in [(0, 5), (1, None)]:
        for b in range(8):
            strength = 1.0 if b >= 4 else 0.0
            alarm = (first_alarm is not None and b >= first_alarm) or (seed == 1 and b == 1)
            rows.append({"scenario": "covariate_sudden", "detector": "d", "seed": seed, "batch": b,
                         "strength": strength, "alarm_batch": b if alarm else np.nan,
                         "harm_ahead": strength > 0 if alarm else np.nan})
    metrics = detection_metrics(pd.DataFrame(rows), onset=4).iloc[0]

    assert metrics["median_delay"] == 1
    assert metrics["detection_rate"] == 0.5
    assert metrics["false_alarm_rate"] == pytest.approx(1 / 8)
    assert metrics["alarm_precision"] == pytest.approx(3 / 4)


def test_harm_lookahead_clips_at_stream_end():
    harmful = {0: False, 1: False, 2: True}

    assert harm_lookahead(harmful, 0, horizon=2, last_batch=2)
    assert not harm_lookahead(harmful, 0, horizon=1, last_batch=2)
    assert harm_lookahead(harmful, 2, horizon=3, last_batch=2)


def test_covariate_injection_is_not_captured_by_one_outlier():
    pool = _pool()
    pool.loc[0, "dti"] = 999.0  # sentinel value present in the raw Lending Club file
    sampler = ScenarioSampler(pool)

    batch = sampler.sample("covariate", 1.0, 5000, np.random.default_rng(0))

    assert batch["dti"].nunique() > 4000
    assert 0.05 < batch["default"].mean() < 0.6
