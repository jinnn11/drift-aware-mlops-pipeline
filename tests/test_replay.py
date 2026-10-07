import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from src.lending_club import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.promotion import NoiseGuardConfig
from src.replay import MonthlyStream, PolicySpec, ReplayConfig, hindsight_reference, run_policy


def _synthetic_stream(months=14, rows=600, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for i, month in enumerate(pd.period_range("2020-01", periods=months, freq="M")):
        fico = rng.normal(700 - 2 * i, 30, rows)
        dti = rng.normal(18, 8, rows)
        logit = -1.5 - 0.03 * (fico - 700) + 0.04 * (dti - 18)
        parts.append(pd.DataFrame({
            "annual_income": rng.lognormal(11, 0.4, rows), "dti": dti,
            "loan_amnt": rng.uniform(1000, 35000, rows), "fico": fico,
            "emp_length": rng.choice(["1 year", "10+ years"], rows),
            "purpose": rng.choice(["credit_card", "debt_consolidation"], rows),
            "home_ownership": rng.choice(["RENT", "MORTGAGE"], rows),
            "addr_state": rng.choice(["CA", "NY", "TX"], rows),
            "default": rng.binomial(1, 1 / (1 + np.exp(-logit))),
            "issue_month": month,
        }))
    return MonthlyStream(pd.concat(parts, ignore_index=True), holdout_frac=0.15, seed=seed)


def test_training_set_respects_label_delay_and_reserves_newest_month():
    stream = _synthetic_stream()
    labeled_through = pd.Period("2020-06", "M")

    data = stream.training_set(labeled_through, window=4)

    assert data.recent_eval["issue_month"].unique().tolist() == [labeled_through]
    assert data.train["issue_month"].max() == labeled_through - 1
    assert data.train["issue_month"].min() == labeled_through - 3
    assert not data.train["_holdout"].any()
    assert data.older_eval["_holdout"].all()


def test_scheduled_policy_never_trains_on_unlabeled_months():
    stream = _synthetic_stream()
    replay = stream.months[8:]
    cfg = ReplayConfig(label_delay_months=2, train_window_months=4, guard=NoiseGuardConfig(min_new_labeled_rows=100),
                       n_jobs=2)

    result = run_policy(stream, replay, PolicySpec("monthly", "scheduled", every_months=1), cfg)

    events = result.events
    assert len(events) == len(replay) - 1  # no retrain after the last month
    for _, event in events.iterrows():
        assert pd.Period(event["labeled_through"], "M") == pd.Period(event["month"], "M") - 2
    assert set(result.monthly["month"]) == {str(m) for m in replay}


def test_drift_policy_respects_cooldown():
    from src.detectors import KSChi2Detector

    stream = _synthetic_stream()
    replay = stream.months[8:]
    cfg = ReplayConfig(label_delay_months=1, train_window_months=4,
                       guard=NoiseGuardConfig(cooldown_periods=3, min_new_labeled_rows=100), n_jobs=2)
    always_alarm = PolicySpec("drift", "drift",
                              detector_factory=lambda champion: KSChi2Detector(NUMERIC_FEATURES, CATEGORICAL_FEATURES, alpha=1.0))

    result = run_policy(stream, replay, always_alarm, cfg, hindsight=hindsight_reference(stream, replay, cfg))

    retrain_idx = np.flatnonzero(result.monthly["retrained"].to_numpy())
    assert np.all(np.diff(retrain_idx) >= 3)
    assert result.monthly["blocked"].eq("cooldown").sum() > 0
    assert result.summary["alarms_scored"] == len(replay) - 1


def test_shadow_monitor_scores_input_and_label_based_detectors():
    from src.detectors import CalibrationDetector, KSChi2Detector
    from src.lending_club import FEATURE_COLUMNS
    from src.replay import shadow_monitor

    stream = _synthetic_stream()
    replay = stream.months[8:]
    cfg = ReplayConfig(label_delay_months=2, train_window_months=4, n_jobs=2)
    factories = {
        "ks": lambda champion: KSChi2Detector(NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        "calibration": lambda champion: CalibrationDetector(champion, FEATURE_COLUMNS, "default"),
    }

    shadow = shadow_monitor(stream, replay, factories, cfg, hindsight_reference(stream, replay, cfg))

    assert set(shadow["detector"]) == {"ks", "calibration"}
    assert len(shadow) == 2 * (len(replay) - 1)  # last month has no future to score harm on
    assert shadow["harm_ahead"].dtype == bool
