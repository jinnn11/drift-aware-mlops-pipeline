from src.promotion import GateConfig, NoiseGuardConfig, check_noise_guard, evaluate_gate


def test_gate_promotes_when_better_on_recent_and_stable_on_older():
    decision = evaluate_gate(champion_recent=0.30, candidate_recent=0.32, champion_older=0.35, candidate_older=0.345)

    assert decision.promote
    assert decision.reason == "promoted"


def test_gate_rejects_candidate_that_does_not_beat_champion_on_recent():
    decision = evaluate_gate(0.30, 0.30, 0.35, 0.40)

    assert not decision.promote
    assert decision.reason == "no_gain_on_recent"


def test_gate_rejects_candidate_that_regresses_on_older_data():
    decision = evaluate_gate(0.30, 0.35, 0.35, 0.33, GateConfig(max_older_degradation=0.01))

    assert not decision.promote
    assert decision.reason == "regressed_on_older"


def test_noise_guard_blocks_during_cooldown():
    allowed, reason = check_noise_guard(1, 10_000, NoiseGuardConfig(cooldown_periods=2))

    assert not allowed and reason == "cooldown"


def test_noise_guard_blocks_with_too_few_new_labels():
    allowed, reason = check_noise_guard(None, 100, NoiseGuardConfig(min_new_labeled_rows=500))

    assert not allowed and reason == "too_few_new_labels"


def test_noise_guard_cooldown_can_be_skipped_for_scheduled_retrains():
    allowed, _ = check_noise_guard(0, 10_000, NoiseGuardConfig(cooldown_periods=2), apply_cooldown=False)

    assert allowed
