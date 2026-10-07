"""Rules for when a retrained model may go live.

Promotion gate: the candidate must beat the champion on the most recent labeled data and
must not get worse on older held-out data.

Noise guard: a retrain trigger is ignored during a cooldown after the last retrain, or when
too few new labeled rows have arrived since the last training cut-off.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GateConfig:
    min_recent_improvement: float = 0.0
    max_older_degradation: float = 0.01


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reason: str
    recent_delta: float
    older_delta: float


def evaluate_gate(
    champion_recent: float,
    candidate_recent: float,
    champion_older: float,
    candidate_older: float,
    config: GateConfig = GateConfig(),
) -> GateDecision:
    """Compare candidate and champion on the same rows; metrics are higher-is-better (e.g. PR-AUC)."""
    recent_delta = candidate_recent - champion_recent
    older_delta = candidate_older - champion_older
    beats_recent = recent_delta > config.min_recent_improvement
    keeps_older = older_delta >= -config.max_older_degradation

    if beats_recent and keeps_older:
        reason = "promoted"
    elif not beats_recent:
        reason = "no_gain_on_recent"
    else:
        reason = "regressed_on_older"
    return GateDecision(beats_recent and keeps_older, reason, recent_delta, older_delta)


@dataclass(frozen=True)
class NoiseGuardConfig:
    cooldown_periods: int = 2
    min_new_labeled_rows: int = 5000


def check_noise_guard(
    periods_since_last_retrain: Optional[int],
    new_labeled_rows: int,
    config: NoiseGuardConfig = NoiseGuardConfig(),
    apply_cooldown: bool = True,
) -> tuple[bool, str]:
    """Return (allowed, reason). `periods_since_last_retrain=None` means never retrained."""
    if apply_cooldown and periods_since_last_retrain is not None and periods_since_last_retrain < config.cooldown_periods:
        return False, "cooldown"
    if new_labeled_rows < config.min_new_labeled_rows:
        return False, "too_few_new_labels"
    return True, "ok"
