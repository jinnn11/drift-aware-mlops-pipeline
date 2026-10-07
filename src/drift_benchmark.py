"""Controlled drift scenarios for evaluating detectors where the drift onset is known.

The base stream is i.i.d.: batches are drawn from a shuffled pool of real loans, so any
drift comes from the injection alone.

  covariate: rows are resampled with weights exp(beta * s * z(x)), where z is a riskiness
             index built from FICO and DTI. Selection depends only on x, so P(y|x) is
             unchanged: input-only drift.
  concept:   rows are drawn uniformly (P(x) unchanged) and labels are flipped depending on x:
             high-FICO non-defaulters become defaulters and low-FICO defaulters become
             non-defaulters. The FICO-risk relationship partly inverts: concept-only drift.

`s` is the drift strength in [0, 1]: a step at `onset` (sudden) or a linear ramp over
`ramp` batches (gradual).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.lending_club import TARGET_COLUMN


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str  # "none" | "covariate" | "concept"
    shape: str  # "none" | "sudden" | "gradual"


SCENARIOS: List[Scenario] = [
    Scenario("no_drift", "none", "none"),
    Scenario("covariate_sudden", "covariate", "sudden"),
    Scenario("covariate_gradual", "covariate", "gradual"),
    Scenario("concept_sudden", "concept", "sudden"),
    Scenario("concept_gradual", "concept", "gradual"),
]


def strength_at(scenario: Scenario, batch: int, onset: int, ramp: int) -> float:
    if scenario.kind == "none" or batch < onset:
        return 0.0
    if scenario.shape == "sudden":
        return 1.0
    return float(min(1.0, (batch - onset + 1) / ramp))


@dataclass(frozen=True)
class InjectionConfig:
    covariate_beta: float = 0.35
    concept_high_fico_flip: float = 0.08
    concept_low_fico_flip: float = 0.40
    fico_quantiles: tuple = (0.3, 0.7)


def _normal_scores(values: pd.Series) -> pd.Series:
    from scipy.stats import norm

    return pd.Series(norm.ppf(values.rank(pct=True) - 0.5 / len(values)), index=values.index)


class ScenarioSampler:
    def __init__(self, pool: pd.DataFrame, config: InjectionConfig = InjectionConfig()):
        self.pool = pool.reset_index(drop=True)
        self.config = config
        # Rank-based normal scores: a single extreme value cannot capture the tilted weights.
        risk = _normal_scores(self.pool["dti"]) - _normal_scores(self.pool["fico"])
        self.risk_z = ((risk - risk.mean()) / risk.std()).to_numpy()
        lo, hi = self.pool["fico"].quantile(list(config.fico_quantiles))
        self.low_fico = (self.pool["fico"] <= lo).to_numpy()
        self.high_fico = (self.pool["fico"] >= hi).to_numpy()

    def sample(self, kind: str, s: float, n: int, rng: np.random.Generator) -> pd.DataFrame:
        if kind == "covariate" and s > 0:
            weights = np.exp(self.config.covariate_beta * s * self.risk_z)
            idx = rng.choice(len(self.pool), n, replace=True, p=weights / weights.sum())
        else:
            idx = rng.choice(len(self.pool), n, replace=False)
        batch = self.pool.iloc[idx].copy()
        if kind == "concept" and s > 0:
            y = batch[TARGET_COLUMN].to_numpy().copy()
            u = rng.random(n)
            to_default = self.high_fico[idx] & (y == 0) & (u < s * self.config.concept_high_fico_flip)
            to_paid = self.low_fico[idx] & (y == 1) & (u < s * self.config.concept_low_fico_flip)
            y[to_default], y[to_paid] = 1, 0
            batch[TARGET_COLUMN] = y
        return batch.reset_index(drop=True)


def detection_metrics(alarms: pd.DataFrame, onset: int) -> pd.DataFrame:
    """Summarize per (scenario, detector) across seeds.

    `alarms` has one row per (scenario, detector, seed, batch) with columns
    `alarm_batch` (when the alarm was raised, NaN if none), `strength` (of the data batch)
    and `harm_ahead` (harm at the raise time or within the next few batches).
    """
    rows = []
    for (scenario, detector), group in alarms.groupby(["scenario", "detector"], sort=False):
        raised = group.dropna(subset=["alarm_batch"])
        false_alarms = raised[raised["strength"] == 0]
        clean_batches = int((group["strength"] == 0).sum())
        delays: List[float] = []
        detected = 0
        seeds = group["seed"].nunique()
        fa_rate = len(false_alarms) / max(clean_batches, 1)
        drifted_batches_per_seed = (group["strength"] > 0).sum() / seeds
        if scenario != "no_drift":
            for _, seed_rows in raised[raised["strength"] > 0].groupby("seed"):
                delays.append(float(seed_rows["alarm_batch"].min() - onset))
                detected += 1
        rows.append({
            "scenario": scenario,
            "detector": detector,
            "false_alarm_rate": fa_rate,
            "detection_rate": detected / seeds if scenario != "no_drift" else np.nan,
            # Share of seeds a detector alarming at random with its false-alarm rate would "detect".
            "chance_detection_rate": 1 - (1 - fa_rate) ** drifted_batches_per_seed if scenario != "no_drift" else np.nan,
            "miss_rate": 1 - detected / seeds if scenario != "no_drift" else np.nan,
            "median_delay": float(np.median(delays)) if delays else np.nan,
            "alarms": int(len(raised)),
            "alarm_precision": float(raised["harm_ahead"].mean()) if len(raised) else np.nan,
        })
    return pd.DataFrame(rows)


def harm_lookahead(harmful_by_batch: Dict[int, bool], raised_at: int, horizon: int, last_batch: int) -> bool:
    """True if harm is present at `raised_at` or within `horizon` batches after it."""
    return any(harmful_by_batch[min(b, last_batch)] for b in range(raised_at, raised_at + horizon + 1))
