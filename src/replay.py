"""Time-ordered replay of a monthly stream with delayed labels.

Each month t:
  1. The current champion scores month t. Its PR-AUC on month t is recorded with hindsight
     labels for reporting only; the policy never sees those labels early.
  2. The policy decides whether to retrain at the end of month t. Labels for month s become
     available at the end of month s + label_delay, so at the end of month t the newest
     labeled month is t - label_delay.
  3. A retrain fits a candidate on the trailing window of labeled months, then the promotion
     gate decides whether it replaces the champion from month t + 1.

A fixed random share of every month (`holdout_frac`) is never trained on by any model, so
the gate's "older held-out data" is unseen by both champion and candidate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.lending_club import FEATURE_COLUMNS, MONTH_COLUMN, TARGET_COLUMN
from src.modeling import build_fixed_pipeline
from src.promotion import GateConfig, NoiseGuardConfig, check_noise_guard, evaluate_gate


@dataclass(frozen=True)
class ReplayConfig:
    label_delay_months: int = 3
    train_window_months: int = 12
    holdout_frac: float = 0.15
    detector_rows: int = 5000
    harm_horizon_months: int = 3
    harm_margin: float = 0.02
    gate: GateConfig = GateConfig()
    guard: NoiseGuardConfig = NoiseGuardConfig()
    n_jobs: int = -1
    seed: int = 0


@dataclass
class PolicySpec:
    name: str
    kind: str  # "never" | "scheduled" | "drift"
    every_months: int = 1
    # Called with the current champion; input detectors ignore it, the label-based one uses it.
    detector_factory: Optional[Callable[[object], object]] = None


@dataclass
class TrainingSet:
    train: pd.DataFrame
    older_eval: pd.DataFrame
    recent_eval: pd.DataFrame
    labeled_through: pd.Period


@dataclass
class PolicyResult:
    name: str
    monthly: pd.DataFrame
    events: pd.DataFrame
    alarms: pd.DataFrame
    summary: Dict[str, float] = field(default_factory=dict)


def pr_auc(model, frame: pd.DataFrame) -> float:
    return float(average_precision_score(frame[TARGET_COLUMN], model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]))


def fit_model(train: pd.DataFrame, n_jobs: int = -1):
    model = build_fixed_pipeline(train[FEATURE_COLUMNS], n_jobs=n_jobs)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model


class MonthlyStream:
    def __init__(self, frame: pd.DataFrame, holdout_frac: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        frame = frame.copy()
        frame["_holdout"] = rng.random(len(frame)) < holdout_frac
        self.by_month: Dict[pd.Period, pd.DataFrame] = {m: g for m, g in frame.groupby(MONTH_COLUMN, sort=True)}
        self.months: List[pd.Period] = sorted(self.by_month)

    def rows_between(self, after: pd.Period, through: pd.Period) -> int:
        return int(sum(len(self.by_month[m]) for m in self.months if after < m <= through))

    def training_set(self, labeled_through: pd.Period, window: int) -> TrainingSet:
        """Trailing `window` labeled months; the newest is reserved as the gate's recent eval."""
        window_months = [m for m in self.months if labeled_through - window < m <= labeled_through]
        if len(window_months) < 2:
            raise ValueError(f"Not enough labeled months through {labeled_through}")
        recent, older = window_months[-1], window_months[:-1]
        older_frames = [self.by_month[m] for m in older]
        return TrainingSet(
            train=pd.concat([f[~f["_holdout"]] for f in older_frames]),
            older_eval=pd.concat([f[f["_holdout"]] for f in older_frames]),
            recent_eval=self.by_month[recent],
            labeled_through=recent,
        )


def hindsight_reference(stream: MonthlyStream, replay_months: List[pd.Period], cfg: ReplayConfig) -> pd.Series:
    """PR-AUC of a model retrained every month with zero label delay (an upper reference, not a policy)."""
    scores = {}
    for t in replay_months:
        prior = [m for m in stream.months if t - cfg.train_window_months <= m < t]
        train = pd.concat([stream.by_month[m] for m in prior])
        scores[t] = pr_auc(fit_model(train[~train["_holdout"]], cfg.n_jobs), stream.by_month[t])
    return pd.Series(scores, name="hindsight_pr_auc")


def _is_label_based(detector) -> bool:
    return getattr(detector, "name", "") == "calibration"


def _sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    return frame if len(frame) <= n else frame.iloc[rng.choice(len(frame), n, replace=False)]


def run_policy(
    stream: MonthlyStream,
    replay_months: List[pd.Period],
    policy: PolicySpec,
    cfg: ReplayConfig,
    hindsight: Optional[pd.Series] = None,
    registry=None,
) -> PolicyResult:
    rng = np.random.default_rng(cfg.seed)
    delay = cfg.label_delay_months

    initial = stream.training_set(replay_months[0] - 1 - delay, cfg.train_window_months)
    champion = fit_model(initial.train, cfg.n_jobs)
    champion_id = "v0"
    if registry is not None:
        champion_id = registry.register(champion, f"{policy.name}-initial", {"policy": policy.name},
                                        {"older_pr_auc": pr_auc(champion, initial.older_eval)})
        registry.promote(champion_id)
    last_cutoff = initial.labeled_through
    last_retrain_index: Optional[int] = None

    detector = None
    if policy.kind == "drift":
        detector = policy.detector_factory(champion).fit(_sample(initial.train[FEATURE_COLUMNS], cfg.detector_rows, rng))

    monthly, events, alarms = [], [], []
    for i, t in enumerate(replay_months):
        month = stream.by_month[t]
        scores = champion.predict_proba(month[FEATURE_COLUMNS])[:, 1]
        row = {
            "policy": policy.name, "month": str(t), "n": len(month),
            "prevalence": float(month[TARGET_COLUMN].mean()),
            "pr_auc": float(average_precision_score(month[TARGET_COLUMN], scores)),
            "roc_auc": float(roc_auc_score(month[TARGET_COLUMN], scores)),
            "champion": champion_id, "alarm": False, "detector_score": np.nan,
            "triggered": False, "blocked": "", "retrained": False, "promoted": False,
        }

        if policy.kind == "scheduled":
            row["triggered"] = (i + 1) % policy.every_months == 0
        elif policy.kind == "drift":
            if _is_label_based(detector):
                result = detector.test(stream.by_month[t - delay])  # newest labeled month
            else:
                result = detector.test(_sample(month[FEATURE_COLUMNS], cfg.detector_rows, rng))
            row.update(alarm=result.alarm, detector_score=result.score, triggered=result.alarm)
            if result.alarm:
                alarms.append({"month": t, "index": i, "champion_model": champion, "champion": champion_id})

        is_last = i == len(replay_months) - 1
        if row["triggered"] and not is_last:
            labeled_through = t - delay
            allowed, reason = check_noise_guard(
                None if last_retrain_index is None else i - last_retrain_index,
                stream.rows_between(last_cutoff, labeled_through),
                cfg.guard,
                apply_cooldown=policy.kind == "drift",
            )
            if not allowed:
                row["blocked"] = reason
            else:
                data = stream.training_set(labeled_through, cfg.train_window_months)
                started = time.perf_counter()
                candidate = fit_model(data.train, cfg.n_jobs)
                train_seconds = time.perf_counter() - started
                metrics = {
                    "champion_recent_pr_auc": pr_auc(champion, data.recent_eval),
                    "candidate_recent_pr_auc": pr_auc(candidate, data.recent_eval),
                    "champion_older_pr_auc": pr_auc(champion, data.older_eval),
                    "candidate_older_pr_auc": pr_auc(candidate, data.older_eval),
                }
                decision = evaluate_gate(
                    metrics["champion_recent_pr_auc"], metrics["candidate_recent_pr_auc"],
                    metrics["champion_older_pr_auc"], metrics["candidate_older_pr_auc"], cfg.gate,
                )
                candidate_id = f"v{len(events) + 1}"
                if registry is not None:
                    candidate_id = registry.register(
                        candidate, f"{policy.name}-{t}",
                        {"policy": policy.name, "month": str(t), "labeled_through": str(labeled_through),
                         "train_rows": len(data.train)},
                        {**metrics, "train_seconds": train_seconds},
                        tags={"gate": decision.reason},
                    )
                if decision.promote:
                    champion, champion_id = candidate, candidate_id
                    if registry is not None:
                        registry.promote(candidate_id)
                    if detector is not None:
                        detector = policy.detector_factory(champion).fit(
                            _sample(data.train[FEATURE_COLUMNS], cfg.detector_rows, rng))
                last_cutoff, last_retrain_index = labeled_through, i
                row.update(retrained=True, promoted=decision.promote)
                events.append({
                    "policy": policy.name, "month": str(t), "labeled_through": str(labeled_through),
                    "candidate": candidate_id, "train_rows": len(data.train), "train_seconds": train_seconds,
                    "gate_reason": decision.reason, "recent_delta": decision.recent_delta,
                    "older_delta": decision.older_delta, **metrics,
                })
        monthly.append(row)

    monthly_df = pd.DataFrame(monthly)
    alarm_df = _score_alarms(stream, replay_months, alarms, hindsight, cfg)
    events_df = pd.DataFrame(events)
    return PolicyResult(policy.name, monthly_df, events_df, alarm_df,
                        _summarize(policy.name, monthly_df, events_df, alarm_df, hindsight))


def _score_alarms(stream, replay_months, alarms, hindsight, cfg) -> pd.DataFrame:
    """An alarm 'preceded real harm' if the champion it fired on, left in place, trails the
    hindsight model by more than `harm_margin` PR-AUC over the following months."""
    rows = []
    for alarm in alarms:
        future = replay_months[alarm["index"] + 1: alarm["index"] + 1 + cfg.harm_horizon_months]
        if hindsight is None or not future:
            continue
        regret = [hindsight[m] - pr_auc(alarm["champion_model"], stream.by_month[m]) for m in future]
        rows.append({"month": str(alarm["month"]), "champion": alarm["champion"],
                     "mean_regret": float(np.mean(regret)), "harmful": bool(np.mean(regret) > cfg.harm_margin)})
    return pd.DataFrame(rows, columns=["month", "champion", "mean_regret", "harmful"])


def shadow_monitor(
    stream: MonthlyStream,
    replay_months: List[pd.Period],
    detector_factories: Dict[str, Callable[[object], object]],
    cfg: ReplayConfig,
    hindsight: pd.Series,
) -> pd.DataFrame:
    """Run every detector each month against a champion that is never replaced.

    Retraining resets the model right after an alarm, so alarm quality is measured here,
    in monitor-only mode. Each factory receives the frozen champion (only the label-based
    detector uses it). Input detectors test month t against the champion's training inputs;
    the label-based detector tests month t - label_delay, the newest month with labels, so
    its alarm is raised at t.

    `harm_ahead` at month t: the frozen champion trails the hindsight model by more than
    `harm_margin` PR-AUC on average over months t+1 .. t+harm_horizon_months.
    """
    rng = np.random.default_rng(cfg.seed)
    initial = stream.training_set(replay_months[0] - 1 - cfg.label_delay_months, cfg.train_window_months)
    champion = fit_model(initial.train, cfg.n_jobs)
    reference = _sample(initial.train[FEATURE_COLUMNS], cfg.detector_rows, rng)

    regret = pd.Series({m: hindsight[m] - pr_auc(champion, stream.by_month[m]) for m in replay_months})
    rows = []
    for name, factory in detector_factories.items():
        detector = factory(champion).fit(reference)
        label_based = _is_label_based(detector)
        for i, t in enumerate(replay_months):
            future = replay_months[i + 1: i + 1 + cfg.harm_horizon_months]
            if not future:
                continue
            if label_based:
                tested = t - cfg.label_delay_months
                if tested not in stream.by_month:
                    continue
                result = detector.test(stream.by_month[tested])
            else:
                result = detector.test(_sample(stream.by_month[t][FEATURE_COLUMNS], cfg.detector_rows, rng))
            future_regret = float(regret[future].mean())
            rows.append({"detector": name, "month": str(t), "alarm": bool(result.alarm), "score": result.score,
                         "regret_now": float(regret[t]), "regret_ahead": future_regret,
                         "harm_ahead": future_regret > cfg.harm_margin})
    return pd.DataFrame(rows)


def _summarize(name, monthly, events, alarms, hindsight) -> Dict[str, float]:
    summary = {
        "policy": name,
        "mean_pr_auc": float(monthly["pr_auc"].mean()),
        "mean_roc_auc": float(monthly["roc_auc"].mean()),
        "retrains": int(len(events)),
        "promotions": int(events["gate_reason"].eq("promoted").sum()) if len(events) else 0,
        "train_seconds": float(events["train_seconds"].sum()) if len(events) else 0.0,
        "alarms": int(monthly["alarm"].sum()),
        "blocked_cooldown": int(monthly["blocked"].eq("cooldown").sum()),
        "blocked_min_labels": int(monthly["blocked"].eq("too_few_new_labels").sum()),
        "alarms_scored": int(len(alarms)),
        "alarms_preceding_harm": int(alarms["harmful"].sum()) if len(alarms) else 0,
    }
    if hindsight is not None:
        gap = hindsight.reindex(pd.PeriodIndex(monthly["month"], freq="M")).to_numpy() - monthly["pr_auc"].to_numpy()
        summary["mean_gap_to_hindsight"] = float(np.mean(gap))
    return summary
