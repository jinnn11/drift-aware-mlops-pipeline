"""Does retraining when drift is detected beat retraining on a schedule, and at what cost?

Replays Lending Club loans month by month (2013-2015 by default) with labels arriving
`--label-delay` months late, and runs each retraining policy on the same stream. All
policies share the model, the training window, and the promotion gate; they differ only in
when they ask for a retrain.

    python -m experiments.run_policy_comparison --label-delay 3
    python -m experiments.run_policy_comparison --label-delay 3 --mlflow-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.detectors import CalibrationDetector, DomainClassifierDetector, KSChi2Detector, PSIDetector
from src.lending_club import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_lending_club,
    monthly_stream,
)
from src.promotion import GateConfig, NoiseGuardConfig
from src.replay import MonthlyStream, PolicySpec, ReplayConfig, hindsight_reference, run_policy, shadow_monitor


def detector_factories(seed: int) -> dict:
    """name -> factory(champion) for every detector used by policies and the shadow monitor."""
    num, cat = NUMERIC_FEATURES, CATEGORICAL_FEATURES
    return {
        "ks_chi2": lambda champion: KSChi2Detector(num, cat),
        "psi": lambda champion: PSIDetector(num, cat, threshold=0.1),
        "domain_classifier": lambda champion: DomainClassifierDetector(num, cat, seed=seed),
        "domain_classifier_auc55": lambda champion: DomainClassifierDetector(num, cat, seed=seed, min_auc=0.55),
        "calibration": lambda champion: CalibrationDetector(champion, FEATURE_COLUMNS, TARGET_COLUMN),
    }


def build_policies(names: list[str], seed: int) -> list[PolicySpec]:
    detectors = detector_factories(seed)
    available = {
        "never": PolicySpec("never", "never"),
        "scheduled_monthly": PolicySpec("scheduled_monthly", "scheduled", every_months=1),
        "scheduled_quarterly": PolicySpec("scheduled_quarterly", "scheduled", every_months=3),
        **{f"drift_{name}": PolicySpec(f"drift_{name}", "drift", detector_factory=factory)
           for name, factory in detectors.items()},
    }
    return [available[name] for name in names]


def run(args: argparse.Namespace) -> None:
    cfg = ReplayConfig(
        label_delay_months=args.label_delay,
        train_window_months=args.train_window,
        gate=GateConfig(args.gate_min_improvement, args.gate_max_older_drop),
        guard=NoiseGuardConfig(args.cooldown, args.min_new_labels),
        seed=args.seed,
    )
    data_start = pd.Period(args.replay_start, "M") - (args.train_window + args.label_delay + 1)
    frame = monthly_stream(load_lending_club(args.data_path), str(data_start), args.replay_end,
                           max_rows_per_month=args.max_rows_per_month, seed=args.seed)
    stream = MonthlyStream(frame, cfg.holdout_frac, cfg.seed)
    replay_months = [m for m in stream.months if m >= pd.Period(args.replay_start, "M")]

    out_dir = Path(args.out_dir) / f"delay_{args.label_delay}" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    hindsight = hindsight_reference(stream, replay_months, cfg)
    hindsight.rename_axis("month").reset_index().assign(month=lambda d: d["month"].astype(str)).to_csv(
        out_dir / "hindsight.csv", index=False)

    if not args.skip_shadow:
        shadow = shadow_monitor(stream, replay_months, detector_factories(args.seed), cfg, hindsight)
        shadow.to_csv(out_dir / "shadow.csv", index=False)

    monthly, events, alarms, summaries = [], [], [], []
    for policy in build_policies(args.policies.split(","), args.seed):
        registry = None
        if args.mlflow_uri:
            from src.mlflow_registry import MlflowChampionRegistry

            registry = MlflowChampionRegistry(
                f"lending-club-delay{args.label_delay}-{policy.name}", args.mlflow_uri,
                f"policy-comparison-delay{args.label_delay}")
        result = run_policy(stream, replay_months, policy, cfg, hindsight, registry)
        monthly.append(result.monthly)
        events.append(result.events)
        if len(result.alarms):
            alarms.append(result.alarms.assign(policy=policy.name))
        summaries.append(result.summary)
        print(json.dumps(result.summary), flush=True)

    pd.concat(monthly).to_csv(out_dir / "monthly.csv", index=False)
    pd.concat([e for e in events if len(e)]).to_csv(out_dir / "events.csv", index=False)
    (pd.concat(alarms) if alarms else pd.DataFrame()).to_csv(out_dir / "alarms.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "replay_config": asdict(cfg)}, indent=2, default=str))
    print(summary.round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default="data/raw/lending_club.csv")
    parser.add_argument("--replay-start", default="2013-01")
    parser.add_argument("--replay-end", default="2015-12")
    parser.add_argument("--max-rows-per-month", type=int, default=8000)
    parser.add_argument("--label-delay", type=int, default=3, help="months until a loan's outcome is known")
    parser.add_argument("--train-window", type=int, default=12, help="trailing labeled months used for training")
    parser.add_argument("--cooldown", type=int, default=2, help="months after a retrain during which drift alarms are ignored")
    parser.add_argument("--min-new-labels", type=int, default=5000, help="new labeled rows required before retraining")
    parser.add_argument("--gate-min-improvement", type=float, default=0.0)
    parser.add_argument("--gate-max-older-drop", type=float, default=0.01)
    parser.add_argument("--policies", default=",".join([
        "never", "scheduled_monthly", "scheduled_quarterly", "drift_ks_chi2", "drift_psi",
        "drift_domain_classifier", "drift_domain_classifier_auc55", "drift_calibration"]))
    parser.add_argument("--skip-shadow", action="store_true", help="skip the monitor-only alarm-quality pass")
    parser.add_argument("--mlflow-uri", default=None, help="e.g. sqlite:///mlflow.db to register every candidate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/policy_comparison")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
