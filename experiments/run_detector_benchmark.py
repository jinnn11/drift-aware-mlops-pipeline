"""Evaluate drift detectors on injected drift with a known onset.

For every seed: train a champion on an i.i.d. sample of real loans, then stream batches
from five scenarios (no drift; sudden/gradual input-only drift; sudden/gradual concept-only
drift) and run each detector on each batch.

Harm is measured as retraining regret: the PR-AUC gap between the champion and an "oracle"
model trained on data from the drifted distribution, evaluated on a large sample from that
distribution. PR-AUC moves with the default rate, so a raw "PR-AUC went down" test would
mistake a safer population for a worse model; regret compares both models on the same rows.

    python -m experiments.run_detector_benchmark --data-path data/raw/lending_club.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.detectors import CalibrationDetector, DomainClassifierDetector, KSChi2Detector, PSIDetector
from src.drift_benchmark import (
    SCENARIOS,
    InjectionConfig,
    ScenarioSampler,
    detection_metrics,
    harm_lookahead,
    strength_at,
)
from src.lending_club import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, load_lending_club, monthly_stream
from src.replay import fit_model, pr_auc


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_all = monthly_stream(load_lending_club(args.data_path), args.pool_start, args.pool_end)
    injection = InjectionConfig()
    last_batch = args.batches - 1

    rows, regret_rows = [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        shuffled = pool_all.iloc[rng.permutation(len(pool_all))].reset_index(drop=True)
        train = shuffled.iloc[: args.train_rows]
        sampler = ScenarioSampler(shuffled.iloc[args.train_rows:], injection)
        champion = fit_model(train)
        reference = train[FEATURE_COLUMNS].sample(args.reference_rows, random_state=seed)

        regret_cache: dict = {}

        def regret(kind: str, s: float) -> float:
            key = (kind if s > 0 else "none", round(s, 6))
            if key not in regret_cache:
                oracle = fit_model(sampler.sample(kind, s, args.train_rows, rng))
                evaluation = sampler.sample(kind, s, args.eval_rows, rng)
                champ, orc = pr_auc(champion, evaluation), pr_auc(oracle, evaluation)
                regret_cache[key] = orc - champ
                regret_rows.append({"seed": seed, "kind": key[0], "strength": s, "champion_pr_auc": champ,
                                    "oracle_pr_auc": orc, "regret": orc - champ,
                                    "prevalence": float(evaluation["default"].mean())})
            return regret_cache[key]

        for scenario in SCENARIOS:
            strengths = [strength_at(scenario, b, args.onset, args.ramp) for b in range(args.batches)]
            harmful = {b: regret(scenario.kind, s) > args.harm_margin for b, s in enumerate(strengths)}
            batches = [sampler.sample(scenario.kind, s, args.batch_size, rng) for s in strengths]
            detectors = {
                "ks_chi2": (KSChi2Detector(NUMERIC_FEATURES, CATEGORICAL_FEATURES), 0),
                "psi": (PSIDetector(NUMERIC_FEATURES, CATEGORICAL_FEATURES, threshold=args.psi_threshold), 0),
                "domain_classifier": (DomainClassifierDetector(NUMERIC_FEATURES, CATEGORICAL_FEATURES, seed=seed), 0),
                "calibration (labels)": (CalibrationDetector(champion, FEATURE_COLUMNS, "default"), args.label_delay),
            }
            for name, (detector, delay) in detectors.items():
                detector.fit(reference)
                for b, batch in enumerate(batches):
                    raised_at = b + delay
                    if raised_at > last_batch:
                        continue  # labels for this batch arrive after the stream ends
                    result = detector.test(batch)
                    rows.append({
                        "seed": seed, "scenario": scenario.name, "detector": name, "batch": b,
                        "strength": strengths[b], "alarm": result.alarm, "score": result.score,
                        "alarm_batch": raised_at if result.alarm else np.nan,
                        "harm_ahead": harm_lookahead(harmful, raised_at, args.harm_horizon, last_batch)
                        if result.alarm else np.nan,
                        "batch_pr_auc": pr_auc(champion, batch),
                    })
        print(f"seed {seed} done", flush=True)

    results = pd.DataFrame(rows)
    regrets = pd.DataFrame(regret_rows)
    metrics = detection_metrics(results, args.onset)
    results.to_csv(out_dir / "batches.csv", index=False)
    regrets.to_csv(out_dir / "regret.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(metrics.round(3).to_string(index=False))
    print(regrets.groupby(["kind", "strength"])[["regret", "prevalence"]].mean().round(4).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default="data/raw/lending_club.csv")
    parser.add_argument("--pool-start", default="2013-01")
    parser.add_argument("--pool-end", default="2015-12")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--onset", type=int, default=8)
    parser.add_argument("--ramp", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--train-rows", type=int, default=30000)
    parser.add_argument("--reference-rows", type=int, default=5000)
    parser.add_argument("--eval-rows", type=int, default=30000)
    parser.add_argument("--label-delay", type=int, default=3)
    parser.add_argument("--psi-threshold", type=float, default=0.1)
    parser.add_argument("--harm-margin", type=float, default=0.02)
    parser.add_argument("--harm-horizon", type=int, default=2)
    parser.add_argument("--out-dir", default="results/detector_benchmark")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
