from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import httpx
import pandas as pd

from monitor import run_monitoring
from src.data import load_dataset, validate_dataset


@dataclass
class PhaseResult:
    phase: str
    requests_sent: int
    requests_ok: int
    requests_failed: int
    monitor_summary: dict | None



def build_phase_datasets(df: pd.DataFrame, holdout_ratio: float) -> Dict[str, pd.DataFrame]:
    holdout_start = int(len(df) * (1.0 - holdout_ratio))
    holdout = df.iloc[holdout_start:].copy().reset_index(drop=True)

    normal_df = holdout.copy()

    recession_df = holdout.copy()
    recession_df["emp.var.rate"] = recession_df["emp.var.rate"] - 3.0
    recession_df["cons.price.idx"] = recession_df["cons.price.idx"] * 1.05
    recession_df["euribor3m"] = recession_df["euribor3m"] * 0.50

    concept_df = recession_df.copy()
    retired_mask = concept_df["job"] == "retired"
    concept_df.loc[retired_mask, "y"] = "no"

    return {
        "phase1_normal": normal_df,
        "phase2_recession": recession_df,
        "phase3_concept": concept_df,
    }



def save_phase_datasets(phase_dfs: Dict[str, pd.DataFrame], output_dir: str) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_paths: Dict[str, str] = {}
    for phase_name, phase_df in phase_dfs.items():
        path = out_dir / f"{phase_name}.csv"
        phase_df.to_csv(path, sep=";", index=False)
        phase_paths[phase_name] = str(path)
    return phase_paths



def _to_feature_payload(row: pd.Series) -> dict:
    payload = row.to_dict()
    payload.pop("y", None)
    payload.pop("duration", None)
    return payload



def send_phase_traffic(
    client: httpx.Client,
    api_url: str,
    phase_df: pd.DataFrame,
    duration_seconds: int,
    requests_per_second: int,
) -> Tuple[int, int, int]:
    total_requests = max(1, duration_seconds * requests_per_second)
    ok_count = 0
    fail_count = 0

    start = time.time()
    for i in range(total_requests):
        row = phase_df.iloc[i % len(phase_df)]
        body = {"instances": [_to_feature_payload(row)]}

        try:
            response = client.post(api_url, json=body, timeout=10.0)
            if response.status_code == 200:
                ok_count += 1
            else:
                fail_count += 1
        except Exception:
            fail_count += 1

        # Pace requests to approximate the requested rate.
        target_elapsed = (i + 1) / max(requests_per_second, 1)
        now_elapsed = time.time() - start
        sleep_for = target_elapsed - now_elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    return total_requests, ok_count, fail_count



def run_simulation(
    dataset_path: str,
    api_url: str,
    holdout_ratio: float,
    phase_seconds: int,
    requests_per_second: int,
    output_dir: str,
    run_monitor_after_phase: bool,
    monitor_min_rows: int = 50,
    client: httpx.Client | None = None,
) -> dict:
    raw_df = load_dataset(dataset_path)
    validate_dataset(raw_df)

    phase_dfs = build_phase_datasets(raw_df, holdout_ratio=holdout_ratio)
    phase_paths = save_phase_datasets(phase_dfs, output_dir=output_dir)

    ordered_phases = ["phase1_normal", "phase2_recession", "phase3_concept"]
    results: list[PhaseResult] = []

    own_client = client is None
    client = client or httpx.Client()
    try:
        for phase_name in ordered_phases:
            phase_df = phase_dfs[phase_name]
            phase_started = datetime.now(timezone.utc)
            total, ok, failed = send_phase_traffic(
                client=client,
                api_url=api_url,
                phase_df=phase_df,
                duration_seconds=phase_seconds,
                requests_per_second=requests_per_second,
            )

            monitor_summary = None
            if run_monitor_after_phase:
                # Monitor exactly the predictions the API logged during this phase.
                monitor_summary = run_monitoring(
                    source="logs",
                    since=phase_started,
                    min_rows=monitor_min_rows,
                )

            results.append(
                PhaseResult(
                    phase=phase_name,
                    requests_sent=total,
                    requests_ok=ok,
                    requests_failed=failed,
                    monitor_summary=monitor_summary,
                )
            )
    finally:
        if own_client:
            client.close()

    return {
        "phase_datasets": phase_paths,
        "phase_duration_seconds": phase_seconds,
        "requests_per_second": requests_per_second,
        "results": [
            {
                "phase": r.phase,
                "requests_sent": r.requests_sent,
                "requests_ok": r.requests_ok,
                "requests_failed": r.requests_failed,
                "monitor_summary": r.monitor_summary,
            }
            for r in results
        ],
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate normal/recession/concept drift traffic against /predict.")
    parser.add_argument("--dataset-path", default="data/raw/bank-additional-full.csv")
    parser.add_argument("--api-url", default="http://localhost:8000/predict")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--phase-seconds", type=int, default=60)
    parser.add_argument("--requests-per-second", type=int, default=1)
    parser.add_argument("--output-dir", default="data/processed/simulation")
    parser.add_argument("--run-monitor-after-phase", action="store_true")
    parser.add_argument("--monitor-min-rows", type=int, default=50,
                        help="logged predictions a phase needs before its drift check runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_simulation(
        dataset_path=args.dataset_path,
        api_url=args.api_url,
        holdout_ratio=args.holdout_ratio,
        phase_seconds=args.phase_seconds,
        requests_per_second=args.requests_per_second,
        output_dir=args.output_dir,
        run_monitor_after_phase=args.run_monitor_after_phase,
        monitor_min_rows=args.monitor_min_rows,
    )
    print(json.dumps(summary, indent=2))
