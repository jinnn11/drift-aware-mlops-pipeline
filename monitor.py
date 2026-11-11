from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import get_settings
from src.data import load_dataset, prepare_features_and_target
from src.db import fetch_logged_predictions, get_engine, init_db, log_monitor_run
from src.registry import load_promoted_model, load_reference_data



def _walk_dict_values(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dict_values(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_dict_values(item)



def extract_drift_summary(report_dict: Dict[str, Any], total_features: int) -> Dict[str, float]:
    drifted_features = None
    drift_score = None

    for item in _walk_dict_values(report_dict):
        if drifted_features is None and "number_of_drifted_columns" in item:
            drifted_features = int(item["number_of_drifted_columns"])
        if drift_score is None and "share_of_drifted_columns" in item:
            drift_score = float(item["share_of_drifted_columns"])

    if drifted_features is None and drift_score is not None:
        drifted_features = int(round(drift_score * total_features))
    if drift_score is None and drifted_features is not None:
        drift_score = float(drifted_features / max(total_features, 1))

    if drifted_features is None:
        drifted_features = 0
    if drift_score is None:
        drift_score = 0.0

    return {
        "drifted_features": drifted_features,
        "drift_score": drift_score,
    }



def _write_latest(settings, summary: dict) -> None:
    Path(settings.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(settings.report_dir) / "latest_monitor.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_monitoring(
    dataset_path: Optional[str] = None,
    current_window_ratio: float = 0.3,
    source: str = "logs",
    since: Optional[datetime] = None,
    min_rows: int = 200,
) -> dict:
    """Compare recent data with the model's training data using Evidently.

    source="logs": the predictions the API actually served, read back from PostgreSQL
    (features and the model's score; no labels, because outcomes arrive later).
    source="file": the tail of a labeled CSV, for offline checks (features, score and label).
    """
    settings = get_settings()

    reference_df = load_reference_data(settings.reference_data_path)
    model, metadata = load_promoted_model(settings.model_path, settings.model_metadata_path)
    feature_columns = metadata["feature_columns"]

    engine = get_engine(settings.database_url)
    init_db(engine)

    if source == "logs":
        logged = fetch_logged_predictions(engine, since=since)
        if len(logged) < min_rows:
            summary = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "rows": len(logged),
                "status": "insufficient_data",
                "detail": f"only {len(logged)} logged predictions since {since}; need {min_rows}",
            }
            _write_latest(settings, summary)
            return summary
        current_df = pd.DataFrame([row["features"] for row in logged])
        current_df["pred_score"] = [float(row["score"]) for row in logged]
        monitored_columns = feature_columns + ["pred_score"]
    elif source == "file":
        if not dataset_path:
            raise ValueError("source='file' needs dataset_path")
        raw_df = load_dataset(dataset_path)
        x, y = prepare_features_and_target(raw_df)
        current_window = max(1000, int(len(x) * current_window_ratio))
        current_df = x.tail(current_window).copy()
        current_df["y"] = y.tail(current_window).values
        current_df["pred_score"] = model.predict_proba(current_df[feature_columns])[:, 1]
        monitored_columns = feature_columns + ["pred_score", "y"]
    else:
        raise ValueError(f"Unknown source: {source}")

    missing = [col for col in monitored_columns if col not in current_df.columns]
    if missing:
        raise ValueError(f"Current data is missing monitored columns: {missing}")

    reference_view = reference_df[monitored_columns].copy()
    current_view = current_df[monitored_columns].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_view, current_data=current_view)

    report_ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    report_path = Path(settings.report_dir) / f"drift_report_{report_ts}.html"
    report.save_html(str(report_path))

    total_features = len(monitored_columns)
    summary = extract_drift_summary(report.as_dict(), total_features=total_features)
    drift_ratio = summary["drifted_features"] / max(total_features, 1)
    status = "drifted" if drift_ratio >= settings.drift_threshold else "ok"

    run_id = log_monitor_run(
        engine=engine,
        drift_score=summary["drift_score"],
        drifted_features=summary["drifted_features"],
        total_features=total_features,
        status=status,
        report_path=str(report_path),
    )

    latest_summary = {
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "rows": len(current_df),
        "drift_score": summary["drift_score"],
        "drifted_features": summary["drifted_features"],
        "total_features": total_features,
        "drift_ratio": drift_ratio,
        "status": status,
        "report_path": str(report_path),
    }
    _write_latest(settings, latest_summary)
    return latest_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data/prediction drift monitoring with Evidently.")
    parser.add_argument("--source", choices=["logs", "file"], default="logs",
                        help="logs: predictions the API served (from PostgreSQL); file: a labeled CSV")
    parser.add_argument("--since-hours", type=float, default=24.0, help="logs: look back this many hours")
    parser.add_argument("--min-rows", type=int, default=200, help="logs: minimum logged predictions to run")
    parser.add_argument("--dataset-path", default=get_settings().dataset_path, help="file: labeled CSV")
    parser.add_argument("--current-window-ratio", type=float, default=0.3, help="file: share of rows at the end")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    since = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)
    result = run_monitoring(
        dataset_path=args.dataset_path,
        current_window_ratio=args.current_window_ratio,
        source=args.source,
        since=since,
        min_rows=args.min_rows,
    )
    print(json.dumps(result, indent=2))
