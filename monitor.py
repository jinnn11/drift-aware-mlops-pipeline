from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import get_settings
from src.data import load_dataset, prepare_features_and_target
from src.db import get_engine, init_db, log_monitor_run
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



def run_monitoring(dataset_path: str, current_window_ratio: float = 0.3) -> dict:
    settings = get_settings()

    reference_df = load_reference_data(settings.reference_data_path)
    model, metadata = load_promoted_model(settings.model_path, settings.model_metadata_path)
    feature_columns = metadata["feature_columns"]

    raw_df = load_dataset(dataset_path)
    x, y = prepare_features_and_target(raw_df)
    current_window = max(1000, int(len(x) * current_window_ratio))

    current_df = x.tail(current_window).copy()
    current_df["y"] = y.tail(current_window).values
    current_df["pred_score"] = model.predict_proba(current_df[feature_columns])[:, 1]

    monitored_columns = feature_columns + ["pred_score", "y"]
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

    engine = get_engine(settings.database_url)
    init_db(engine)
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
        "drift_score": summary["drift_score"],
        "drifted_features": summary["drifted_features"],
        "total_features": total_features,
        "drift_ratio": drift_ratio,
        "status": status,
        "report_path": str(report_path),
    }
    Path(settings.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(settings.report_dir) / "latest_monitor.json").write_text(
        json.dumps(latest_summary, indent=2),
        encoding="utf-8",
    )

    return latest_summary



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data/prediction drift monitoring with Evidently.")
    parser.add_argument("--dataset-path", default=get_settings().dataset_path)
    parser.add_argument("--current-window-ratio", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_monitoring(args.dataset_path, args.current_window_ratio)
    print(json.dumps(result, indent=2))
