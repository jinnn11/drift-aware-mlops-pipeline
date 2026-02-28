from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, insert, select
from sqlalchemy.engine import Engine

metadata = MetaData()

prediction_logs = Table(
    "prediction_logs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("request_id", String(128), nullable=False),
    Column("model_version", String(128), nullable=False),
    Column("features", JSON, nullable=False),
    Column("prediction", Integer, nullable=False),
    Column("score", Float, nullable=False),
)

monitor_runs = Table(
    "monitor_runs",
    metadata,
    Column("run_id", String(36), primary_key=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("drift_score", Float, nullable=False),
    Column("drifted_features", Integer, nullable=False),
    Column("total_features", Integer, nullable=False),
    Column("status", String(32), nullable=False),
    Column("report_path", String(512), nullable=False),
)

training_runs = Table(
    "training_runs",
    metadata,
    Column("run_id", String(36), primary_key=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("mlflow_run_id", String(128), nullable=False),
    Column("model_version", String(128), nullable=False),
    Column("cv_roc_auc", Float, nullable=False),
    Column("val_roc_auc", Float, nullable=False),
    Column("test_roc_auc", Float, nullable=False),
)



def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, future=True)



def init_db(engine: Engine) -> None:
    metadata.create_all(engine)



def log_prediction_batch(
    engine: Engine,
    request_id: str,
    model_version: str,
    instances: Iterable[Dict[str, Any]],
    predictions: Iterable[int],
    scores: Iterable[float],
) -> None:
    now = datetime.now(timezone.utc)
    rows = []
    for features, pred, score in zip(instances, predictions, scores):
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "ts": now,
                "request_id": request_id,
                "model_version": model_version,
                "features": features,
                "prediction": int(pred),
                "score": float(score),
            }
        )

    with engine.begin() as conn:
        conn.execute(insert(prediction_logs), rows)



def log_monitor_run(
    engine: Engine,
    drift_score: float,
    drifted_features: int,
    total_features: int,
    status: str,
    report_path: str,
) -> str:
    run_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            insert(monitor_runs),
            {
                "run_id": run_id,
                "ts": datetime.now(timezone.utc),
                "drift_score": float(drift_score),
                "drifted_features": int(drifted_features),
                "total_features": int(total_features),
                "status": status,
                "report_path": report_path,
            },
        )
    return run_id



def get_latest_monitor_run(engine: Engine) -> Optional[Dict[str, Any]]:
    stmt = select(monitor_runs).order_by(monitor_runs.c.ts.desc()).limit(1)
    with engine.begin() as conn:
        row = conn.execute(stmt).mappings().first()
    return dict(row) if row else None



def log_training_run(
    engine: Engine,
    mlflow_run_id: str,
    model_version: str,
    cv_roc_auc: float,
    val_roc_auc: float,
    test_roc_auc: float,
) -> str:
    run_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            insert(training_runs),
            {
                "run_id": run_id,
                "ts": datetime.now(timezone.utc),
                "mlflow_run_id": mlflow_run_id,
                "model_version": model_version,
                "cv_roc_auc": float(cv_roc_auc),
                "val_roc_auc": float(val_roc_auc),
                "test_roc_auc": float(test_roc_auc),
            },
        )
    return run_id



def get_best_historical_test_auc(engine: Engine) -> Optional[float]:
    stmt = select(training_runs.c.test_roc_auc).order_by(training_runs.c.test_roc_auc.desc()).limit(1)
    with engine.begin() as conn:
        row = conn.execute(stmt).first()
    return float(row[0]) if row else None
