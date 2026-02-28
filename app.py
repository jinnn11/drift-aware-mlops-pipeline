from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from src.config import get_settings
from src.db import get_engine, get_latest_monitor_run, init_db, log_prediction_batch
from src.registry import load_promoted_model
from src.schemas import PredictRequest, PredictResponse, PredictionItem

settings = get_settings()
app = FastAPI(title="Bank Marketing Inference API", version="1.0.0")

PROM_REGISTRY = CollectorRegistry()
REQUEST_COUNTER = Counter("api_prediction_requests_total", "Total prediction requests", registry=PROM_REGISTRY)
PREDICTION_COUNTER = Counter("api_predictions_total", "Total predictions produced", registry=PROM_REGISTRY)
REQUEST_LATENCY = Histogram("api_prediction_latency_seconds", "Prediction endpoint latency", registry=PROM_REGISTRY)
DRIFT_SCORE_GAUGE = Gauge("ml_drift_score", "Latest drift score from monitoring", registry=PROM_REGISTRY)
DRIFT_RATIO_GAUGE = Gauge("ml_drifted_feature_ratio", "Latest drifted feature ratio", registry=PROM_REGISTRY)
LAST_MONITOR_SUCCESS = Gauge(
    "ml_last_monitor_success",
    "Latest monitor run status (1=ok,0=drifted)",
    registry=PROM_REGISTRY,
)

model_lock = Lock()
csv_lock = Lock()
MODEL = None
MODEL_METADATA = {}
FEATURE_COLUMNS = []
ENGINE = None



def _try_load_model() -> bool:
    global MODEL, MODEL_METADATA, FEATURE_COLUMNS
    try:
        MODEL, MODEL_METADATA = load_promoted_model(settings.model_path, settings.model_metadata_path)
        FEATURE_COLUMNS = MODEL_METADATA.get("feature_columns", [])
        return True
    except FileNotFoundError:
        MODEL = None
        MODEL_METADATA = {}
        FEATURE_COLUMNS = []
        return False


def _refresh_drift_metrics() -> None:
    if ENGINE is None:
        return
    latest = get_latest_monitor_run(ENGINE)
    if not latest:
        return
    drift_score = float(latest["drift_score"])
    drift_ratio = float(latest["drifted_features"]) / max(int(latest["total_features"]), 1)
    status = str(latest["status"])

    DRIFT_SCORE_GAUGE.set(drift_score)
    DRIFT_RATIO_GAUGE.set(drift_ratio)
    LAST_MONITOR_SUCCESS.set(1.0 if status == "ok" else 0.0)



def _append_csv_logs(request_id: str, model_version: str, instances, predictions, scores) -> None:
    date_suffix = datetime.now(timezone.utc).strftime("%Y%m%d")
    csv_path = Path(settings.log_dir) / f"predictions_{date_suffix}.csv"
    file_exists = csv_path.exists()

    with csv_lock:
        with open(csv_path, "a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["timestamp", "request_id", "model_version", "features", "prediction", "score"],
            )
            if not file_exists:
                writer.writeheader()

            ts = datetime.now(timezone.utc).isoformat()
            for features, pred, score in zip(instances, predictions, scores):
                writer.writerow(
                    {
                        "timestamp": ts,
                        "request_id": request_id,
                        "model_version": model_version,
                        "features": json.dumps(features),
                        "prediction": int(pred),
                        "score": float(score),
                    }
                )


@app.on_event("startup")
def startup_event() -> None:
    global ENGINE

    ENGINE = get_engine(settings.database_url)
    init_db(ENGINE)
    _try_load_model()
    _refresh_drift_metrics()


@app.get("/health")
def health() -> dict:
    if MODEL is None:
        _try_load_model()
    model_loaded = MODEL is not None
    model_version = MODEL_METADATA.get("model_version", "unavailable")
    return {"status": "ok", "model_loaded": model_loaded, "model_version": model_version}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    REQUEST_COUNTER.inc()
    start = datetime.now(timezone.utc)

    if MODEL is None and not _try_load_model():
        raise HTTPException(status_code=503, detail="Model not loaded")

    instances = payload.instances
    data = pd.DataFrame(instances)

    missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required feature columns: {missing}")

    with model_lock:
        probs = MODEL.predict_proba(data[FEATURE_COLUMNS])[:, 1]

    predictions = (probs >= 0.5).astype(int).tolist()
    scores = probs.tolist()

    request_id = str(uuid.uuid4())
    model_version = MODEL_METADATA.get("model_version", "unknown")

    log_prediction_batch(
        engine=ENGINE,
        request_id=request_id,
        model_version=model_version,
        instances=instances,
        predictions=predictions,
        scores=scores,
    )
    _append_csv_logs(request_id, model_version, instances, predictions, scores)

    PREDICTION_COUNTER.inc(len(predictions))
    duration = (datetime.now(timezone.utc) - start).total_seconds()
    REQUEST_LATENCY.observe(duration)
    _refresh_drift_metrics()

    return PredictResponse(
        predictions=[PredictionItem(label=int(p), score=float(s)) for p, s in zip(predictions, scores)],
        model_version=model_version,
        request_id=request_id,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(PROM_REGISTRY), media_type=CONTENT_TYPE_LATEST)
