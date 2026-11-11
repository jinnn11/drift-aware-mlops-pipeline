import importlib
import json
from datetime import datetime, timedelta, timezone

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
joblib = pytest.importorskip("joblib")
pytest.importorskip("sqlalchemy")

from src.db import fetch_logged_predictions, get_engine, init_db, log_prediction_batch


class DummyModel:
    def predict_proba(self, x):
        probs = np.full(len(x), 0.3)
        return np.column_stack([1.0 - probs, probs])


def _instances(n: int, age_offset: int, rng) -> list[dict]:
    jobs = ["admin.", "services", "technician"]
    return [{"age": int(30 + age_offset + rng.integers(0, 10)), "job": jobs[i % 3]} for i in range(n)]


def test_logged_predictions_round_trip_and_time_filter(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'logs.db'}")
    init_db(engine)
    log_prediction_batch(engine, "r1", "v1", [{"age": 40, "job": "admin."}], [1], [0.8])
    cutoff = datetime.now(timezone.utc) + timedelta(seconds=1)

    rows = fetch_logged_predictions(engine)
    assert len(rows) == 1
    assert rows[0]["features"] == {"age": 40, "job": "admin."}
    assert rows[0]["score"] == pytest.approx(0.8)
    assert fetch_logged_predictions(engine, since=cutoff) == []


def _monitor_module(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    model_path, metadata_path = tmp_path / "model.pkl", tmp_path / "metadata.json"
    reference_path, reports = tmp_path / "reference.csv", tmp_path / "reports"
    reports.mkdir()
    joblib.dump(DummyModel(), model_path)
    metadata_path.write_text(json.dumps({"model_version": "v1", "feature_columns": ["age", "job"]}), encoding="utf-8")
    reference = pd.DataFrame(_instances(400, 0, rng))
    reference["pred_score"] = rng.uniform(0.2, 0.4, len(reference))
    reference["y"] = rng.integers(0, 2, len(reference))
    reference.to_csv(reference_path, index=False)

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("REFERENCE_DATA_PATH", str(reference_path))
    monkeypatch.setenv("REPORT_DIR", str(reports))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'mlops.db'}")
    import src.config

    importlib.reload(src.config)
    monitor = importlib.reload(importlib.import_module("monitor"))
    engine = get_engine(f"sqlite:///{tmp_path / 'mlops.db'}")
    init_db(engine)
    return monitor, engine, rng


def test_monitor_skips_when_too_few_predictions_were_logged(tmp_path, monkeypatch):
    monitor, engine, rng = _monitor_module(tmp_path, monkeypatch)
    log_prediction_batch(engine, "r1", "v1", _instances(10, 0, rng), [0] * 10, [0.3] * 10)

    summary = monitor.run_monitoring(source="logs", min_rows=50)

    assert summary["status"] == "insufficient_data"
    assert summary["rows"] == 10


def test_monitor_flags_drift_in_logged_traffic(tmp_path, monkeypatch):
    monitor, engine, rng = _monitor_module(tmp_path, monkeypatch)
    shifted = _instances(300, 40, rng)  # much older customers than in training
    log_prediction_batch(engine, "r1", "v1", shifted, [1] * 300, list(rng.uniform(0.7, 0.9, 300)))

    summary = monitor.run_monitoring(source="logs", min_rows=50)

    assert summary["source"] == "logs"
    assert summary["rows"] == 300
    assert summary["total_features"] == 3  # age, job and the model's score; no labels in live traffic
    assert summary["drifted_features"] >= 2
    assert summary["status"] == "drifted"
