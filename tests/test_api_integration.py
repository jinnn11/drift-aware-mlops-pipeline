import importlib
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

joblib = pytest.importorskip("joblib")
np = pytest.importorskip("numpy")


class DummyModel:
    def predict_proba(self, x):
        n = len(x)
        probs = np.full(n, 0.7)
        return np.column_stack([1.0 - probs, probs])



def test_predict_endpoint_roundtrip(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "metadata.json"
    logs_dir = tmp_path / "logs"
    reports_dir = tmp_path / "reports"
    db_file = tmp_path / "api.db"

    logs_dir.mkdir()
    reports_dir.mkdir()

    joblib.dump(DummyModel(), model_path)
    metadata_path.write_text(
        json.dumps({"model_version": "dummy-v1", "feature_columns": ["age", "job"]}),
        encoding="utf-8",
    )

    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(metadata_path))
    monkeypatch.setenv("LOG_DIR", str(logs_dir))
    monkeypatch.setenv("REPORT_DIR", str(reports_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    app_module = importlib.import_module("app")
    app_module = importlib.reload(app_module)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/predict",
            json={"instances": [{"age": 32, "job": "admin."}, {"age": 44, "job": "services"}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "dummy-v1"
    assert len(body["predictions"]) == 2
    assert "request_id" in body
