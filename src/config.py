from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    timezone: str = os.getenv("TZ", "UTC")

    dataset_path: str = os.getenv("DATASET_PATH", "data/raw/bank-additional-full.csv")
    reference_data_path: str = os.getenv("REFERENCE_DATA_PATH", "data/processed/reference.csv")
    model_path: str = os.getenv("MODEL_PATH", "models/model.pkl")
    model_metadata_path: str = os.getenv("MODEL_METADATA_PATH", "models/model_metadata.json")
    log_dir: str = os.getenv("LOG_DIR", "logs")
    report_dir: str = os.getenv("REPORT_DIR", "reports")

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "bank_marketing_xgboost")

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://mlops:mlops@localhost:5432/mlops",
    )

    drift_threshold: float = float(os.getenv("DRIFT_THRESHOLD", "0.30"))
    retrain_improvement_threshold: float = float(os.getenv("RETRAIN_IMPROVEMENT_THRESHOLD", "0.005"))
    degradation_threshold: float = float(os.getenv("DEGRADATION_THRESHOLD", "0.03"))

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    def ensure_dirs(self) -> None:
        for path in [
            Path(self.model_path).parent,
            Path(self.model_metadata_path).parent,
            Path(self.log_dir),
            Path(self.report_dir),
            Path(self.reference_data_path).parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
