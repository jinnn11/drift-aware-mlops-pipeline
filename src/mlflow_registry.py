"""Champion / rollback bookkeeping on the MLflow model registry using aliases.

Serving loads `models:/<name>@champion`. Promotion moves the `champion` alias to the new
version and points `previous_champion` at the old one, so a rollback is a single alias swap
with no retraining and no file copies.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

CHAMPION = "champion"
PREVIOUS = "previous_champion"


class MlflowChampionRegistry:
    def __init__(self, model_name: str, tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_registry_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        self.model_name = model_name
        self.client = MlflowClient()

    def register(self, model, run_name: str, params: Dict[str, Any], metrics: Dict[str, float],
                 tags: Optional[Dict[str, str]] = None) -> str:
        """Log a candidate as a run and a new registered-model version; returns the version."""
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
            if tags:
                mlflow.set_tags(tags)
            mlflow.sklearn.log_model(model, artifact_path="model")
        return self.register_run(run.info.run_id)

    def register_run(self, run_id: str, artifact_path: str = "model") -> str:
        """Register a model already logged under `run_id`; returns the new version."""
        return str(mlflow.register_model(f"runs:/{run_id}/{artifact_path}", self.model_name).version)

    def alias_version(self, alias: str) -> Optional[str]:
        try:
            return str(self.client.get_model_version_by_alias(self.model_name, alias).version)
        except MlflowException:
            return None

    def promote(self, version: str) -> None:
        current = self.alias_version(CHAMPION)
        if current is not None and current != version:
            self.client.set_registered_model_alias(self.model_name, PREVIOUS, current)
        self.client.set_registered_model_alias(self.model_name, CHAMPION, version)
        self.client.set_model_version_tag(self.model_name, version, "promoted", "true")

    def rollback(self) -> str:
        """Point `champion` back at `previous_champion` (and vice versa, so it can be undone)."""
        current, previous = self.alias_version(CHAMPION), self.alias_version(PREVIOUS)
        if previous is None:
            raise RuntimeError(f"No '{PREVIOUS}' alias on {self.model_name}; nothing to roll back to")
        self.client.set_registered_model_alias(self.model_name, CHAMPION, previous)
        if current is not None:
            self.client.set_registered_model_alias(self.model_name, PREVIOUS, current)
        return previous

    def load_champion(self):
        return mlflow.sklearn.load_model(f"models:/{self.model_name}@{CHAMPION}")

    def champion_metadata(self) -> Dict[str, Any]:
        """The `model_metadata.json` logged with the champion's run (threshold, feature columns...)."""
        version = self.client.get_model_version_by_alias(self.model_name, CHAMPION)
        return mlflow.artifacts.load_dict(f"runs:/{version.run_id}/model_metadata.json")
