from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn

from src.config import get_settings
from src.data import load_dataset, prepare_features_and_target, time_ordered_split, validate_dataset
from src.db import get_engine, init_db, log_training_run
from src.evaluation import evaluate_probabilistic_classifier, find_optimal_threshold
from src.features import build_preprocessor
from src.modeling import compute_scale_pos_weight, train_with_cv
from src.registry import save_promoted_model, save_reference_data


def _log_metrics(prefix: str, metrics: dict) -> None:
    for key, value in metrics.items():
        if key == "confusion_matrix":
            for cm_key, cm_value in value.items():
                mlflow.log_metric(f"{prefix}_{cm_key}", cm_value)
            continue
        mlflow.log_metric(f"{prefix}_{key}", value)


def run_training(dataset_path: str, validate_only: bool = False) -> dict:
    settings = get_settings()

    df = load_dataset(dataset_path)
    validate_dataset(df)

    if validate_only:
        return {"status": "validated", "rows": len(df), "columns": len(df.columns)}

    x, y = prepare_features_and_target(df)
    split = time_ordered_split(x, y)

    preprocessor = build_preprocessor(split.x_train)
    scale_pos_weight = compute_scale_pos_weight(split.y_train)
    best_model, best_params, cv_score = train_with_cv(
        split.x_train,
        split.y_train,
        preprocessor,
        scale_pos_weight=scale_pos_weight,
    )

    optimal_threshold = find_optimal_threshold(
        split.y_val,
        best_model.predict_proba(split.x_val)[:, 1],
    )
    val_metrics = evaluate_probabilistic_classifier(best_model, split.x_val, split.y_val, threshold=optimal_threshold)
    test_metrics = evaluate_probabilistic_classifier(best_model, split.x_test, split.y_test, threshold=optimal_threshold)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    model_version = f"xgb-{timestamp}"

    with mlflow.start_run(run_name=model_version) as run:
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("train_rows", len(split.x_train))
        mlflow.log_param("val_rows", len(split.x_val))
        mlflow.log_param("test_rows", len(split.x_test))
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        for key, value in best_params.items():
            mlflow.log_param(key, value)

        mlflow.log_metric("cv_roc_auc", cv_score)
        mlflow.log_metric("optimal_threshold", optimal_threshold)
        _log_metrics("val", val_metrics)
        _log_metrics("test", test_metrics)

        mlflow.sklearn.log_model(best_model, artifact_path="model")

        tmp_dir = Path("/tmp/ml-system-artifacts")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        feature_file = tmp_dir / "feature_list.json"
        metrics_file = tmp_dir / "metrics.json"
        feature_file.write_text(json.dumps({"features": split.x_train.columns.tolist()}, indent=2), encoding="utf-8")
        metrics_file.write_text(
            json.dumps(
                {
                    "cv_roc_auc": cv_score,
                    "optimal_threshold": optimal_threshold,
                    "val": val_metrics,
                    "test": test_metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(feature_file))
        mlflow.log_artifact(str(metrics_file))

        metadata = {
            "model_version": model_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": split.x_train.columns.tolist(),
            "best_params": best_params,
            "cv_roc_auc": cv_score,
            "scale_pos_weight": scale_pos_weight,
            "optimal_threshold": optimal_threshold,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "mlflow_run_id": run.info.run_id,
        }

        save_promoted_model(
            best_model,
            settings.model_path,
            settings.model_metadata_path,
            metadata,
        )

        reference_df = split.x_train.copy()
        reference_df["y"] = split.y_train.values
        reference_df["pred_score"] = best_model.predict_proba(split.x_train)[:, 1]
        save_reference_data(reference_df, settings.reference_data_path)

        engine = get_engine(settings.database_url)
        init_db(engine)
        log_training_run(
            engine=engine,
            mlflow_run_id=run.info.run_id,
            model_version=model_version,
            cv_roc_auc=cv_score,
            val_roc_auc=val_metrics["roc_auc"],
            test_roc_auc=test_metrics["roc_auc"],
        )

    return {
        "status": "trained",
        "model_version": model_version,
        "cv_roc_auc": cv_score,
        "optimal_threshold": optimal_threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register XGBoost model for Bank Marketing.")
    parser.add_argument("--dataset-path", default=get_settings().dataset_path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_training(dataset_path=args.dataset_path, validate_only=args.validate_only)
    print(json.dumps(result, indent=2))
