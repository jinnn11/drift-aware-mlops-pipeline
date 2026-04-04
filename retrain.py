from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.config import get_settings
from src.data import load_dataset, prepare_features_and_target, time_ordered_split
from src.db import get_best_historical_test_auc, get_engine, get_latest_monitor_run, init_db, log_training_run
from src.evaluation import evaluate_probabilistic_classifier, find_optimal_threshold
from src.features import build_preprocessor
from src.modeling import compute_scale_pos_weight, train_with_cv
from src.policy import should_retrain
from src.registry import load_promoted_model, save_promoted_model, save_reference_data


def _log_metrics(prefix: str, metrics: dict | None) -> None:
    if metrics is None:
        return

    for key, value in metrics.items():
        if key == "confusion_matrix":
            for cm_key, cm_value in value.items():
                mlflow.log_metric(f"{prefix}_{cm_key}", cm_value)
            continue
        mlflow.log_metric(f"{prefix}_{key}", value)


def run_smoke_test(dataset_path: str) -> dict:
    settings = get_settings()
    model, metadata = load_promoted_model(settings.model_path, settings.model_metadata_path)
    feature_columns = metadata["feature_columns"]

    df = load_dataset(dataset_path)
    x, _ = prepare_features_and_target(df)
    sample = x[feature_columns].head(5)
    scores = model.predict_proba(sample)[:, 1].tolist()
    return {
        "status": "ok",
        "model_version": metadata.get("model_version", "unknown"),
        "sample_scores": scores,
    }


def _split_recent_labeled_data(df: pd.DataFrame, holdout_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_rows = max(300, int(len(df) * holdout_ratio))
    if len(df) <= eval_rows:
        raise ValueError("Recent labeled dataset is too small for requested holdout ratio")
    recent_train = df.iloc[:-eval_rows].copy()
    recent_eval = df.iloc[-eval_rows:].copy()
    return recent_train, recent_eval


def _combine_historical_and_recent(
    historical_df: pd.DataFrame,
    recent_train_df: pd.DataFrame,
    recent_weight: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    combined_df = pd.concat([historical_df, recent_train_df], axis=0, ignore_index=True)
    weights = np.ones(len(combined_df), dtype=float)
    weights[len(historical_df) :] = float(recent_weight)
    return combined_df, weights


def run_retrain(
    dataset_path: str,
    force: bool = False,
    new_data_path: str | None = None,
    new_data_holdout_ratio: float = 0.2,
    new_data_weight: float = 3.0,
) -> dict:
    settings = get_settings()
    engine = get_engine(settings.database_url)
    init_db(engine)

    latest_monitor = get_latest_monitor_run(engine)
    if not latest_monitor and not force:
        return {"status": "skipped", "reason": "no_monitor_run"}

    if not force:
        if not should_retrain(
            drifted_features=int(latest_monitor["drifted_features"]),
            total_features=int(latest_monitor["total_features"]),
            threshold=settings.drift_threshold,
        ):
            return {
                "status": "skipped",
                "reason": "drift_below_threshold",
                "drift_ratio": int(latest_monitor["drifted_features"]) / max(int(latest_monitor["total_features"]), 1),
            }

    historical_df = load_dataset(dataset_path)
    historical_x, historical_y = prepare_features_and_target(historical_df)
    historical_split = time_ordered_split(historical_x, historical_y)

    recent_train_df = None
    recent_eval_df = None
    if new_data_path:
        path_obj = Path(new_data_path)
        if path_obj.exists():
            recent_full_df = load_dataset(str(path_obj))
            recent_train_df, recent_eval_df = _split_recent_labeled_data(recent_full_df, new_data_holdout_ratio)

    if recent_train_df is not None:
        combined_df, sample_weights = _combine_historical_and_recent(
            historical_df=historical_df,
            recent_train_df=recent_train_df,
            recent_weight=new_data_weight,
        )
        training_mode = "historical_plus_recent_weighted"
    else:
        combined_df = historical_df.copy()
        sample_weights = np.ones(len(combined_df), dtype=float)
        training_mode = "historical_only"

    train_x, train_y = prepare_features_and_target(combined_df)
    preprocessor = build_preprocessor(train_x)
    scale_pos_weight = compute_scale_pos_weight(train_y)
    candidate_model, best_params, cv_score = train_with_cv(
        x_train=train_x,
        y_train=train_y,
        preprocessor=preprocessor,
        scale_pos_weight=scale_pos_weight,
        sample_weight=sample_weights,
    )

    candidate_recent_metrics = None
    recent_eval_x = None
    recent_eval_y = None
    if recent_eval_df is not None:
        recent_eval_x, recent_eval_y = prepare_features_and_target(recent_eval_df)

    threshold_x = recent_eval_x if recent_eval_x is not None else historical_split.x_val
    threshold_y = recent_eval_y if recent_eval_y is not None else historical_split.y_val
    candidate_optimal_threshold = find_optimal_threshold(
        threshold_y,
        candidate_model.predict_proba(threshold_x)[:, 1],
    )
    candidate_hist_metrics = evaluate_probabilistic_classifier(
        candidate_model,
        historical_split.x_test,
        historical_split.y_test,
        threshold=candidate_optimal_threshold,
    )
    if recent_eval_x is not None and recent_eval_y is not None:
        candidate_recent_metrics = evaluate_probabilistic_classifier(
            candidate_model,
            recent_eval_x,
            recent_eval_y,
            threshold=candidate_optimal_threshold,
        )

    current_metrics = None
    current_recent_metrics = None
    current_threshold = 0.5
    try:
        current_model, current_metadata = load_promoted_model(settings.model_path, settings.model_metadata_path)
        current_threshold = float(current_metadata.get("optimal_threshold", 0.5))
        current_metrics = evaluate_probabilistic_classifier(
            current_model,
            historical_split.x_test,
            historical_split.y_test,
            threshold=current_threshold,
        )
        current_auc = current_metrics["roc_auc"]
    except FileNotFoundError:
        current_auc = -1.0
        current_metadata = {"model_version": "none"}
        current_model = None

    if recent_eval_x is not None and recent_eval_y is not None and current_model is not None:
        current_recent_metrics = evaluate_probabilistic_classifier(
            current_model,
            recent_eval_x,
            recent_eval_y,
            threshold=current_threshold,
        )

    historical_best = get_best_historical_test_auc(engine)
    if historical_best is None:
        historical_best = current_auc if current_auc >= 0 else candidate_hist_metrics["roc_auc"]

    if candidate_recent_metrics is not None and current_recent_metrics is not None:
        improved_recent = (
            candidate_recent_metrics["pr_auc"]
            >= current_recent_metrics["pr_auc"] + settings.retrain_improvement_threshold
        )
        base_guard = candidate_hist_metrics["roc_auc"] >= (
            current_metrics["roc_auc"] - settings.degradation_threshold
        )
        accepted = (current_auc < 0) or (improved_recent and base_guard)
    else:
        improved_enough = candidate_hist_metrics["roc_auc"] >= current_auc + settings.retrain_improvement_threshold
        degraded_current = (historical_best - current_auc) >= settings.degradation_threshold
        acceptable_recovery = degraded_current and (candidate_hist_metrics["roc_auc"] >= current_auc)
        accepted = (current_auc < 0) or improved_enough or acceptable_recovery

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    model_version = f"xgb-retrain-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    with mlflow.start_run(run_name=model_version) as run:
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("retrain_trigger", "drift")
        mlflow.log_param("force_retrain", force)
        mlflow.log_param("training_mode", training_mode)
        mlflow.log_param("historical_rows", len(historical_df))
        mlflow.log_param("recent_rows", len(recent_train_df) if recent_train_df is not None else 0)
        mlflow.log_param("recent_weight", float(new_data_weight))
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        for key, value in best_params.items():
            mlflow.log_param(key, value)

        mlflow.log_metric("cv_roc_auc", cv_score)
        mlflow.log_metric("candidate_optimal_threshold", candidate_optimal_threshold)
        _log_metrics("candidate_hist_test", candidate_hist_metrics)
        if current_auc >= 0:
            _log_metrics("current_hist_test", current_metrics)
        _log_metrics("candidate_recent_test", candidate_recent_metrics)
        _log_metrics("current_recent_test", current_recent_metrics)
        mlflow.log_metric("accepted", 1 if accepted else 0)

        mlflow.sklearn.log_model(candidate_model, artifact_path="model_candidate")

        if accepted:
            metadata = {
                "model_version": model_version,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "feature_columns": train_x.columns.tolist(),
                "best_params": best_params,
                "cv_roc_auc": cv_score,
                "scale_pos_weight": scale_pos_weight,
                "optimal_threshold": candidate_optimal_threshold,
                "historical_test_metrics": candidate_hist_metrics,
                "recent_unseen_test_metrics": candidate_recent_metrics,
                "mlflow_run_id": run.info.run_id,
                "retrained": True,
                "previous_model_version": current_metadata.get("model_version", "none"),
                "training_mode": training_mode,
                "recent_weight": float(new_data_weight),
            }
            save_promoted_model(candidate_model, settings.model_path, settings.model_metadata_path, metadata)

            reference_df = train_x.copy()
            reference_df["y"] = train_y.values
            reference_df["pred_score"] = candidate_model.predict_proba(train_x)[:, 1]
            save_reference_data(reference_df, settings.reference_data_path)

            chosen_test_auc = (
                candidate_recent_metrics["roc_auc"]
                if candidate_recent_metrics is not None
                else candidate_hist_metrics["roc_auc"]
            )
            log_training_run(
                engine=engine,
                mlflow_run_id=run.info.run_id,
                model_version=model_version,
                cv_roc_auc=cv_score,
                val_roc_auc=candidate_hist_metrics["roc_auc"],
                test_roc_auc=chosen_test_auc,
            )

    improvement_recent_roc_auc = None
    improvement_recent_pr_auc = None
    if candidate_recent_metrics is not None and current_recent_metrics is not None:
        improvement_recent_roc_auc = candidate_recent_metrics["roc_auc"] - current_recent_metrics["roc_auc"]
        improvement_recent_pr_auc = candidate_recent_metrics["pr_auc"] - current_recent_metrics["pr_auc"]

    return {
        "status": "accepted" if accepted else "rejected",
        "current_model_version": current_metadata.get("model_version", "none"),
        "candidate_model_version": model_version,
        "training_mode": training_mode,
        "candidate_optimal_threshold": candidate_optimal_threshold,
        "current_historical_test_metrics": current_metrics,
        "candidate_historical_test_metrics": candidate_hist_metrics,
        "current_recent_unseen_metrics": current_recent_metrics,
        "candidate_recent_unseen_metrics": candidate_recent_metrics,
        "recent_unseen_roc_auc_delta": improvement_recent_roc_auc,
        "recent_unseen_pr_auc_delta": improvement_recent_pr_auc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditionally retrain model based on drift and quality gates.")
    parser.add_argument("--dataset-path", default=get_settings().dataset_path)
    parser.add_argument("--new-data-path", default="data/processed/recession/recession_incoming_full.csv")
    parser.add_argument("--new-data-holdout-ratio", type=float, default=0.2)
    parser.add_argument("--new-data-weight", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        result = run_smoke_test(args.dataset_path)
    else:
        result = run_retrain(
            dataset_path=args.dataset_path,
            force=args.force,
            new_data_path=args.new_data_path,
            new_data_holdout_ratio=args.new_data_holdout_ratio,
            new_data_weight=args.new_data_weight,
        )
    print(json.dumps(result, indent=2))
