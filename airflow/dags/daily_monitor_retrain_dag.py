from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator

REPORT_FILE = Path("/app/reports/latest_monitor.json")



def branch_on_drift(**kwargs) -> str:
    if not REPORT_FILE.exists():
        return "publish_metrics"

    data = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    status = data.get("status", "ok")
    return "retrain_model" if status == "drifted" else "publish_metrics"


with DAG(
    dag_id="ml_daily_monitor_and_retrain",
    start_date=datetime(2025, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    tags=["ml", "monitoring", "retraining"],
) as dag:
    extract_recent_data = BashOperator(
        task_id="extract_recent_data",
        bash_command=(
            "python /app/generate_recession_data.py "
            "--source-path /app/data/raw/bank-additional-full.csv "
            "--output-dir /app/data/processed/recession "
            "--recent-ratio 0.35 "
            "--holdout-ratio 0.2"
        ),
    )

    run_drift_monitor = BashOperator(
        task_id="run_drift_monitor",
        bash_command=(
            "python /app/monitor.py "
            "--dataset-path /app/data/processed/recession/recession_incoming_full.csv "
            "--current-window-ratio 1.0"
        ),
    )

    decide_retrain = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=branch_on_drift,
    )

    retrain_model = BashOperator(
        task_id="retrain_model",
        bash_command=(
            "python /app/retrain.py "
            "--dataset-path /app/data/raw/bank-additional-full.csv "
            "--new-data-path /app/data/processed/recession/recession_incoming_full.csv "
            "--new-data-holdout-ratio 0.2 "
            "--new-data-weight 4.0"
        ),
    )

    smoke_test_model = BashOperator(
        task_id="smoke_test_model",
        bash_command="python /app/retrain.py --smoke-test",
    )

    promote_or_rollback = EmptyOperator(task_id="promote_or_rollback")

    publish_metrics = BashOperator(
        task_id="publish_metrics",
        bash_command="cat /app/reports/latest_monitor.json || true",
    )

    extract_recent_data >> run_drift_monitor >> decide_retrain
    decide_retrain >> publish_metrics
    decide_retrain >> retrain_model >> smoke_test_model >> promote_or_rollback >> publish_metrics
