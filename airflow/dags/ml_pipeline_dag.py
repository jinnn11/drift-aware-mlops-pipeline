from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="ml_initial_training",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "training"],
) as dag:
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command="python /app/train.py --validate-only",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python /app/train.py",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="python /app/retrain.py --smoke-test",
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="echo 'Model promotion is completed in train.py by design.'",
    )

    validate_data >> train_model >> evaluate_model >> register_model
