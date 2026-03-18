from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


with DAG(
    dag_id="churn_training_pipeline",
    description="Daily churn prediction training and packaging workflow",
    default_args=default_args,
    schedule="0 2 * * *",
    start_date=datetime(2026, 3, 18),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "churn"],
) as dag:
    bootstrap_data = BashOperator(
        task_id="bootstrap_data_if_missing",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/project; "
            "(test -f data/raw/churn_customers.csv || python -m churn_ml.cli bootstrap-data --rows 5000); "
            "python - <<'PY'\n"
            "import pandas as pd\n"
            "frame = pd.read_csv('data/raw/churn_customers.csv')\n"
            "print({'rows': len(frame), 'columns': len(frame.columns), 'positive_rate': round(float(frame['churned'].mean()), 4)})\n"
            "PY"
        ),
    )

    run_training = BashOperator(
        task_id="retrain_candidates",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/project; "
            "python -m churn_ml.cli retrain --rounds {{ dag_run.conf.get('retrain_rounds', 3) }}; "
            "echo '--- RETRAINING SUMMARY ---'; "
            "cat reports/retraining_summary.md; "
            "echo '--- LATEST REPORT ---'; "
            "cat reports/latest_evaluation.md"
        ),
    )

    report_metrics = BashOperator(
        task_id="report_latest_metrics",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/project; "
            "python -m churn_ml.cli report; "
            "echo '--- TRACKING TAIL ---'; "
            "tail -n 5 tracking/runs.jsonl"
        ),
    )

    bootstrap_data >> run_training >> report_metrics
