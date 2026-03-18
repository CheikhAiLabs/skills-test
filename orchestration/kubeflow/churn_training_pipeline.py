from __future__ import annotations

from kfp import dsl


@dsl.component(base_image="python:3.11-slim")
def bootstrap_data(rows: int = 5000) -> None:
    import subprocess

    subprocess.run(
        ["python", "-m", "churn_ml.cli", "bootstrap-data", "--rows", str(rows)],
        check=True,
    )


@dsl.component(base_image="python:3.11-slim")
def train_pipeline() -> None:
    import subprocess

    subprocess.run(["python", "-m", "churn_ml.cli", "run"], check=True)


@dsl.component(base_image="python:3.11-slim")
def report_metrics() -> None:
    import subprocess

    subprocess.run(["python", "-m", "churn_ml.cli", "report"], check=True)


@dsl.pipeline(name="churn-training-pipeline")
def churn_training_pipeline(rows: int = 5000) -> None:
    bootstrap_task = bootstrap_data(rows=rows)
    training_task = train_pipeline().after(bootstrap_task)
    report_metrics().after(training_task)
