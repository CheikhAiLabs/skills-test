from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from churn_ml.config import PipelineConfig


class MLflowTracker:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.enabled = config.mlflow.enabled
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", str(config.mlflow.tracking_uri))
        self._autolog_enabled = False

        if self.enabled:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(config.training.experiment_name)

    def enable_sklearn_autolog(self) -> None:
        if not self.enabled or self._autolog_enabled:
            return
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True,
            log_datasets=True,
            exclusive=False,
            disable_for_unsupported_versions=False,
            silent=False,
            log_post_training_metrics=True,
        )
        self._autolog_enabled = True

    def airflow_tags(self) -> dict[str, str]:
        context_keys = [
            "AIRFLOW_CTX_DAG_ID",
            "AIRFLOW_CTX_TASK_ID",
            "AIRFLOW_CTX_DAG_RUN_ID",
            "AIRFLOW_CTX_TRY_NUMBER",
            "AIRFLOW_CTX_EXECUTION_DATE",
        ]
        return {
            key.lower().replace("airflow_ctx_", "airflow_"): value
            for key, value in ((key, os.getenv(key)) for key in context_keys)
            if value
        }

    def start_run(
        self,
        *,
        run_name: str,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ):
        if not self.enabled:
            return _NoOpRun()
        merged_tags = {"pipeline": "churn_prediction", **self.airflow_tags(), **(tags or {})}
        return mlflow.start_run(run_name=run_name, nested=nested, tags=merged_tags)

    def log_params(self, payload: dict[str, Any], prefix: str | None = None) -> None:
        if not self.enabled:
            return
        params = {}
        for key, value in payload.items():
            param_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    params[f"{param_key}.{child_key}"] = child_value
            else:
                params[param_key] = value
        if params:
            mlflow.log_params({key: str(value) for key, value in params.items()})

    def log_metrics(self, payload: dict[str, Any], prefix: str | None = None) -> None:
        if not self.enabled:
            return
        metrics = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                metric_key = f"{prefix}.{key}" if prefix else key
                metrics[metric_key] = float(value)
        if metrics:
            mlflow.log_metrics(metrics)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.enabled and tags:
            mlflow.set_tags({key: str(value) for key, value in tags.items()})

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        if self.enabled:
            mlflow.log_dict(payload, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        if self.enabled:
            mlflow.log_text(text, artifact_file)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self.enabled:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_model(
        self,
        *,
        model: Any,
        artifact_name: str,
        input_example: Any,
        predictions: Any,
        registered_model_name: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        signature = infer_signature(input_example, predictions)
        kwargs = {}
        if registered_model_name and self.tracking_uri.startswith(("http://", "https://")):
            kwargs["registered_model_name"] = registered_model_name
        mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_name,
            input_example=input_example,
            signature=signature,
            **kwargs,
        )

    def active_run_id(self) -> str | None:
        if not self.enabled:
            return None
        active_run = mlflow.active_run()
        return active_run.info.run_id if active_run else None

    def log_json_text(self, payload: dict[str, Any], artifact_file: str) -> None:
        self.log_text(json.dumps(payload, indent=2, sort_keys=True), artifact_file)


class _NoOpRun:
    def __enter__(self) -> "_NoOpRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False
