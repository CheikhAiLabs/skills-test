from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    raw_data_path: Path
    artifact_root: Path
    tracking_path: Path
    report_path: Path
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    random_state: int


@dataclass(frozen=True)
class ValidationConfig:
    target_column: str
    entity_id_column: str
    max_missing_fraction: float
    min_rows: int
    positive_class_min_rate: float
    positive_class_max_rate: float


@dataclass(frozen=True)
class ModelConfig:
    numeric_features: list[str]
    categorical_features: list[str]
    boolean_features: list[str]
    logistic_regression: dict[str, Any]

    @property
    def feature_columns(self) -> list[str]:
        return [
            *self.numeric_features,
            *self.categorical_features,
            *self.boolean_features,
        ]


@dataclass(frozen=True)
class ThresholdConfig:
    min_precision: float
    min_recall: float


@dataclass(frozen=True)
class MetricGateConfig:
    min_roc_auc: float
    min_average_precision: float
    min_f1: float
    min_recall: float


@dataclass(frozen=True)
class PackagingConfig:
    sample_payload_rows: int


@dataclass(frozen=True)
class TrainingConfig:
    experiment_name: str
    registered_model_name: str
    selection_metric: str
    retrain_rounds: int
    log_model_candidates: bool
    candidate_models: list[dict[str, Any]]


@dataclass(frozen=True)
class MlflowConfig:
    enabled: bool
    tracking_uri: Path | str


@dataclass(frozen=True)
class PipelineConfig:
    run_name_prefix: str
    data: DataConfig
    validation: ValidationConfig
    model: ModelConfig
    training: TrainingConfig
    mlflow: MlflowConfig
    thresholding: ThresholdConfig
    metrics: MetricGateConfig
    packaging: PackagingConfig


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_mlflow_tracking_uri(base_dir: Path, value: str) -> Path | str:
    if "://" in value:
        return value
    return _resolve_path(base_dir, value)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    base_dir = config_path.parent.parent
    data = raw["data"]
    training = raw.get("training", {})
    mlflow = raw.get("mlflow", {})
    return PipelineConfig(
        run_name_prefix=raw["run_name_prefix"],
        data=DataConfig(
            raw_data_path=_resolve_path(base_dir, data["raw_data_path"]),
            artifact_root=_resolve_path(base_dir, data["artifact_root"]),
            tracking_path=_resolve_path(base_dir, data["tracking_path"]),
            report_path=_resolve_path(base_dir, data["report_path"]),
            train_fraction=data["train_fraction"],
            validation_fraction=data["validation_fraction"],
            test_fraction=data["test_fraction"],
            random_state=data["random_state"],
        ),
        validation=ValidationConfig(**raw["validation"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(
            experiment_name=training.get("experiment_name", "churn-prediction"),
            registered_model_name=training.get(
                "registered_model_name", "churn_prediction_model"
            ),
            selection_metric=training.get("selection_metric", "roc_auc"),
            retrain_rounds=training.get("retrain_rounds", 1),
            log_model_candidates=training.get("log_model_candidates", True),
            candidate_models=training.get(
                "candidate_models",
                [
                    {
                        "name": "default_logistic",
                        "estimator": "logistic_regression",
                        "params": raw["model"]["logistic_regression"],
                    }
                ],
            ),
        ),
        mlflow=MlflowConfig(
            enabled=mlflow.get("enabled", True),
            tracking_uri=_resolve_mlflow_tracking_uri(
                base_dir, mlflow.get("tracking_uri", "mlruns")
            ),
        ),
        thresholding=ThresholdConfig(**raw["thresholding"]),
        metrics=MetricGateConfig(**raw["metrics"]),
        packaging=PackagingConfig(**raw["packaging"]),
    )
