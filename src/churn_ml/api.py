from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from churn_ml.artifacts import find_latest_bundle, load_bundle
from churn_ml.config import load_config
from churn_ml.inference import prepare_inference_frame


class PredictionRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1)


class PredictionRow(BaseModel):
    churn_probability: float
    predicted_churn: bool
    threshold: float


class PredictionResponse(BaseModel):
    predictions: list[PredictionRow]
    model_run_id: str


def _runtime_config_path() -> Path:
    return Path(os.getenv("PIPELINE_CONFIG_PATH", "configs/pipeline.yaml")).resolve()


def _cors_origins() -> list[str]:
    raw_value = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:9634,http://127.0.0.1:9634",
    )
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


class ModelService:
    def __init__(self) -> None:
        self.runtime_config = load_config(_runtime_config_path())
        bundle_path = self._resolve_bundle_path()
        self.bundle_path = bundle_path
        self.bundle = load_bundle(bundle_path)

    def _resolve_bundle_path(self) -> Path:
        explicit_path = os.getenv("MODEL_BUNDLE_PATH")
        if explicit_path:
            return Path(explicit_path).resolve()

        return find_latest_bundle(self.runtime_config.data.artifact_root)

    def _feature_groups(self) -> dict[str, list[str]]:
        return {
            "numeric": self.bundle.get(
                "numeric_features", self.runtime_config.model.numeric_features
            ),
            "categorical": self.bundle.get(
                "categorical_features", self.runtime_config.model.categorical_features
            ),
            "boolean": self.bundle.get(
                "boolean_features", self.runtime_config.model.boolean_features
            ),
        }

    def _example_row(self) -> dict[str, Any] | None:
        sample_payload_path = self.bundle_path.parent / "sample_payload.json"
        if not sample_payload_path.exists():
            return None
        try:
            payload = json.loads(sample_payload_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not payload:
            return None
        return payload[0]

    def _feature_schema(self) -> dict[str, list[dict[str, Any]]]:
        feature_groups = self._feature_groups()
        model = self.bundle["model"]
        preprocessor = model.named_steps["preprocessor"]

        numeric_columns = [
            *feature_groups["numeric"],
            *feature_groups["boolean"],
        ]
        numeric_pipeline = preprocessor.named_transformers_["numeric"]
        numeric_imputer = numeric_pipeline.named_steps["imputer"]
        numeric_defaults = {
            feature: value
            for feature, value in zip(numeric_columns, numeric_imputer.statistics_, strict=True)
        }

        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        categorical_imputer = categorical_pipeline.named_steps["imputer"]
        categorical_encoder = categorical_pipeline.named_steps["encoder"]
        categorical_defaults = {
            feature: value
            for feature, value in zip(
                feature_groups["categorical"],
                categorical_imputer.statistics_,
                strict=True,
            )
        }
        categorical_options = {
            feature: [str(option) for option in options.tolist()]
            for feature, options in zip(
                feature_groups["categorical"],
                categorical_encoder.categories_,
                strict=True,
            )
        }

        return {
            "numeric": [
                {
                    "name": feature,
                    "label": feature.replace("_", " ").title(),
                    "default": round(float(numeric_defaults[feature]), 2),
                }
                for feature in feature_groups["numeric"]
            ],
            "categorical": [
                {
                    "name": feature,
                    "label": feature.replace("_", " ").title(),
                    "default": str(categorical_defaults[feature]),
                    "options": categorical_options[feature],
                }
                for feature in feature_groups["categorical"]
            ],
            "boolean": [
                {
                    "name": feature,
                    "label": feature.replace("_", " ").title(),
                    "default": bool(round(float(numeric_defaults[feature]))),
                }
                for feature in feature_groups["boolean"]
            ],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "bundle_path": str(self.bundle_path),
            "run_id": self.bundle["run_id"],
            "threshold": self.bundle["threshold"],
            "metrics": self.bundle["metrics"]["test"],
            "feature_schema": self._feature_schema(),
            "example_row": self._example_row(),
        }

    def predict(self, rows: list[dict[str, Any]]) -> PredictionResponse:
        feature_groups = self._feature_groups()
        try:
            frame = prepare_inference_frame(
                rows,
                feature_columns=self.bundle["feature_columns"],
                numeric_features=feature_groups["numeric"],
                categorical_features=feature_groups["categorical"],
                boolean_features=feature_groups["boolean"],
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        probabilities = self.bundle["model"].predict_proba(frame)[:, 1]
        threshold = float(self.bundle["threshold"])
        predictions = [
            PredictionRow(
                churn_probability=round(float(probability), 6),
                predicted_churn=bool(probability >= threshold),
                threshold=round(threshold, 6),
            )
            for probability in probabilities
        ]
        return PredictionResponse(
            predictions=predictions,
            model_run_id=self.bundle["run_id"],
        )


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService()


app = FastAPI(title="Churn Prediction Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return get_model_service().metadata()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    return get_model_service().predict(request.rows)
