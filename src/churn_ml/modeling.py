from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_ml.config import PipelineConfig


def build_preprocessor(config: PipelineConfig) -> ColumnTransformer:
    numeric_features = [
        *config.model.numeric_features,
        *config.model.boolean_features,
    ]
    categorical_features = config.model.categorical_features

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_classifier(estimator_name: str, estimator_params: dict[str, Any]) -> Any:
    if estimator_name == "logistic_regression":
        return LogisticRegression(**estimator_params)
    if estimator_name == "random_forest":
        return RandomForestClassifier(**estimator_params)
    if estimator_name == "extra_trees":
        return ExtraTreesClassifier(**estimator_params)
    raise ValueError(f"Unsupported estimator: {estimator_name}")


def build_model_pipeline(
    config: PipelineConfig,
    estimator_name: str = "logistic_regression",
    estimator_params: dict[str, Any] | None = None,
) -> Pipeline:
    preprocessor = build_preprocessor(config)
    classifier = build_classifier(
        estimator_name,
        estimator_params or config.model.logistic_regression,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def extract_feature_importances(model_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        raw_values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        raw_values = classifier.coef_[0]
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": raw_values,
        }
    )
    frame["abs_importance"] = frame["importance"].abs()
    return frame.sort_values("abs_importance", ascending=False).reset_index(drop=True)
