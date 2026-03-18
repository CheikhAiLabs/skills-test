from __future__ import annotations

from pathlib import Path

import yaml


def write_test_config(tmp_path: Path) -> Path:
    config = {
        "run_name_prefix": "test-churn",
        "data": {
            "raw_data_path": str(tmp_path / "data" / "raw" / "churn_customers.csv"),
            "artifact_root": str(tmp_path / "artifacts"),
            "tracking_path": str(tmp_path / "tracking" / "runs.jsonl"),
            "report_path": str(tmp_path / "reports" / "latest_evaluation.md"),
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "random_state": 42,
        },
        "validation": {
            "target_column": "churned",
            "entity_id_column": "customer_id",
            "max_missing_fraction": 0.2,
            "min_rows": 500,
            "positive_class_min_rate": 0.05,
            "positive_class_max_rate": 0.7,
        },
        "model": {
            "numeric_features": [
                "age",
                "tenure_months",
                "monthly_charges",
                "total_charges",
                "support_tickets_90d",
                "avg_weekly_logins",
                "late_payments_12m",
            ],
            "categorical_features": [
                "geography",
                "gender",
                "contract_type",
                "payment_method",
            ],
            "boolean_features": [
                "paperless_billing",
                "has_internet_service",
                "has_phone_service",
                "is_premium_plan",
            ],
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 400,
                "class_weight": "balanced",
            },
        },
        "training": {
            "experiment_name": "test-churn-experiment",
            "registered_model_name": "test_churn_model",
            "selection_metric": "roc_auc",
            "retrain_rounds": 2,
            "log_model_candidates": True,
            "candidate_models": [
                {
                    "name": "logistic_default",
                    "estimator": "logistic_regression",
                    "params": {
                        "C": 1.0,
                        "max_iter": 400,
                        "class_weight": "balanced",
                    },
                },
                {
                    "name": "random_forest_default",
                    "estimator": "random_forest",
                    "params": {
                        "n_estimators": 50,
                        "max_depth": 8,
                        "min_samples_leaf": 2,
                        "class_weight": "balanced_subsample",
                        "n_jobs": -1,
                    },
                },
            ],
        },
        "mlflow": {
            "enabled": True,
            "tracking_uri": str(tmp_path / "mlruns"),
        },
        "thresholding": {
            "min_precision": 0.30,
            "min_recall": 0.30,
        },
        "metrics": {
            "min_roc_auc": 0.79,
            "min_average_precision": 0.35,
            "min_f1": 0.38,
            "min_recall": 0.35,
        },
        "packaging": {
            "sample_payload_rows": 3,
        },
    }
    config_path = tmp_path / "configs" / "pipeline.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path
