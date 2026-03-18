from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_ml.config import ModelConfig, PipelineConfig
from churn_ml.exceptions import DataValidationError


BOOLEAN_TRUE_VALUES = {"1", "true", "yes", "y", "t"}
BOOLEAN_FALSE_VALUES = {"0", "false", "no", "n", "f"}


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def generate_synthetic_churn_data(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    geographies = np.array(["France", "Germany", "Spain", "United Kingdom"])
    genders = np.array(["Female", "Male", "Non-binary"])
    contract_types = np.array(["Month-to-month", "One year", "Two year"])
    payment_methods = np.array(
        ["Credit card", "Bank transfer", "Direct debit", "Digital wallet"]
    )

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{100000 + idx}" for idx in range(rows)],
            "geography": rng.choice(geographies, size=rows, p=[0.28, 0.24, 0.18, 0.30]),
            "gender": rng.choice(genders, size=rows, p=[0.48, 0.48, 0.04]),
            "age": rng.integers(18, 81, size=rows),
            "tenure_months": rng.integers(1, 73, size=rows),
            "contract_type": rng.choice(
                contract_types, size=rows, p=[0.54, 0.24, 0.22]
            ),
            "payment_method": rng.choice(payment_methods, size=rows),
            "monthly_charges": rng.normal(72, 24, size=rows).clip(18, 180).round(2),
            "support_tickets_90d": rng.poisson(1.8, size=rows),
            "avg_weekly_logins": rng.normal(4.4, 2.1, size=rows).clip(0.1, 20).round(2),
            "paperless_billing": rng.choice([True, False], size=rows, p=[0.71, 0.29]),
            "has_internet_service": rng.choice([True, False], size=rows, p=[0.88, 0.12]),
            "has_phone_service": rng.choice([True, False], size=rows, p=[0.84, 0.16]),
            "late_payments_12m": rng.poisson(1.3, size=rows),
            "is_premium_plan": rng.choice([True, False], size=rows, p=[0.34, 0.66]),
        }
    )

    df["total_charges"] = (
        df["monthly_charges"] * df["tenure_months"] * rng.uniform(0.92, 1.08, size=rows)
    ).round(2)

    contract_risk = df["contract_type"].map(
        {"Month-to-month": 1.25, "One year": -0.05, "Two year": -0.8}
    )
    payment_risk = df["payment_method"].map(
        {
            "Credit card": -0.08,
            "Bank transfer": 0.02,
            "Direct debit": 0.28,
            "Digital wallet": 0.12,
        }
    )
    geography_risk = df["geography"].map(
        {"France": -0.08, "Germany": 0.04, "Spain": 0.18, "United Kingdom": 0.02}
    )
    premium_risk = np.where(df["is_premium_plan"], -0.3, 0.2)
    internet_risk = np.where(df["has_internet_service"], 0.09, -0.12)
    phone_risk = np.where(df["has_phone_service"], 0.05, -0.04)
    paperless_risk = np.where(df["paperless_billing"], 0.2, -0.08)

    logit = (
        -1.95
        + 0.024 * (df["age"] - 40)
        - 0.052 * df["tenure_months"]
        + 0.034 * (df["monthly_charges"] - 70)
        + 0.22 * df["support_tickets_90d"]
        - 0.28 * df["avg_weekly_logins"]
        + 0.24 * df["late_payments_12m"]
        + contract_risk
        + payment_risk
        + geography_risk
        + premium_risk
        + internet_risk
        + phone_risk
        + paperless_risk
        + rng.normal(0, 0.2, size=rows)
    )
    probabilities = 1 / (1 + np.exp(-logit))
    df["churned"] = rng.binomial(1, probabilities, size=rows)
    return df


def write_synthetic_dataset(path: Path, rows: int, seed: int) -> Path:
    frame = generate_synthetic_churn_data(rows=rows, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data file does not exist: {path}")
    return pd.read_csv(path)


def _parse_boolean(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return float(value)
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return value
    text = str(value).strip().lower()
    if text in BOOLEAN_TRUE_VALUES:
        return 1.0
    if text in BOOLEAN_FALSE_VALUES:
        return 0.0
    return np.nan


def coerce_schema(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    frame = df.copy()
    required_columns = [
        config.validation.entity_id_column,
        *config.model.feature_columns,
        config.validation.target_column,
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    for column in config.model.numeric_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in config.model.boolean_features:
        frame[column] = frame[column].map(_parse_boolean)

    for column in config.model.categorical_features:
        frame[column] = frame[column].astype("string")
        frame[column] = frame[column].replace({"<NA>": pd.NA})

    frame[config.validation.entity_id_column] = frame[
        config.validation.entity_id_column
    ].astype("string")
    frame[config.validation.target_column] = pd.to_numeric(
        frame[config.validation.target_column], errors="coerce"
    )
    return frame


def validate_dataframe(df: pd.DataFrame, config: PipelineConfig) -> None:
    entity_id_column = config.validation.entity_id_column
    target_column = config.validation.target_column

    if len(df) < config.validation.min_rows:
        raise DataValidationError(
            f"Dataset too small: got {len(df)} rows, expected at least "
            f"{config.validation.min_rows}"
        )

    if df[entity_id_column].isna().any():
        raise DataValidationError("Entity identifiers contain null values.")

    if df[entity_id_column].duplicated().any():
        duplicate_count = int(df[entity_id_column].duplicated().sum())
        raise DataValidationError(
            f"Duplicate entity identifiers detected: {duplicate_count}"
        )

    missing_fraction = df.isna().mean().sort_values(ascending=False)
    violating_columns = missing_fraction[
        missing_fraction > config.validation.max_missing_fraction
    ]
    if not violating_columns.empty:
        raise DataValidationError(
            "Columns exceed missing-value threshold: "
            f"{violating_columns.to_dict()}"
        )

    target_values = set(df[target_column].dropna().astype(int).unique().tolist())
    if not target_values.issubset({0, 1}) or len(target_values) < 2:
        raise DataValidationError("Target column must be binary with both classes present.")

    positive_rate = float(df[target_column].mean())
    if positive_rate < config.validation.positive_class_min_rate:
        raise DataValidationError(
            f"Positive class rate {positive_rate:.3f} is below minimum "
            f"{config.validation.positive_class_min_rate:.3f}"
        )
    if positive_rate > config.validation.positive_class_max_rate:
        raise DataValidationError(
            f"Positive class rate {positive_rate:.3f} is above maximum "
            f"{config.validation.positive_class_max_rate:.3f}"
        )


def dataset_summary(df: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    target_column = config.validation.target_column
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "positive_rate": round(float(df[target_column].mean()), 4),
        "missing_fraction": {
            key: round(float(value), 4) for key, value in df.isna().mean().to_dict().items()
        },
    }


def split_dataset(
    df: pd.DataFrame,
    config: PipelineConfig,
    random_state: int | None = None,
) -> DatasetSplits:
    target_column = config.validation.target_column
    feature_columns = [config.validation.entity_id_column, *config.model.feature_columns, target_column]
    effective_random_state = (
        config.data.random_state if random_state is None else random_state
    )

    test_size = config.data.test_fraction
    validation_share = config.data.validation_fraction / (
        config.data.train_fraction + config.data.validation_fraction
    )

    train_val, test = train_test_split(
        df[feature_columns],
        test_size=test_size,
        random_state=effective_random_state,
        stratify=df[target_column],
    )
    train, validation = train_test_split(
        train_val,
        test_size=validation_share,
        random_state=effective_random_state,
        stratify=train_val[target_column],
    )
    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
