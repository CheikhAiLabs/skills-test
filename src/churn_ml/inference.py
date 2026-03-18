from __future__ import annotations

from typing import Any

import pandas as pd

from churn_ml.data import _parse_boolean


def prepare_inference_frame(
    records: list[dict[str, Any]],
    *,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    boolean_features: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required inference columns: {missing}")

    prepared = frame[feature_columns].copy()
    for column in numeric_features:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in categorical_features:
        prepared[column] = prepared[column].astype("string").replace({"<NA>": pd.NA})
    for column in boolean_features:
        prepared[column] = prepared[column].map(_parse_boolean)
    return prepared
