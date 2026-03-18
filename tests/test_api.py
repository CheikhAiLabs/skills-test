from __future__ import annotations

import json

from fastapi.testclient import TestClient

from churn_ml.config import load_config
from churn_ml.data import write_synthetic_dataset
from churn_ml.pipeline import PipelineRunner

from tests.conftest import write_test_config


def test_prediction_api_returns_probabilities(tmp_path, monkeypatch):
    config_path = write_test_config(tmp_path)
    config = load_config(config_path)
    write_synthetic_dataset(config.data.raw_data_path, rows=2500, seed=9)
    run_dir = PipelineRunner(config).run()

    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(run_dir / "model_bundle.joblib"))

    from churn_ml.api import app, get_model_service

    get_model_service.cache_clear()
    sample_payload = json.loads((run_dir / "sample_payload.json").read_text(encoding="utf-8"))

    client = TestClient(app)
    response = client.post("/predict", json={"rows": sample_payload})

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == len(sample_payload)
    assert all(0.0 <= row["churn_probability"] <= 1.0 for row in body["predictions"])


def test_metadata_endpoint_exposes_ui_schema(tmp_path, monkeypatch):
    config_path = write_test_config(tmp_path)
    config = load_config(config_path)
    write_synthetic_dataset(config.data.raw_data_path, rows=2500, seed=12)
    run_dir = PipelineRunner(config).run()

    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(run_dir / "model_bundle.joblib"))
    monkeypatch.setenv("PIPELINE_CONFIG_PATH", str(config_path))

    from churn_ml.api import app, get_model_service

    get_model_service.cache_clear()
    client = TestClient(app)
    response = client.get("/metadata")

    assert response.status_code == 200
    body = response.json()
    assert "feature_schema" in body
    assert body["feature_schema"]["numeric"]
    assert body["feature_schema"]["categorical"]
    assert body["feature_schema"]["boolean"]
    assert body["example_row"]
