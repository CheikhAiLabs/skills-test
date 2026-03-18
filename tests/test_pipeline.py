from __future__ import annotations

import json

from churn_ml.config import load_config
from churn_ml.data import write_synthetic_dataset
from churn_ml.pipeline import PipelineRunner

from tests.conftest import write_test_config


def test_pipeline_run_produces_bundle_and_reports(tmp_path):
    config_path = write_test_config(tmp_path)
    config = load_config(config_path)
    write_synthetic_dataset(config.data.raw_data_path, rows=2500, seed=7)

    run_dir = PipelineRunner(config).run()

    assert (run_dir / "model_bundle.joblib").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "model_card.md").exists()
    assert (run_dir / "sample_payload.json").exists()
    assert config.data.report_path.exists()
    assert config.data.tracking_path.exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"]["roc_auc"] >= config.metrics.min_roc_auc
