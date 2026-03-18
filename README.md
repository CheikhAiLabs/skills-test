# 🚀 Churn Prediction MLOps Platform

An end-to-end churn prediction project with a modern inference UI, a production-style FastAPI service, experiment tracking in MLflow, and workflow orchestration in Airflow.

## ✨ Highlights

- 🧠 churn prediction training pipeline with validation, gating, packaging, and promotion
- 🌐 responsive frontend UI for inference visualization
- ⚡ FastAPI backend with `/healthz`, `/metadata`, and `/predict`
- 📈 MLflow tracking with nested runs, candidate comparison, artifacts, and model registry
- ⏱️ Airflow DAG for retraining, reporting, and scheduled orchestration
- 🐳 Docker Compose stack for local end-to-end execution

## 🏗️ Architecture

```text
Frontend UI  ->  FastAPI Inference API  ->  Latest model bundle
                      |
                      v
              Training / Retraining CLI
                      |
          +-----------+-----------+
          |                       |
          v                       v
      MLflow UI             Airflow DAG
 tracking, registry      orchestration, logs
```

## 🌍 Access URLs

- Frontend UI: `http://localhost:9634`
- Backend API: `http://localhost:9471`
- API health: `http://localhost:9471/healthz`
- API metadata: `http://localhost:9471/metadata`
- MLflow UI: `http://localhost:9861`
- Airflow UI: `http://localhost:9726`

Airflow starts in standalone mode with an `admin` user. To retrieve the current password:

```bash
docker compose logs airflow | rg "Password for user 'admin'"
```

## 🧩 Project Layout

```text
configs/                  Pipeline configuration
data/raw/                 Input datasets
artifacts/                Packaged model outputs
reports/                  Generated evaluation reports
tracking/                 Local run metadata
src/churn_ml/             Core training, inference, and tracking code
frontend/                 UI for prediction and visualization
orchestration/airflow/    Airflow DAG for retraining and reporting
deploy/                   Deployment manifests
tests/                    API and pipeline tests
```

## ⚙️ Local Setup

Create the environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Generate a dataset and run the training pipeline:

```bash
python -m churn_ml.cli bootstrap-data --rows 5000
python -m churn_ml.cli run
python -m churn_ml.cli report
```

Run the API locally:

```bash
MODEL_BUNDLE_PATH=$(ls -dt artifacts/* | head -n 1)/model_bundle.joblib \
uvicorn churn_ml.api:app --host 0.0.0.0 --port 9471
```

Run the UI locally:

```bash
python3 -m http.server 9634 --directory frontend
```

## 🐳 Docker Compose

Start the full platform:

```bash
docker compose up --build
```

Included services:

- `frontend` on `9634`
- `backend` on `9471`
- `mlflow` on `9861`
- `airflow` on `9726`

The backend container preserves the training flow and automatically:

1. creates synthetic data if needed
2. trains a model if no bundle is available
3. serves inference with the latest packaged bundle

## 📈 Training, Retraining, and Tracking

Main commands:

```bash
python -m churn_ml.cli bootstrap-data
python -m churn_ml.cli run
python -m churn_ml.cli retrain --rounds 3
python -m churn_ml.cli report
```

MLflow logs:

- training runs and retraining campaigns
- nested candidate runs
- metrics, params, tags, datasets, signatures, and model artifacts
- registered versions of `churn_prediction_model`

Airflow DAG:

- DAG id: `churn_training_pipeline`
- tasks:
  - `bootstrap_data_if_missing`
  - `retrain_candidates`
  - `report_latest_metrics`

Manual trigger:

```bash
docker compose exec airflow airflow dags trigger churn_training_pipeline
```

## ✅ Validation Gates

The selected model must pass:

- schema and entity-level validation
- missing value thresholds
- minimum ROC AUC
- minimum average precision
- minimum F1
- minimum recall
- baseline outperformance checks

If a round fails a gate, the run is still visible in MLflow and the failure remains traceable.

## 🧪 Quality Checks

Run tests locally:

```bash
source .venv/bin/activate
python -m pytest -q
```

## 🧼 Repository Hygiene

Runtime outputs are intentionally excluded from version control:

- virtual environments and Python caches
- generated datasets
- MLflow local storage
- Airflow logs
- artifacts, reports, and tracking outputs

Only source code, configuration, UI assets, manifests, orchestration code, and tests are committed.
