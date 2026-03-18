# Churn Prediction ML Pipeline

Production-ready churn prediction stack with:

- data generation, validation, splitting, training, gating, packaging
- FastAPI inference API
- frontend inference UI
- MLflow experiment tracking and model registry
- Airflow orchestration
- additive Kubeflow pipeline definition

## Stack Access

- Frontend UI: `http://localhost:9634`
- Backend API: `http://localhost:9471`
- API health: `http://localhost:9471/healthz`
- API metadata: `http://localhost:9471/metadata`
- MLflow UI: `http://localhost:9861`
- Airflow UI: `http://localhost:9726`

Airflow runs in standalone mode and creates an `admin` user at startup. Retrieve the current password with:

```bash
docker compose logs airflow | rg "Password for user 'admin'"
```

## Project Structure

```text
configs/                  Pipeline configuration
data/raw/                 Input datasets
artifacts/                Model bundles and packaged outputs
reports/                  Latest markdown/json reports
tracking/                 Local run tracking metadata
src/churn_ml/             Training, evaluation, tracking, serving
frontend/                 Inference visualization UI
orchestration/airflow/    Airflow DAG
orchestration/kubeflow/   Kubeflow pipeline definition
deploy/                   Deployment manifests
tests/                    API and pipeline tests
```

## Local Development

Create the environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Generate data, train, and inspect metrics:

```bash
python -m churn_ml.cli bootstrap-data --rows 5000
python -m churn_ml.cli run
python -m churn_ml.cli report
```

Serve the API locally:

```bash
MODEL_BUNDLE_PATH=$(ls -dt artifacts/* | head -n 1)/model_bundle.joblib \
uvicorn churn_ml.api:app --host 0.0.0.0 --port 9471
```

Launch the frontend locally:

```bash
python3 -m http.server 9634 --directory frontend
```

## Docker Compose

Start the full stack:

```bash
docker compose up --build
```

Services:

- `backend`: FastAPI inference API on `9471`
- `frontend`: static UI on `9634`
- `mlflow`: MLflow `3.10.1` on `9861`
- `airflow`: Airflow `3.1.8` on `9726`

The backend container preserves the training architecture. On startup it:

1. bootstraps synthetic data if `data/raw/churn_customers.csv` is missing
2. trains and packages a model if no bundle exists in `artifacts/`
3. serves inference with the latest bundle

## Training and Tracking

Main CLI commands:

- `python -m churn_ml.cli bootstrap-data`
- `python -m churn_ml.cli run`
- `python -m churn_ml.cli retrain --rounds 3`
- `python -m churn_ml.cli report`

MLflow captures:

- parent campaign runs
- nested training runs
- nested candidate runs
- parameters, metrics, tags, artifacts, model signatures, datasets
- registered versions of `churn_prediction_model`

Airflow DAG:

- DAG id: `churn_training_pipeline`
- tasks:
  - `bootstrap_data_if_missing`
  - `retrain_candidates`
  - `report_latest_metrics`

Trigger manually:

```bash
docker compose exec airflow airflow dags trigger churn_training_pipeline
```

## Validation Gates

Promotion is blocked if the selected model fails any configured gate:

- schema or entity-level validation
- missing-value thresholds
- minimum ROC AUC
- minimum average precision
- minimum F1
- minimum recall
- baseline comparison

## Repository Hygiene

Generated runtime outputs are intentionally ignored:

- virtualenv and caches
- MLflow local storage
- Airflow logs
- generated datasets
- artifacts, reports, and tracking outputs

Only the code, configs, manifests, UI, orchestration definitions, and tests are meant to be versioned.
