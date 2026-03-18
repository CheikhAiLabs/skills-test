FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY data /app/data
COPY artifacts /app/artifacts
COPY reports /app/reports
COPY tracking /app/tracking
COPY orchestration /app/orchestration
COPY deploy /app/deploy

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

ENV PYTHONPATH=/app/src
EXPOSE 9471

CMD ["uvicorn", "churn_ml.api:app", "--host", "0.0.0.0", "--port", "9471"]
