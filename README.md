# DRIFT-AWARE MLOps PIPELINE

Production-style MLOps project for binary classification on UCI Bank Marketing, with online serving, drift monitoring, and conditional retraining.

## Impact

- Built a full ML lifecycle system: training, registry, serving, monitoring, retraining, and promotion gates.
- Implemented drift-triggered retraining with **weighted recent data** to adapt to distribution shifts.
- Improved performance on unseen shifted-distribution data:
  - ROC-AUC: **0.5443 -> 0.6621**
  - Delta: **+0.1178**
- Protected production with acceptance criteria so retraining does not blindly replace the model.

## What This System Does

1. Trains and tracks an XGBoost classifier in MLflow.
2. Serves predictions through FastAPI (`/predict`, `/health`, `/metrics`).
3. Monitors data drift with Evidently and stores run summaries.
4. Triggers retraining when drift crosses threshold.
5. Retrains on historical + incoming labeled data, upweighting recent samples.
6. Promotes candidate model only if quality gates pass.

## Architecture

### Core Components
- **Modeling**: `xgboost`, `scikit-learn`
- **Tracking**: `MLflow`
- **Serving**: `FastAPI`
- **Orchestration**: `Apache Airflow`
- **Drift Monitoring**: `Evidently AI`
- **Observability**: `Prometheus`, `Grafana`
- **Storage**: `PostgreSQL`
- **Runtime**: `Docker Compose`
- **Deployment-ready manifests**: `k8s/*.yaml`

### Key Workflows
- **Training**: `train.py` -> logs runs/metrics -> promotes baseline model
- **Monitoring**: `monitor.py` -> drift report HTML + JSON summary + DB entry
- **Retraining**: `retrain.py` -> weighted training + unseen-shift evaluation + gated promotion
- **Automation**: `airflow/dags/daily_monitor_retrain_dag.py`

## Repository Layout

```text
.
├── app.py
├── train.py
├── monitor.py
├── retrain.py
├── generate_recession_data.py
├── simulate_traffic.py
├── docker-compose.yml
├── src/
├── airflow/dags/
├── grafana/
├── prometheus/
├── k8s/
├── tests/
└── docs/
    ├── screenshots/
    └── videos/
```

## Reproducible Run (Local)

### Prerequisites
- Docker + Docker Compose
- Dataset at `data/raw/bank-additional-full.csv`

### Start Services

```bash
cp .env.example .env
docker compose down -v
docker compose up -d --build
```

### Baseline Training

```bash
docker compose exec -T trainer python /app/train.py
```

### Generate Incoming Shifted Data

```bash
docker compose exec -T trainer python /app/generate_recession_data.py \
  --source-path /app/data/raw/bank-additional-full.csv \
  --output-dir /app/data/processed/recession \
  --recent-ratio 0.35 \
  --holdout-ratio 0.2
```

### Simulate Inference Traffic and Drift

```bash
docker compose exec -T trainer python /app/simulate_traffic.py \
  --api-url http://api:8000/predict \
  --phase-seconds 60 \
  --requests-per-second 1 \
  --run-monitor-after-phase
```

### Weighted Retraining + Promotion Decision

```bash
docker compose exec -T trainer python /app/retrain.py \
  --dataset-path /app/data/raw/bank-additional-full.csv \
  --new-data-path /app/data/processed/recession/recession_incoming_full.csv \
  --new-data-holdout-ratio 0.2 \
  --new-data-weight 4.0
```

## Endpoints

- Airflow: `http://localhost:8080`
- Grafana: `http://localhost:3000`
- MLflow: `http://localhost:5001`
- Prometheus: `http://localhost:9090`
- FastAPI: `http://localhost:8000`
- FastAPI OpenAPI: `http://localhost:8000/docs`

## Quality Gates (Promotion Policy)

Retrained candidate is accepted only when:
- It improves unseen shifted-distribution ROC-AUC by at least `0.005`, and
- It does not degrade historical ROC-AUC by more than `0.03`.

## Validation

```bash
docker compose exec -T trainer pytest -q
```

## Run Screenshots

- [Docker services up](docs/screenshots/01-docker-services-up.png)
- [Training + health check](docs/screenshots/02-train-and-health-check.png)
- [Shifted data generation](docs/screenshots/03-recession-data-generated.png)
- [Traffic + drift simulation](docs/screenshots/04-simulate-traffic-drift-output.png)
- [Retraining decision](docs/screenshots/05-retrain-decision-output.png)
- [MLflow runs](docs/screenshots/06-mlflow-experiment-runs.png)
- [Airflow DAG success](docs/screenshots/07-airflow-daily-dag-graph-success.png)
- [Evidently drift details](docs/screenshots/08-evidently-drift-report-details.png)

