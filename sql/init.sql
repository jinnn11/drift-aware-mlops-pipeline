CREATE TABLE IF NOT EXISTS prediction_logs (
  id UUID PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  request_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  features JSONB NOT NULL,
  prediction INT NOT NULL,
  score DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS monitor_runs (
  run_id UUID PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  drift_score DOUBLE PRECISION NOT NULL,
  drifted_features INT NOT NULL,
  total_features INT NOT NULL,
  status TEXT NOT NULL,
  report_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_runs (
  run_id UUID PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  mlflow_run_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  cv_roc_auc DOUBLE PRECISION NOT NULL,
  val_roc_auc DOUBLE PRECISION NOT NULL,
  test_roc_auc DOUBLE PRECISION NOT NULL
);
