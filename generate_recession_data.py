from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset, prepare_features_and_target, validate_dataset
from src.evaluation import evaluate_probabilistic_classifier
from src.registry import load_promoted_model


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def create_recession_view(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()

    # Covariate shift: macroeconomic deterioration.
    out["emp.var.rate"] = out["emp.var.rate"] - rng.uniform(2.0, 3.5, size=len(out))
    out["cons.price.idx"] = out["cons.price.idx"] * rng.uniform(1.03, 1.08, size=len(out))
    out["euribor3m"] = out["euribor3m"] * rng.uniform(0.35, 0.65, size=len(out))
    out["cons.conf.idx"] = out["cons.conf.idx"] - rng.uniform(2.0, 7.0, size=len(out))
    out["campaign"] = np.clip(out["campaign"] + rng.integers(0, 3, size=len(out)), 1, None)
    out["previous"] = np.clip(out["previous"] + rng.integers(0, 2, size=len(out)), 0, None)

    # Concept shift: label behavior changes under recession.
    score = (
        1.4 * (out["euribor3m"] < 2.0).astype(float)
        + 1.0 * (out["emp.var.rate"] < -1.0).astype(float)
        + 0.8 * (out["cons.conf.idx"] < -35.0).astype(float)
        + 0.5 * (out["previous"] > 0).astype(float)
        + 0.7 * out["job"].isin(["retired", "student", "unemployed"]).astype(float)
        - 0.7 * (out["housing"] == "yes").astype(float)
        - 0.6 * (out["loan"] == "yes").astype(float)
        + 0.4 * out["month"].isin(["mar", "apr", "may", "jun"]).astype(float)
        + rng.normal(0.0, 0.7, size=len(out))
    )

    probs = _sigmoid(score - 1.4)
    y_new = rng.binomial(1, probs)
    out["y"] = np.where(y_new == 1, "yes", "no")
    return out


def generate_recession_datasets(
    source_path: str,
    output_dir: str,
    recent_ratio: float = 0.35,
    holdout_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    raw_df = load_dataset(source_path)
    validate_dataset(raw_df)

    recent_rows = max(3000, int(len(raw_df) * recent_ratio))
    recent_base = raw_df.tail(recent_rows).copy().reset_index(drop=True)
    recession_full = create_recession_view(recent_base, seed=seed)

    eval_rows = max(500, int(len(recession_full) * holdout_ratio))
    recession_train = recession_full.iloc[:-eval_rows].copy()
    recession_test = recession_full.iloc[-eval_rows:].copy()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "recession_incoming_full.csv"
    train_path = out_dir / "recession_incoming_train.csv"
    test_path = out_dir / "recession_unseen_test.csv"

    recession_full.to_csv(full_path, sep=";", index=False)
    recession_train.to_csv(train_path, sep=";", index=False)
    recession_test.to_csv(test_path, sep=";", index=False)

    result = {
        "source_rows": len(raw_df),
        "recent_slice_rows": len(recent_base),
        "recession_full_rows": len(recession_full),
        "recession_train_rows": len(recession_train),
        "recession_unseen_test_rows": len(recession_test),
        "train_positive_rate": float((recession_train["y"] == "yes").mean()),
        "test_positive_rate": float((recession_test["y"] == "yes").mean()),
        "paths": {
            "full": str(full_path),
            "train": str(train_path),
            "unseen_test": str(test_path),
        },
    }

    # Optional sanity check: current model quality on unseen recession distribution.
    try:
        model, metadata = load_promoted_model("models/model.pkl", "models/model_metadata.json")
        x_test, y_test = prepare_features_and_target(recession_test)
        feature_columns = metadata["feature_columns"]
        result["current_model_on_recession_unseen"] = evaluate_probabilistic_classifier(
            model,
            x_test[feature_columns],
            y_test,
            threshold=float(metadata.get("optimal_threshold", 0.5)),
        )
        result["current_model_version"] = metadata.get("model_version", "unknown")
    except FileNotFoundError:
        result["current_model_on_recession_unseen"] = None

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labeled recession-era incoming data for realistic retraining.")
    parser.add_argument("--source-path", default="data/raw/bank-additional-full.csv")
    parser.add_argument("--output-dir", default="data/processed/recession")
    parser.add_argument("--recent-ratio", type=float, default=0.35)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = generate_recession_datasets(
        source_path=args.source_path,
        output_dir=args.output_dir,
        recent_ratio=args.recent_ratio,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
