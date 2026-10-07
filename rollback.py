"""Switch the serving model back to the previous champion in one step.

    python rollback.py --model-name bank_marketing_xgboost --restore-serving-files
    curl -X POST http://localhost:8000/reload
"""
from __future__ import annotations

import argparse
import json

from src.config import get_settings
from src.mlflow_registry import CHAMPION, PREVIOUS, MlflowChampionRegistry
from src.registry import save_promoted_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tracking-uri", default=get_settings().mlflow_tracking_uri)
    parser.add_argument(
        "--restore-serving-files",
        action="store_true",
        help="also write the restored champion to MODEL_PATH/MODEL_METADATA_PATH (then POST /reload on the API)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    registry = MlflowChampionRegistry(args.model_name, args.tracking_uri)
    before = {CHAMPION: registry.alias_version(CHAMPION), PREVIOUS: registry.alias_version(PREVIOUS)}
    registry.rollback()
    after = {CHAMPION: registry.alias_version(CHAMPION), PREVIOUS: registry.alias_version(PREVIOUS)}
    if args.restore_serving_files:
        settings = get_settings()
        save_promoted_model(registry.load_champion(), settings.model_path, settings.model_metadata_path,
                            registry.champion_metadata())
    print(json.dumps({"model": args.model_name, "before": before, "after": after}, indent=2))
