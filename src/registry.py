from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd



def save_promoted_model(model, model_path: str, metadata_path: str, metadata: Dict[str, Any]) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)



def load_promoted_model(model_path: str, metadata_path: str) -> Tuple[Any, Dict[str, Any]]:
    model = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    return model, metadata



def save_reference_data(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def load_reference_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
