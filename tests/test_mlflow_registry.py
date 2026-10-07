import numpy as np
import pytest

pytest.importorskip("mlflow")

from sklearn.linear_model import LogisticRegression

from src.mlflow_registry import CHAMPION, PREVIOUS, MlflowChampionRegistry


def _model(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(50, 2))
    return LogisticRegression().fit(x, (x[:, 0] > 0).astype(int))


def test_promote_and_one_step_rollback(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    registry = MlflowChampionRegistry("test-model", uri, "test-exp")

    v1 = registry.register(_model(0), "v1", {"p": 1}, {"m": 0.1})
    registry.promote(v1)
    v2 = registry.register(_model(1), "v2", {"p": 2}, {"m": 0.2})
    registry.promote(v2)

    assert registry.alias_version(CHAMPION) == v2
    assert registry.alias_version(PREVIOUS) == v1

    assert registry.rollback() == v1
    assert registry.alias_version(CHAMPION) == v1
    assert registry.alias_version(PREVIOUS) == v2
    assert registry.load_champion().predict(np.zeros((1, 2))).shape == (1,)


def test_rollback_without_previous_raises(tmp_path):
    registry = MlflowChampionRegistry("solo", f"sqlite:///{tmp_path / 'mlflow.db'}", "test-exp")
    registry.promote(registry.register(_model(0), "v1", {}, {}))

    with pytest.raises(RuntimeError):
        registry.rollback()
