import pytest

pd = pytest.importorskip("pandas")

from src.data import EXPECTED_COLUMNS, prepare_features_and_target, time_ordered_split



def _sample_df(n: int = 20) -> pd.DataFrame:
    base_row = {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 120,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.9,
        "cons.conf.idx": -40.0,
        "euribor3m": 4.8,
        "nr.employed": 5191.0,
        "y": "no",
    }
    rows = []
    for i in range(n):
        row = dict(base_row)
        row["age"] = 20 + i
        row["y"] = "yes" if i % 2 == 0 else "no"
        rows.append(row)
    df = pd.DataFrame(rows)
    assert set(df.columns) == EXPECTED_COLUMNS
    return df



def test_prepare_features_and_target_drops_leakage_and_maps_target():
    df = _sample_df(10)
    x, y = prepare_features_and_target(df)

    assert "duration" not in x.columns
    assert set(y.unique()) == {0, 1}



def test_time_ordered_split_shapes():
    df = _sample_df(20)
    x, y = prepare_features_and_target(df)
    split = time_ordered_split(x, y)

    assert len(split.x_train) == 14
    assert len(split.x_val) == 3
    assert len(split.x_test) == 3
