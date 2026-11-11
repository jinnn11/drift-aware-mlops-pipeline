import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from src.retrain_data import build_retrain_data, split_recent_labeled_data


def _frame(n: int, start: int) -> pd.DataFrame:
    # `row_id` marks each row so the tests can check which rows end up where.
    return pd.DataFrame({"row_id": np.arange(start, start + n), "y": ["yes", "no"] * (n // 2)})


def test_recent_split_is_time_ordered_and_disjoint():
    df = _frame(2000, 0)
    train, val, evaluation = split_recent_labeled_data(df, holdout_ratio=0.2, val_ratio=0.1)

    assert len(evaluation) == 400 and len(val) == 200 and len(train) == 1400
    assert train["row_id"].max() < val["row_id"].min()
    assert val["row_id"].max() < evaluation["row_id"].min()


def test_recent_split_rejects_too_little_data():
    with pytest.raises(ValueError):
        split_recent_labeled_data(_frame(400, 0), holdout_ratio=0.2, val_ratio=0.1)


def test_candidate_never_trains_on_rows_it_is_tuned_or_scored_on():
    historical = _frame(1000, 0)
    recent = _frame(2000, 10_000)
    data = build_retrain_data(historical, recent, holdout_ratio=0.2, val_ratio=0.1, recent_weight=4.0)

    trained = set(data.train_df["row_id"])
    held_out = (
        set(data.historical_val["row_id"])
        | set(data.historical_test["row_id"])
        | set(data.recent_val["row_id"])
        | set(data.recent_eval["row_id"])
    )
    assert trained.isdisjoint(held_out)
    # The candidate trains on the same historical rows as the baseline: the first 70%.
    assert data.historical_train_rows == 700
    assert data.train_df["row_id"].iloc[:700].tolist() == list(range(700))
    # Only the new rows are upweighted.
    assert (data.sample_weights[:700] == 1.0).all()
    assert (data.sample_weights[700:] == 4.0).all()
    assert data.new_labeled_rows == 2000


def test_without_new_data_the_candidate_trains_on_the_baseline_rows_only():
    data = build_retrain_data(_frame(1000, 0), None)

    assert data.training_mode == "historical_only"
    assert len(data.train_df) == 700
    assert set(data.train_df["row_id"]).isdisjoint(set(data.historical_test["row_id"]))
    assert data.recent_eval is None and data.recent_val is None
