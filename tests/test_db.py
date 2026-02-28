import pytest

pytest.importorskip("sqlalchemy")

from src.db import get_engine, get_latest_monitor_run, init_db, log_monitor_run



def test_monitor_run_roundtrip(tmp_path):
    db_file = tmp_path / "test.db"
    engine = get_engine(f"sqlite:///{db_file}")
    init_db(engine)

    run_id = log_monitor_run(
        engine=engine,
        drift_score=0.4,
        drifted_features=4,
        total_features=10,
        status="drifted",
        report_path="/tmp/report.html",
    )

    latest = get_latest_monitor_run(engine)
    assert latest is not None
    assert latest["run_id"] == run_id
    assert latest["status"] == "drifted"
