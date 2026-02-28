from src.policy import should_retrain



def test_should_retrain_true_on_threshold_boundary():
    assert should_retrain(drifted_features=3, total_features=10, threshold=0.3)



def test_should_retrain_false_below_threshold():
    assert not should_retrain(drifted_features=2, total_features=10, threshold=0.3)
