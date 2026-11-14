"""Tests for the retrain triggers."""
from activelearn.triggers import RetrainPolicy, RetrainState


def test_batch_size_trigger():
    policy = RetrainPolicy(batch_size=10)
    fired, reasons = policy.should_retrain(
        RetrainState(labels_since_retrain=10), round_no=1)
    assert fired
    assert any("batch_size" in r for r in reasons)
    fired, _ = policy.should_retrain(
        RetrainState(labels_since_retrain=9), round_no=2)
    assert not fired


def test_uncertainty_trigger():
    policy = RetrainPolicy(batch_size=100, uncertainty_threshold=0.8)
    fired, reasons = policy.should_retrain(
        RetrainState(labels_since_retrain=5,
                     mean_pool_uncertainty=0.85), round_no=1)
    assert fired
    assert any("uncertainty" in r for r in reasons)
    fired, _ = policy.should_retrain(
        RetrainState(labels_since_retrain=5,
                     mean_pool_uncertainty=0.79), round_no=2)
    assert not fired


def test_drift_trigger():
    policy = RetrainPolicy(batch_size=100, drift_threshold=1.0)
    fired, reasons = policy.should_retrain(
        RetrainState(labels_since_retrain=5, centroid_shift=1.2),
        round_no=1)
    assert fired
    assert any("drift" in r for r in reasons)


def test_staleness_trigger():
    policy = RetrainPolicy(batch_size=100, max_rounds_between_retrains=3)
    fired, reasons = policy.should_retrain(
        RetrainState(rounds_since_retrain=3), round_no=3)
    assert fired
    assert any("staleness" in r for r in reasons)


def test_no_trigger_when_all_quiet():
    policy = RetrainPolicy(batch_size=10, uncertainty_threshold=0.9,
                           drift_threshold=1.0,
                           max_rounds_between_retrains=5)
    fired, reasons = policy.should_retrain(
        RetrainState(labels_since_retrain=3, mean_pool_uncertainty=0.5,
                     centroid_shift=0.2, rounds_since_retrain=1),
        round_no=1)
    assert not fired
    assert reasons == []


def test_history_records_fired_retrains():
    policy = RetrainPolicy(batch_size=5)
    policy.should_retrain(RetrainState(labels_since_retrain=5), round_no=1)
    policy.should_retrain(RetrainState(labels_since_retrain=2), round_no=2)
    policy.should_retrain(RetrainState(labels_since_retrain=6), round_no=3)
    assert policy.summary()["n_retrains"] == 2
    assert policy.summary()["history"][0]["round"] == 1
