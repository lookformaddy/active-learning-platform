"""Tests for the full active-learning pipeline run."""
import pytest

from activelearn.pipeline import ActiveLearningRun, RunConfig


def _small_run(dataset, **overrides):
    kwargs = {"strategy": "margin", "model_kind": "knn", "rounds": 3,
              "batch_size": 5, "total_budget": 15,
              "seed_labels_per_class": 2, "target_accuracy": 0.60,
              "stability_seeds": 2}
    kwargs.update(overrides)
    config = RunConfig(**kwargs)
    run = ActiveLearningRun(dataset, config)
    return run, run.run()


def test_full_run_respects_budget(moons):
    run, version = _small_run(moons)
    assert version.n_labels == 6 + 15  # 2/class seed + 3 rounds x 5
    assert run.budget.spent == 15
    assert run.budget.exhausted
    assert version.metrics["test_accuracy"] >= 0.5
    assert version.baseline_metrics["majority_test_accuracy"] > 0.0


def test_full_run_records_rounds_and_retrains(moons):
    run, version = _small_run(moons)
    assert len(run.round_log) == 3
    assert run.retrain_count >= 1
    assert all(entry["retrain"] for entry in run.round_log)
    assert version.metrics["seed_accuracies"]
    assert version.metrics["seed_accuracy_std"] >= 0.0


def test_full_run_queue_agreement(moons):
    run, _version = _small_run(moons)
    summary = run.queue.summary()
    assert summary["n_items"] == 15
    assert summary["agreement"]["n_double_labelled"] == 3
    assert summary["n_disagreements"] >= 0
    assert summary["n_adjudicated"] == summary["n_disagreements"]


def test_run_with_logistic_model(blobs):
    run, version = _small_run(blobs, model_kind="logistic", rounds=2,
                              batch_size=5, total_budget=10, epochs=60)
    assert version.n_labels == 6 + 10
    assert version.model_kind == "logistic"
    assert version.model_state["kind"] == "logistic_regression"
    assert version.scaler_state["mean"]


def test_run_stops_when_budget_exhausted(moons):
    run, version = _small_run(moons, rounds=10, total_budget=10)
    assert run.budget.spent == 10
    assert version.n_labels == 6 + 10
    assert len(run.round_log) == 2  # only 2 rounds fit the budget


def test_run_with_diversity_strategy(blobs):
    _run, version = _small_run(blobs, strategy="diversity", rounds=2,
                               batch_size=5, total_budget=10)
    assert version.strategy == "diversity"
    assert version.n_labels == 16


def test_unknown_model_kind_rejected(moons):
    with pytest.raises(ValueError, match="unknown model_kind"):
        _small_run(moons, model_kind="transformer")
