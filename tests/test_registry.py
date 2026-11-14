"""Tests for the gated model registry."""
import pytest

from activelearn.pipeline import ModelVersion
from activelearn.registry import ModelRegistry


def _version(name, test_accuracy, baseline_accuracy=0.34,
             seed_accuracies=None, labels_to_target=50,
             random_labels_to_target=100):
    metrics = {"test_accuracy": test_accuracy,
               "min_per_class_recall": 0.5,
               "seed_accuracies": seed_accuracies or [test_accuracy],
               "labels_to_target": labels_to_target}
    return ModelVersion(name=name, seed=42, strategy="margin",
                        n_labels=50, model_kind="knn", metrics=metrics,
                        baseline_metrics={
                            "majority_test_accuracy": baseline_accuracy,
                            "random_labels_to_target": random_labels_to_target},
                        model_state={}, scaler_state={})


def test_promotion_passes_when_all_gates_pass():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.85), role="champion")
    registry.register(_version("v2", 0.88), role="challenger")
    result = registry.promote_challenger()
    assert result.passed
    assert registry.champion_name == "v2"
    assert "PROMOTED" in result.summary()


def test_accuracy_floor_blocks_promotion():
    registry = ModelRegistry(accuracy_floor=0.90)
    registry.register(_version("v1", 0.85), role="champion")
    registry.register(_version("v2", 0.88), role="challenger")
    result = registry.promote_challenger()
    assert not result.passed
    assert result.checks["accuracy_floor"] is False
    assert registry.champion_name == "v1"


def test_must_beat_baseline_blocks_promotion():
    registry = ModelRegistry(min_improvement_over_baseline=0.60)
    registry.register(_version("v1", 0.85), role="champion")
    registry.register(_version("v2", 0.85), role="challenger")
    result = registry.promote_challenger()
    assert not result.passed
    assert result.checks["must_beat_baseline"] is False


def test_seed_stability_blocks_promotion():
    registry = ModelRegistry(max_seed_std=0.01)
    registry.register(_version("v1", 0.85), role="champion")
    registry.register(_version("v2", 0.85, seed_accuracies=[0.70, 0.90]),
                      role="challenger")
    result = registry.promote_challenger()
    assert not result.passed
    assert result.checks["seed_stability"] is False


def test_label_efficiency_blocks_when_worse_than_random():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.85), role="champion")
    registry.register(_version("v2", 0.85, labels_to_target=120,
                               random_labels_to_target=100),
                      role="challenger")
    result = registry.promote_challenger()
    assert not result.passed
    assert result.checks["label_efficiency"] is False


def test_label_efficiency_passes_when_random_never_reaches_target():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.85), role="champion")
    registry.register(_version("v2", 0.85, labels_to_target=120,
                               random_labels_to_target=None),
                      role="challenger")
    result = registry.promote_challenger()
    assert result.checks["label_efficiency"] is True
    assert result.passed


def test_rollback_restores_previous_champion():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.85), role="champion")
    registry.register(_version("v2", 0.88), role="challenger")
    registry.promote_challenger()
    assert registry.champion_name == "v2"
    assert registry.rollback() == "v1"
    assert registry.champion_name == "v1"


def test_promote_without_challenger_raises():
    registry = ModelRegistry()
    with pytest.raises(ValueError, match="no challenger"):
        registry.promote_challenger()


def test_rollback_without_history_raises():
    registry = ModelRegistry()
    with pytest.raises(ValueError, match="no previous champion"):
        registry.rollback()


def test_list_versions_reports_roles(version):
    registry = ModelRegistry()
    registry.register(version, role="champion")
    listed = registry.list_versions()
    assert listed[0]["name"] == "v1-test"
    assert listed[0]["role"] == "champion"
