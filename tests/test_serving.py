"""Tests for latency-budgeted serving with fallback."""
import time

import pytest

from activelearn.pipeline import ModelVersion
from activelearn.registry import ModelRegistry
from activelearn.serving import ServingConfig, ServingLayer


def _layer_with_slow_model(blobs, delay_s=0.0, budget_ms=100.0):
    """Serving layer whose model answers after ``delay_s`` seconds."""
    registry = ModelRegistry()
    version = ModelVersion(
        name="v1", seed=42, strategy="margin", n_labels=9,
        model_kind="logistic",
        metrics={"test_accuracy": 0.85,
                 "train_class_counts": {"0": 3, "1": 3, "2": 3}},
        baseline_metrics={"majority_test_accuracy": 0.34},
        model_state={}, scaler_state={})
    registry.register(version, role="champion")

    def slow_predict(_version, _features):
        if delay_s:
            time.sleep(delay_s)
        return 1, 0.9

    return ServingLayer(registry, ServingConfig(latency_budget_ms=budget_ms),
                        predict_fn=slow_predict)


def test_champion_predicts_within_budget(blobs, serving):
    pred = serving.predict(blobs.X_test[0], request_id="t-1")
    assert pred.arm == "champion"
    assert pred.value in (0, 1, 2)
    assert 0.0 < pred.confidence <= 1.0
    assert pred.latency_ms >= 0.0
    assert serving.stats_dict()["n_requests"] == 1
    assert serving.stats_dict()["n_fallback"] == 0


def test_tiny_budget_forces_fallback(blobs):
    layer = _layer_with_slow_model(blobs, budget_ms=0.0001)
    pred = layer.predict(blobs.X_test[0], request_id="t-2")
    assert pred.arm == "fallback"
    assert layer.stats_dict()["n_fallback"] == 1
    assert layer.stats_dict()["fallback_rate"] == 1.0
    assert "latency_budget_exceeded" in layer.fallback_reasons


def test_model_error_falls_back_when_enabled(blobs):
    registry = ModelRegistry()
    version = ModelVersion(
        name="v1", seed=42, strategy="margin", n_labels=9,
        model_kind="logistic",
        metrics={"test_accuracy": 0.85,
                 "train_class_counts": {"0": 5, "1": 3, "2": 1}},
        baseline_metrics={"majority_test_accuracy": 0.34},
        model_state={}, scaler_state={})
    registry.register(version, role="champion")

    def boom(_version, _features):
        raise RuntimeError("model exploded")

    layer = ServingLayer(registry, ServingConfig(), predict_fn=boom)
    pred = layer.predict(blobs.X_test[0], request_id="t-3")
    assert pred.arm == "fallback"
    assert pred.value == 0  # majority class from the stored counts


def test_model_error_raises_when_fallback_disabled(blobs):
    registry = ModelRegistry()
    version = ModelVersion(
        name="v1", seed=42, strategy="margin", n_labels=9,
        model_kind="logistic", metrics={"test_accuracy": 0.85},
        baseline_metrics={"majority_test_accuracy": 0.34},
        model_state={}, scaler_state={})
    registry.register(version, role="champion")

    def boom(_version, _features):
        raise RuntimeError("model exploded")

    layer = ServingLayer(registry,
                         ServingConfig(fallback_on_error=False),
                         predict_fn=boom)
    with pytest.raises(RuntimeError, match="model exploded"):
        layer.predict(blobs.X_test[0], request_id="t-4")


def test_canary_routing_is_deterministic(blobs, version):
    registry = ModelRegistry()
    registry.register(version, role="champion")
    challenger = ModelVersion(
        name="v2", seed=42, strategy="margin", n_labels=9,
        model_kind="logistic", metrics=dict(version.metrics),
        baseline_metrics=dict(version.baseline_metrics),
        model_state=dict(version.model_state),
        scaler_state=dict(version.scaler_state))
    registry.register(challenger, role="challenger")
    layer = ServingLayer(registry, ServingConfig(canary_pct=100.0))
    pred = layer.predict(blobs.X_test[0], request_id="canary-1")
    assert pred.arm == "challenger"
    assert layer.stats_dict()["n_canary"] == 1


def test_no_model_falls_back(blobs):
    registry = ModelRegistry()
    layer = ServingLayer(registry, ServingConfig())
    pred = layer.predict(blobs.X_test[0], request_id="t-5")
    assert pred.arm == "fallback"


def test_serving_config_validation():
    with pytest.raises(ValueError):
        ServingConfig(canary_pct=101.0)
    with pytest.raises(ValueError):
        ServingConfig(latency_budget_ms=0.0)
