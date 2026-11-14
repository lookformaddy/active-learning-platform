"""Pytest configuration: put the src layout on the import path so plain
``pytest`` passes from the project root, with or without an install.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activelearn.data import generate_dataset, make_pool  # noqa: E402
from activelearn.pipeline import (ModelVersion, RunConfig,  # noqa: E402
                                  _fit_and_score)
from activelearn.registry import ModelRegistry  # noqa: E402
from activelearn.serving import ServingConfig, ServingLayer  # noqa: E402

SEED = 42


@pytest.fixture(scope="session")
def blobs():
    return generate_dataset(n_samples=150, n_test=100, seed=SEED)


@pytest.fixture(scope="session")
def moons():
    return generate_dataset(n_samples=150, n_test=100, seed=SEED,
                            shape="moons")


@pytest.fixture(scope="session")
def pool(blobs):
    return make_pool(blobs, seed_per_class=3, seed=SEED)


@pytest.fixture(scope="session")
def version(blobs, pool):
    """A real trained version (logistic, seed labels only)."""
    config = RunConfig(strategy="margin", rounds=1, batch_size=5,
                       epochs=60, seed=SEED)
    model, scaler, metrics = _fit_and_score(pool, blobs, config, SEED)
    metrics["seed_accuracies"] = [metrics["test_accuracy"]]
    metrics["seed_accuracy_std"] = 0.0
    metrics["labels_to_target"] = None
    return ModelVersion(name="v1-test", seed=SEED, strategy="margin",
                        n_labels=pool.n_labeled, model_kind="logistic",
                        metrics=metrics,
                        baseline_metrics={"majority_test_accuracy": 0.34,
                                          "random_labels_to_target": None},
                        model_state=model.to_dict(),
                        scaler_state=scaler.to_dict())


@pytest.fixture(scope="session")
def registry(version):
    reg = ModelRegistry()
    reg.register(version, role="champion")
    return reg


@pytest.fixture(scope="session")
def serving(registry):
    return ServingLayer(registry, ServingConfig(canary_pct=0.0))
