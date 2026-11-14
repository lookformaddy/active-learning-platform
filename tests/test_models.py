"""Tests for the from-scratch models (pure stdlib)."""
import math

import pytest

from activelearn.models import (KNNClassifier, LogisticRegression,
                                MajorityBaseline, Standardizer, accuracy,
                                per_class_recall)


def test_standardizer_centres_and_scales(blobs):
    scaler = Standardizer().fit(blobs.X)
    scaled = scaler.transform(blobs.X)
    dim = len(blobs.X[0])
    for d in range(dim):
        mean = sum(r[d] for r in scaled) / len(scaled)
        assert abs(mean) < 1e-9
        var = sum((r[d] - mean) ** 2 for r in scaled) / len(scaled)
        assert abs(math.sqrt(var) - 1.0) < 1e-9


def test_standardizer_roundtrip_dict(blobs):
    scaler = Standardizer().fit(blobs.X)
    restored = Standardizer.from_dict(scaler.to_dict())
    assert restored.mean == scaler.mean
    assert restored.std == scaler.std


def test_logreg_learns_blobs(blobs):
    scaler = Standardizer().fit(blobs.X)
    model = LogisticRegression(n_classes=3, epochs=150, seed=0)
    model.fit(scaler.transform(blobs.X), blobs.y)
    acc = accuracy(blobs.y_test,
                   model.predict(scaler.transform(blobs.X_test)))
    assert acc >= 0.80
    assert model.loss_history[0] > model.loss_history[-1]
    probas = model.predict_proba(scaler.transform(blobs.X_test[:5]))
    for p in probas:
        assert abs(sum(p) - 1.0) < 1e-9
        assert all(0.0 <= v <= 1.0 for v in p)


def test_logreg_serialises(blobs):
    scaler = Standardizer().fit(blobs.X)
    model = LogisticRegression(n_classes=3, epochs=20, seed=0)
    model.fit(scaler.transform(blobs.X), blobs.y)
    restored = LogisticRegression.from_dict(model.to_dict())
    assert restored.predict(scaler.transform(blobs.X_test)) == \
        model.predict(scaler.transform(blobs.X_test))


def test_knn_learns_blobs(blobs):
    scaler = Standardizer().fit(blobs.X)
    knn = KNNClassifier(k=5, n_classes=3)
    knn.fit(scaler.transform(blobs.X), blobs.y)
    acc = accuracy(blobs.y_test,
                   knn.predict(scaler.transform(blobs.X_test)))
    assert acc >= 0.80
    probas = knn.predict_proba(scaler.transform(blobs.X_test[:5]))
    for p in probas:
        assert abs(sum(p) - 1.0) < 1e-9


def test_knn_needs_enough_points():
    knn = KNNClassifier(k=5, n_classes=3)
    with pytest.raises(ValueError, match="at least k=5"):
        knn.fit([[0.0, 0.0]] * 3, [0, 1, 0])


def test_knn_serialises(blobs):
    scaler = Standardizer().fit(blobs.X)
    knn = KNNClassifier(k=3, n_classes=3)
    knn.fit(scaler.transform(blobs.X), blobs.y)
    restored = KNNClassifier.from_dict(knn.to_dict())
    assert restored.predict(scaler.transform(blobs.X_test)) == \
        knn.predict(scaler.transform(blobs.X_test))


def test_majority_baseline(blobs):
    baseline = MajorityBaseline().fit(blobs.y)
    preds = baseline.predict(len(blobs.y_test))
    assert all(p == baseline.majority for p in preds)
    assert 0.0 < accuracy(blobs.y_test, preds) < 0.6
    probas = baseline.predict_proba(3, 3)
    assert all(abs(sum(p) - 1.0) < 1e-9 for p in probas)


def test_accuracy_and_recall():
    assert accuracy([0, 1, 1, 2], [0, 1, 0, 2]) == 0.75
    assert accuracy([], []) == 0.0
    recalls = per_class_recall([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert recalls[0] == 0.5
    assert recalls[1] == 1.0
    assert per_class_recall([0, 0], [0, 0], 3)[2] == 0.0
