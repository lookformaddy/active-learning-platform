"""Tests for learning-curve evaluation."""
from activelearn.data import generate_dataset
from activelearn.evaluation import (baseline_accuracy, compare_curves,
                                    mean_learning_curve, run_curve)


def test_run_curve_grows_labels_and_accuracy(blobs):
    curve = run_curve(blobs, "margin", rounds=3, batch_size=5,
                      seed_labels_per_class=2, epochs=60)
    assert len(curve.points) == 4  # round 0 + 3 rounds
    assert curve.points[0].n_labels == 6
    assert curve.points[-1].n_labels == 21
    assert all(0.0 <= p.test_accuracy <= 1.0 for p in curve.points)
    assert curve.points[-1].test_accuracy >= 0.5


def test_run_curve_random_is_control(blobs):
    active = run_curve(blobs, "margin", rounds=3, batch_size=5,
                       seed_labels_per_class=2, epochs=60)
    control = run_curve(blobs, "random", rounds=3, batch_size=5,
                        seed_labels_per_class=2, epochs=60)
    # Same seed labels and same budget: the curves start identically.
    assert active.points[0].n_labels == control.points[0].n_labels
    assert active.points[0].test_accuracy == control.points[0].test_accuracy


def test_labels_to_target_and_auc():
    from activelearn.evaluation import CurvePoint, LearningCurve
    curve = LearningCurve(strategy="margin", target_accuracy=0.8,
                          points=[CurvePoint(0, 6, 0.60, 1.0),
                                  CurvePoint(1, 16, 0.75, 0.9),
                                  CurvePoint(2, 26, 0.85, 0.8)])
    assert curve.labels_to_target() == 26
    assert curve.final_accuracy() == 0.85
    assert curve.area_under_curve() > 0
    never = LearningCurve(strategy="random", target_accuracy=0.99,
                          points=[CurvePoint(0, 6, 0.60, 1.0)])
    assert never.labels_to_target() is None


def test_compare_curves_reports_savings():
    from activelearn.evaluation import CurvePoint, LearningCurve
    active = LearningCurve(strategy="margin", target_accuracy=0.8,
                           points=[CurvePoint(0, 6, 0.60, 1.0),
                                   CurvePoint(1, 26, 0.85, 0.8)])
    control = LearningCurve(strategy="random", target_accuracy=0.8,
                            points=[CurvePoint(0, 6, 0.60, 1.0),
                                    CurvePoint(1, 56, 0.85, 0.8)])
    report = compare_curves(active, control)
    assert report["active_labels_to_target"] == 26
    assert report["random_labels_to_target"] == 56
    assert report["label_savings_pct"] == 53.6


def test_compare_curves_handles_unreached_target():
    from activelearn.evaluation import CurvePoint, LearningCurve
    active = LearningCurve(strategy="margin", target_accuracy=0.8,
                           points=[CurvePoint(0, 6, 0.60, 1.0),
                                   CurvePoint(1, 26, 0.85, 0.8)])
    control = LearningCurve(strategy="random", target_accuracy=0.8,
                            points=[CurvePoint(0, 6, 0.60, 1.0),
                                    CurvePoint(1, 26, 0.70, 0.9)])
    report = compare_curves(active, control)
    assert report["random_labels_to_target"] is None
    assert report["label_savings_pct"] is None


def test_mean_learning_curve_averages_seeds():
    def fn(seed):
        return generate_dataset(n_samples=120, n_test=80, seed=seed)

    curve = mean_learning_curve(fn, "margin", rounds=2, batch_size=5,
                                seeds=(42, 7), seed_labels_per_class=2,
                                model_kind="knn", target_accuracy=0.8)
    assert len(curve.points) == 3
    assert "mean of 2 seeds" in curve.strategy
    assert all(0.0 <= p.test_accuracy <= 1.0 for p in curve.points)


def test_baseline_accuracy(blobs):
    acc = baseline_accuracy(blobs, blobs.y[:9])
    assert 0.0 <= acc <= 0.6
