"""Learning-curve evaluation: active learning vs the random baseline.

The whole point of the platform is label efficiency, so evaluation is
done as a *curve*, not a single number: after every query round the
model is retrained and scored on the held-out test set, for both the
strategy under test and a random-sampling control run on the same seed
and the same starting labels. The headline metric is
``labels_to_target`` — how many labels each arm needed to first reach
the target accuracy — and the derived ``label_savings_pct``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from activelearn.budget import LabelingBudget
from activelearn.data import Dataset, LabelledPool, make_pool, mean_uncertainty
from activelearn.models import (KNNClassifier, LogisticRegression,
                                MajorityBaseline, Standardizer, accuracy)
from activelearn.strategies import select_batch


@dataclass
class CurvePoint:
    round_no: int
    n_labels: int
    test_accuracy: float
    mean_pool_uncertainty: float


@dataclass
class LearningCurve:
    strategy: str
    points: list[CurvePoint] = field(default_factory=list)
    target_accuracy: float = 0.80

    def labels_to_target(self) -> int | None:
        """Labels needed to first reach the target, or None if never."""
        for p in self.points:
            if p.test_accuracy >= self.target_accuracy:
                return p.n_labels
        return None

    def final_accuracy(self) -> float:
        return self.points[-1].test_accuracy if self.points else 0.0

    def area_under_curve(self) -> float:
        """Trapezoid area over (n_labels, accuracy) — higher is better."""
        if len(self.points) < 2:
            return 0.0
        area = 0.0
        for a, b in zip(self.points, self.points[1:]):
            width = b.n_labels - a.n_labels
            area += width * (a.test_accuracy + b.test_accuracy) / 2.0
        return area

    def summary(self) -> dict:
        return {"strategy": self.strategy,
                "target_accuracy": self.target_accuracy,
                "labels_to_target": self.labels_to_target(),
                "final_accuracy": round(self.final_accuracy(), 4),
                "final_n_labels": self.points[-1].n_labels if self.points else 0,
                "area_under_curve": round(self.area_under_curve(), 2),
                "points": [{"round": p.round_no, "n_labels": p.n_labels,
                            "test_accuracy": round(p.test_accuracy, 4)}
                           for p in self.points]}


def _make_model(model_kind: str, n_classes: int, seed: int,
                epochs: int = 200):
    if model_kind == "logistic":
        return LogisticRegression(n_classes=n_classes, epochs=epochs,
                                  seed=seed)
    if model_kind == "knn":
        return KNNClassifier(k=5, n_classes=n_classes)
    raise ValueError(f"unknown model_kind {model_kind!r}")


def run_curve(dataset: Dataset, strategy: str, rounds: int,
              batch_size: int, seed_labels_per_class: int = 3,
              model_kind: str = "logistic", seed: int = 42,
              target_accuracy: float = 0.80,
              epochs: int = 200) -> LearningCurve:
    """Run one arm of the experiment and return its learning curve.

    The pool, seed labels, test set and model hyper-parameters are
    identical across arms — only the query strategy differs, so the
    comparison is apples-to-apples.
    """
    pool: LabelledPool = make_pool(dataset,
                                   seed_per_class=seed_labels_per_class,
                                   seed=seed)
    curve = LearningCurve(strategy=strategy,
                          target_accuracy=target_accuracy)
    budget = LabelingBudget(total_budget=rounds * batch_size,
                            per_round_cap=batch_size)
    for round_no in range(rounds + 1):
        scaler = Standardizer().fit(pool.X_labeled)
        X_train = scaler.transform(pool.X_labeled)
        X_test = scaler.transform(dataset.X_test)
        model = _make_model(model_kind, dataset.n_classes, seed, epochs)
        model.fit(X_train, pool.y_labeled)
        test_acc = accuracy(dataset.y_test, model.predict(X_test))
        unlabeled = pool.unlabeled_indices()
        probas = model.predict_proba(scaler.transform(
            [pool.X_pool[i] for i in unlabeled])) if unlabeled else []
        curve.points.append(CurvePoint(
            round_no=round_no, n_labels=pool.n_labeled,
            test_accuracy=test_acc,
            mean_pool_uncertainty=mean_uncertainty(probas)))
        if round_no == rounds or not unlabeled:
            break
        budget.request(batch_size, round_no, purpose=strategy)
        chosen = select_batch(unlabeled, probas,
                              [pool.X_pool[i] for i in unlabeled],
                              batch_size, strategy, seed=seed + round_no)
        pool.annotate(chosen)
    return curve


def mean_learning_curve(dataset_fn, strategy: str, rounds: int,
                        batch_size: int, seeds: tuple[int, ...],
                        seed_labels_per_class: int = 2,
                        model_kind: str = "logistic",
                        target_accuracy: float = 0.80,
                        epochs: int = 200) -> LearningCurve:
    """Mean learning curve across seeds — the honest comparison.

    ``labels_to_target`` on a single seed is noisy: one lucky round
    can cross the target early. Averaging the per-round accuracy over
    several seeds (same config, different data draws) gives a stable
    curve; the headline label savings are computed from these means.
    ``dataset_fn`` maps a seed to a fresh :class:`Dataset`.
    """
    curves = [run_curve(dataset_fn(s), strategy, rounds, batch_size,
                        seed_labels_per_class=seed_labels_per_class,
                        model_kind=model_kind, seed=s,
                        target_accuracy=target_accuracy, epochs=epochs)
              for s in seeds]
    points = []
    for i, template in enumerate(curves[0].points):
        accs = [c.points[i].test_accuracy for c in curves]
        uncs = [c.points[i].mean_pool_uncertainty for c in curves]
        points.append(CurvePoint(round_no=template.round_no,
                                 n_labels=template.n_labels,
                                 test_accuracy=sum(accs) / len(accs),
                                 mean_pool_uncertainty=sum(uncs) / len(uncs)))
    return LearningCurve(strategy=f"{strategy} (mean of {len(seeds)} seeds)",
                         points=points, target_accuracy=target_accuracy)


def compare_curves(active: LearningCurve, control: LearningCurve) -> dict:
    """Headline comparison: label savings of the active arm vs random."""
    a_target = active.labels_to_target()
    c_target = control.labels_to_target()
    savings = None
    if a_target is not None and c_target is not None and c_target > 0:
        savings = (c_target - a_target) / c_target
    return {
        "active_strategy": active.strategy,
        "control_strategy": control.strategy,
        "target_accuracy": active.target_accuracy,
        "active_labels_to_target": a_target,
        "random_labels_to_target": c_target,
        "label_savings_pct": (round(savings * 100, 1)
                              if savings is not None else None),
        "active_final_accuracy": round(active.final_accuracy(), 4),
        "random_final_accuracy": round(control.final_accuracy(), 4),
        "active_auc": round(active.area_under_curve(), 2),
        "random_auc": round(control.area_under_curve(), 2),
    }


def baseline_accuracy(dataset: Dataset, y_labeled: list[int]) -> float:
    """Majority-class baseline accuracy on the test set."""
    baseline = MajorityBaseline().fit(y_labeled)
    return accuracy(dataset.y_test, baseline.predict(len(dataset.y_test)))
