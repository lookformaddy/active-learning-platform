"""Active-learning run pipeline: seed → query → annotate → retrain.

One :class:`ActiveLearningRun` owns a single training run end to end:

1. build a seeded pool (a few labels per class);
2. each round: score the pool with the current model, pick a batch with
   the query strategy, spend labelling budget (fail-closed), send the
   batch through the annotation queue (simulated annotators here — the
   queue itself is the human-in-the-loop contract), and retrain when
   the :class:`RetrainPolicy` fires;
3. evaluate on the held-out test set, re-train across seeds for the
   stability gate, and emit a :class:`ModelVersion` the registry can
   promote.

The simulated annotators answer from the pool's hidden ground truth,
exactly as a human would answer the queried items — and only those.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from activelearn.budget import LabelingBudget
from activelearn.data import (Dataset, LabelledPool, centroid_shift,
                              make_pool, mean_uncertainty)
from activelearn.evaluation import baseline_accuracy, run_curve
from activelearn.models import (KNNClassifier, LogisticRegression,
                                MajorityBaseline, Standardizer, accuracy,
                                per_class_recall)
from activelearn.queue import AnnotationQueue
from activelearn.strategies import select_batch
from activelearn.triggers import RetrainPolicy, RetrainState


@dataclass
class ModelVersion:
    """A trained model version, ready for registry promotion gates."""

    name: str
    seed: int
    strategy: str
    n_labels: int
    model_kind: str
    metrics: dict
    baseline_metrics: dict
    model_state: dict
    scaler_state: dict
    manifest: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    strategy: str = "margin"
    model_kind: str = "logistic"
    rounds: int = 10
    batch_size: int = 10
    seed_labels_per_class: int = 3
    total_budget: int = 100
    target_accuracy: float = 0.80
    epochs: int = 200
    seed: int = 42
    stability_seeds: int = 3


def _train_model(model_kind: str, n_classes: int, seed: int, epochs: int):
    if model_kind == "logistic":
        return LogisticRegression(n_classes=n_classes, epochs=epochs,
                                  seed=seed)
    if model_kind == "knn":
        return KNNClassifier(k=5, n_classes=n_classes)
    raise ValueError(f"unknown model_kind {model_kind!r}")


def _fit_and_score(pool: LabelledPool, dataset: Dataset, config: RunConfig,
                   seed: int) -> tuple[object, Standardizer, dict]:
    scaler = Standardizer().fit(pool.X_labeled)
    X_train = scaler.transform(pool.X_labeled)
    X_test = scaler.transform(dataset.X_test)
    model = _train_model(config.model_kind, dataset.n_classes, seed,
                         config.epochs)
    model.fit(X_train, pool.y_labeled)
    preds = model.predict(X_test)
    recalls = per_class_recall(dataset.y_test, preds, dataset.n_classes)
    metrics = {
        "test_accuracy": accuracy(dataset.y_test, preds),
        "min_per_class_recall": min(recalls.values()),
        "per_class_recall": {str(k): round(v, 4)
                             for k, v in recalls.items()},
        # Stored so the serving fallback can reconstruct the majority
        # baseline without keeping the training data around.
        "train_class_counts": {str(k): pool.y_labeled.count(k)
                               for k in set(pool.y_labeled)},
    }
    return model, scaler, metrics


class ActiveLearningRun:
    """One full active-learning run with budget, queue and triggers."""

    def __init__(self, dataset: Dataset, config: RunConfig,
                 annotators: list[str] | None = None,
                 senior_annotators: list[str] | None = None) -> None:
        self.dataset = dataset
        self.config = config
        self.pool: LabelledPool = make_pool(
            dataset, seed_per_class=config.seed_labels_per_class,
            seed=config.seed)
        self.budget = LabelingBudget(total_budget=config.total_budget,
                                     per_round_cap=config.batch_size)
        self.policy = RetrainPolicy(batch_size=config.batch_size)
        self.queue = AnnotationQueue(
            annotators or ["annotator-1", "annotator-2"],
            senior_annotators or ["senior-1"])
        self.round_log: list[dict] = []
        self.retrain_count = 0
        self._labels_since_retrain = 0
        self._train_centroid: list[float] | None = None

    def _current_model(self):
        model, _scaler, _metrics = _fit_and_score(
            self.pool, self.dataset, self.config, self.config.seed)
        return model

    def run(self) -> ModelVersion:
        cfg = self.config
        # Initial model on the seed labels.
        model = self._current_model()
        self.retrain_count = 1
        self._labels_since_retrain = 0
        self._train_centroid = self._centroid(self.pool.X_labeled)

        for round_no in range(1, cfg.rounds + 1):
            unlabeled = self.pool.unlabeled_indices()
            if not unlabeled or self.budget.exhausted:
                break
            scaler = Standardizer().fit(self.pool.X_labeled)
            probas = model.predict_proba(scaler.transform(
                [self.pool.X_pool[i] for i in unlabeled]))
            batch = min(cfg.batch_size, len(unlabeled),
                        self.budget.remaining)
            chosen = select_batch(unlabeled, probas,
                                  [self.pool.X_pool[i] for i in unlabeled],
                                  batch, cfg.strategy,
                                  seed=cfg.seed + round_no)
            self.budget.request(len(chosen), round_no,
                                purpose=cfg.strategy)
            # Human-in-the-loop: queue, assign, label, adjudicate.
            for idx in chosen:
                self.queue.enqueue(
                    idx, self.pool.X_pool[idx],
                    model_prediction=int(model.predict(
                        scaler.transform([self.pool.X_pool[idx]]))[0]),
                    model_confidence=float(max(model.predict_proba(
                        scaler.transform([self.pool.X_pool[idx]]))[0])))
            for idx in chosen:
                self.queue.assign(idx)
                # Simulated annotators answer from hidden ground truth.
                self.queue.submit_label(
                    idx, self.queue.items[idx].assignees[0],
                    self.pool.y_pool[idx])
            # Double-label 1 in 5 items to measure agreement; the
            # second annotator is deliberately noisy on a small share
            # (one item every third round) so adjudication is exercised,
            # not just defined.
            for j, idx in enumerate(chosen):
                if j % 5 == 4:
                    second = self.queue.assign(idx)
                    true_label = self.pool.y_pool[idx]
                    noisy = (true_label + 1) % self.dataset.n_classes \
                        if j % 10 == 9 and round_no % 3 == 0 else true_label
                    self.queue.submit_label(idx, second, noisy)
                    item = self.queue.items[idx]
                    if item.disagreement:
                        self.queue.adjudicate(idx, true_label)
            # Labels enter the pool only after the queue finalises them.
            newly = [(it.pool_index, it.final_label)
                     for it in self.queue.labelled()
                     if it.pool_index in chosen and it.final_label is not None]
            self.pool.annotate([i for i, _ in newly])
            self._labels_since_retrain += len(newly)

            # Retrain decision.
            new_centroid = self._centroid(self.pool.X_labeled)
            shift = centroid_shift([self._train_centroid or [0.0] * self.dataset.n_features],
                                   [new_centroid]) if self._train_centroid else 0.0
            state = RetrainState(
                labels_since_retrain=self._labels_since_retrain,
                mean_pool_uncertainty=mean_uncertainty(probas),
                centroid_shift=shift,
                rounds_since_retrain=round_no)
            fired, reasons = self.policy.should_retrain(state, round_no)
            if fired:
                model = self._current_model()
                self.retrain_count += 1
                self._labels_since_retrain = 0
                self._train_centroid = new_centroid
            self.round_log.append({
                "round": round_no, "n_labels": self.pool.n_labeled,
                "batch": len(chosen), "retrain": fired,
                "reasons": reasons,
                "budget_remaining": self.budget.remaining,
                "disagreements": self.queue.summary()["n_disagreements"],
            })

        # Final fit + stability across seeds.
        model, scaler, metrics = _fit_and_score(
            self.pool, self.dataset, cfg, cfg.seed)
        seed_accs = []
        for s in range(cfg.stability_seeds):
            m, _sc, met = _fit_and_score(self.pool, self.dataset, cfg,
                                         seed=cfg.seed + 100 + s)
            seed_accs.append(met["test_accuracy"])
        metrics["seed_accuracies"] = [round(a, 4) for a in seed_accs]
        metrics["seed_accuracy_std"] = round(statistics.pstdev(seed_accs)
                                             if len(seed_accs) > 1 else 0.0, 4)

        # Learning-curve comparison for the label-efficiency gate.
        curve = run_curve(self.dataset, cfg.strategy, cfg.rounds,
                          cfg.batch_size,
                          seed_labels_per_class=cfg.seed_labels_per_class,
                          model_kind=cfg.model_kind, seed=cfg.seed,
                          target_accuracy=cfg.target_accuracy,
                          epochs=cfg.epochs)
        control = run_curve(self.dataset, "random", cfg.rounds,
                            cfg.batch_size,
                            seed_labels_per_class=cfg.seed_labels_per_class,
                            model_kind=cfg.model_kind, seed=cfg.seed,
                            target_accuracy=cfg.target_accuracy,
                            epochs=cfg.epochs)
        metrics["labels_to_target"] = curve.labels_to_target()
        metrics["learning_curve"] = curve.summary()
        baseline_metrics = {
            "majority_test_accuracy": baseline_accuracy(
                self.dataset, self.pool.y_labeled),
            "random_labels_to_target": control.labels_to_target(),
            "random_final_accuracy": round(control.final_accuracy(), 4),
        }
        model_state = model.to_dict() if hasattr(model, "to_dict") else {}
        return ModelVersion(
            name=f"{cfg.strategy}-{cfg.model_kind}-v1",
            seed=cfg.seed, strategy=cfg.strategy,
            n_labels=self.pool.n_labeled, model_kind=cfg.model_kind,
            metrics=metrics, baseline_metrics=baseline_metrics,
            model_state=model_state,
            scaler_state=scaler.to_dict())

    def _centroid(self, X: list[list[float]]) -> list[float]:
        dim = len(X[0]) if X else 0
        return [sum(r[d] for r in X) / len(X) for d in range(dim)] if X else [0.0] * dim

    def summary(self) -> dict:
        return {"strategy": self.config.strategy,
                "model_kind": self.config.model_kind,
                "n_labels": self.pool.n_labeled,
                "budget": self.budget.summary(),
                "retrains": self.retrain_count,
                "queue": self.queue.summary(),
                "rounds": self.round_log}
