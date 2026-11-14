"""From-scratch classifiers (pure stdlib — no sklearn / numpy / torch).

- :class:`Standardizer` — mean/std scaling, fitted on the labelled set
  only, so pool statistics can never leak into training.
- :class:`LogisticRegression` — multinomial softmax regression, full-batch
  gradient descent with L2 regularisation and a stable softmax.
- :class:`KNNClassifier` — k-nearest-neighbour with distance-weighted
  voting, returning a proper probability distribution so every query
  strategy works on either model.
- :class:`MajorityBaseline` — the trivial classifier every learned
  model must beat; also the serving fallback.
"""

from __future__ import annotations

import math
import random


def _softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class Standardizer:
    """Mean/std scaler (fit on train, applied everywhere)."""

    def __init__(self) -> None:
        self.mean: list[float] = []
        self.std: list[float] = []

    def fit(self, X: list[list[float]]) -> "Standardizer":
        n = len(X)
        dim = len(X[0]) if X else 0
        self.mean = [sum(row[d] for row in X) / n for d in range(dim)]
        self.std = []
        for d in range(dim):
            var = sum((row[d] - self.mean[d]) ** 2 for row in X) / n
            self.std.append(math.sqrt(var) if var > 1e-12 else 1.0)
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [[(row[d] - self.mean[d]) / self.std[d]
                 for d in range(len(row))] for row in X]

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, data: dict) -> "Standardizer":
        obj = cls()
        obj.mean = list(data["mean"])
        obj.std = list(data["std"])
        return obj


class LogisticRegression:
    """Multinomial logistic regression, full-batch gradient descent."""

    def __init__(self, n_classes: int, lr: float = 0.1, epochs: int = 200,
                 l2: float = 1e-3, seed: int = 0) -> None:
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.weights: list[list[float]] = []  # [class][feature]
        self.bias: list[float] = []
        self.loss_history: list[float] = []

    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        rng = random.Random(self.seed)
        n = len(X)
        dim = len(X[0])
        self.weights = [[rng.gauss(0.0, 0.01) for _ in range(dim)]
                        for _ in range(self.n_classes)]
        self.bias = [0.0] * self.n_classes
        self.loss_history = []
        for _epoch in range(self.epochs):
            grad_w = [[0.0] * dim for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            loss = 0.0
            for i in range(n):
                logits = [sum(self.weights[c][d] * X[i][d]
                              for d in range(dim)) + self.bias[c]
                          for c in range(self.n_classes)]
                probs = _softmax(logits)
                loss -= math.log(max(probs[y[i]], 1e-15))
                for c in range(self.n_classes):
                    err = probs[c] - (1.0 if c == y[i] else 0.0)
                    grad_b[c] += err
                    for d in range(dim):
                        grad_w[c][d] += err * X[i][d]
            for c in range(self.n_classes):
                for d in range(dim):
                    grad = grad_w[c][d] / n + self.l2 * self.weights[c][d]
                    self.weights[c][d] -= self.lr * grad
                self.bias[c] -= self.lr * grad_b[c] / n
            self.loss_history.append(loss / n)
        return self

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        out = []
        dim = len(self.weights[0]) if self.weights else 0
        for row in X:
            logits = [sum(self.weights[c][d] * row[d] for d in range(dim))
                      + self.bias[c] for c in range(self.n_classes)]
            out.append(_softmax(logits))
        return out

    def predict(self, X: list[list[float]]) -> list[int]:
        return [max(range(self.n_classes), key=lambda c: p[c])
                for p in self.predict_proba(X)]

    def to_dict(self) -> dict:
        return {"kind": "logistic_regression", "n_classes": self.n_classes,
                "lr": self.lr, "epochs": self.epochs, "l2": self.l2,
                "seed": self.seed, "weights": self.weights,
                "bias": self.bias}

    @classmethod
    def from_dict(cls, data: dict) -> "LogisticRegression":
        obj = cls(n_classes=data["n_classes"], lr=data["lr"],
                  epochs=data["epochs"], l2=data["l2"], seed=data["seed"])
        obj.weights = [list(w) for w in data["weights"]]
        obj.bias = list(data["bias"])
        return obj


class KNNClassifier:
    """k-NN with inverse-distance-weighted voting.

    ``predict_proba`` returns a proper distribution (votes normalised),
    so uncertainty-based query strategies treat k-NN exactly like the
    logistic model. Ties break toward the lower class index for
    determinism.
    """

    def __init__(self, k: int = 5, n_classes: int = 3) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.n_classes = n_classes
        self.X_train: list[list[float]] = []
        self.y_train: list[int] = []

    def fit(self, X: list[list[float]], y: list[int]) -> "KNNClassifier":
        if len(X) < self.k:
            raise ValueError(
                f"k-NN needs at least k={self.k} training points, "
                f"got {len(X)}")
        self.X_train = [list(r) for r in X]
        self.y_train = list(y)
        return self

    def _distances(self, row: list[float]) -> list[tuple[float, int]]:
        out = []
        for i, train_row in enumerate(self.X_train):
            d = math.sqrt(sum((a - b) ** 2
                              for a, b in zip(row, train_row)))
            out.append((d, i))
        out.sort(key=lambda t: (t[0], t[1]))
        return out[: self.k]

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        out = []
        for row in X:
            votes = [0.0] * self.n_classes
            for dist, i in self._distances(row):
                weight = 1.0 / (dist + 1e-9)
                votes[self.y_train[i]] += weight
            total = sum(votes)
            out.append([v / total for v in votes])
        return out

    def predict(self, X: list[list[float]]) -> list[int]:
        return [max(range(self.n_classes), key=lambda c: p[c])
                for p in self.predict_proba(X)]

    def to_dict(self) -> dict:
        return {"kind": "knn", "k": self.k, "n_classes": self.n_classes,
                "X_train": self.X_train, "y_train": self.y_train}

    @classmethod
    def from_dict(cls, data: dict) -> "KNNClassifier":
        obj = cls(k=data["k"], n_classes=data["n_classes"])
        obj.X_train = [list(r) for r in data["X_train"]]
        obj.y_train = list(data["y_train"])
        return obj


class MajorityBaseline:
    """Predicts the most frequent training label for every input."""

    def __init__(self) -> None:
        self.majority: int = 0
        self.distribution: dict[int, float] = {}

    def fit(self, y: list[int]) -> "MajorityBaseline":
        counts: dict[int, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        self.majority = max(counts, key=lambda k: counts[k])
        total = len(y)
        self.distribution = {k: v / total for k, v in counts.items()}
        return self

    def predict(self, n: int) -> list[int]:
        return [self.majority] * n

    def predict_proba(self, n: int, n_classes: int) -> list[list[float]]:
        return [[self.distribution.get(c, 0.0) for c in range(n_classes)]
                for _ in range(n)]


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Plain classification accuracy; empty input scores 0.0."""
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def per_class_recall(y_true: list[int], y_pred: list[int],
                     n_classes: int) -> dict[int, float]:
    """Recall per class; a class absent from the truth scores 0.0."""
    out: dict[int, float] = {}
    for c in range(n_classes):
        support = [i for i, t in enumerate(y_true) if t == c]
        if not support:
            out[c] = 0.0
        else:
            out[c] = sum(1 for i in support if y_pred[i] == c) / len(support)
    return out
