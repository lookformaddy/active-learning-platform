"""Seeded synthetic data for the active-learning platform.

The dataset is deliberately small and fully synthetic: three Gaussian
blobs in a 6-dimensional feature space with a band of overlap between
classes, so that uncertainty-based query strategies have a real signal
to exploit and random sampling wastes labels on the easy interior of
each blob. Nothing here comes from any real system or employer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field


def _gauss(rng: random.Random) -> float:
    return rng.gauss(0.0, 1.0)


@dataclass
class Dataset:
    """A labelled synthetic dataset with a held-out test split."""

    X: list[list[float]]
    y: list[int]
    X_test: list[list[float]]
    y_test: list[int]
    n_classes: int = 3
    n_features: int = 6
    seed: int = 42

    def fingerprint(self) -> str:
        """Stable SHA-256 over the data — recorded in manifests."""
        payload = json.dumps(
            {"X": self.X, "y": self.y, "X_test": self.X_test,
             "y_test": self.y_test},
            sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for label in self.y:
            counts[label] = counts.get(label, 0) + 1
        return counts


# Class centres: pairwise distances are ~2.5–3.5 with unit-ish spread,
# so the classes are learnable but the boundaries are genuinely
# ambiguous — exactly where active learning earns its keep.
_CENTRES = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.6, 1.2, -0.8, 0.6, 1.4, -0.4],
    [-0.6, 2.4, 1.6, -1.2, 0.4, 1.8],
]
_SIGMA = 1.15


def generate_dataset(n_samples: int = 600, n_test: int = 300,
                     n_features: int = 6, n_classes: int = 3,
                     sigma: float = _SIGMA, seed: int = 42,
                     shape: str = "blobs") -> Dataset:
    """Generate a seeded synthetic classification dataset.

    ``shape="blobs"`` (default) draws Gaussian blobs per class — a
    near-linear problem where logistic regression is the right model.
    ``shape="moons"`` draws two interleaving crescents plus a central
    blob — a curved boundary where uncertainty sampling earns its keep
    and a linear model is at a structural disadvantage. The test split
    is drawn from the same distribution but with a different RNG
    stream, so test labels can never leak into pool selection.
    """
    if shape not in ("blobs", "moons"):
        raise ValueError(f"unknown shape {shape!r}; "
                         "expected 'blobs' or 'moons'")
    if shape == "moons":
        return _generate_moons(n_samples, n_test, n_features, seed)
    rng = random.Random(seed)
    centres = [_CENTRES[c][:n_features] for c in range(n_classes)]

    def draw() -> tuple[list[float], int]:
        label = rng.randrange(n_classes)
        centre = centres[label]
        return ([centre[d] + sigma * _gauss(rng) for d in range(n_features)],
                label)

    X, y = [], []
    for _ in range(n_samples):
        row, label = draw()
        X.append(row)
        y.append(label)
    # Shuffle the pool so class order carries no information.
    order = list(range(n_samples))
    rng.shuffle(order)
    X = [X[i] for i in order]
    y = [y[i] for i in order]

    rng_test = random.Random(seed + 1_000_003)
    X_test, y_test = [], []
    for _ in range(n_test):
        label = rng_test.randrange(n_classes)
        centre = centres[label]
        row = [centre[d] + sigma * _gauss(rng_test)
               for d in range(n_features)]
        X_test.append(row)
        y_test.append(label)
    return Dataset(X=X, y=y, X_test=X_test, y_test=y_test,
                   n_classes=n_classes, n_features=n_features, seed=seed)


def _generate_moons(n_samples: int, n_test: int, n_features: int,
                    seed: int, noise: float = 0.15) -> Dataset:
    """Two interleaving crescents + a central blob (3 classes).

    The first two features carry the crescent geometry; any remaining
    features are pure noise, so the problem also tests whether a model
    can ignore irrelevant dimensions. The curved boundary is why
    uncertainty sampling beats random here: random labels land mostly
    inside the crescents, margin sampling lands on the boundary.
    """
    rng = random.Random(seed)
    n_noise = max(0, n_features - 2)

    def draw(rng: random.Random) -> tuple[list[float], int]:
        label = rng.randrange(3)
        if label == 2:
            row = [rng.gauss(0.0, 0.4), rng.gauss(2.2, 0.4)]
        else:
            t = rng.uniform(0.0, math.pi)
            x = 2.0 * math.cos(t)
            y = 2.0 * math.sin(t)
            if label == 1:
                x = 1.0 - x
                y = 0.6 - y
            row = [x + rng.gauss(0.0, noise), y + rng.gauss(0.0, noise)]
        row.extend(rng.gauss(0.0, 0.3) for _ in range(n_noise))
        return row, label

    X, y = [], []
    for _ in range(n_samples):
        row, label = draw(rng)
        X.append(row)
        y.append(label)
    order = list(range(n_samples))
    rng.shuffle(order)
    X = [X[i] for i in order]
    y = [y[i] for i in order]

    rng_test = random.Random(seed + 1_000_003)
    X_test, y_test = [], []
    for _ in range(n_test):
        row, label = draw(rng_test)
        X_test.append(row)
        y_test.append(label)
    return Dataset(X=X, y=y, X_test=X_test, y_test=y_test,
                   n_classes=3, n_features=n_features, seed=seed)


@dataclass
class LabelledPool:
    """The active-learning state: labelled set + unlabelled pool.

    ``y_pool`` holds the *hidden* true labels used only to simulate an
    annotator answering a query. Production code must never read
    ``y_pool`` except through :meth:`annotate`, which mirrors a human
    labelling exactly the queried items.
    """

    X_pool: list[list[float]]
    y_pool: list[int]  # hidden ground truth, annotator-only access
    X_labeled: list[list[float]] = field(default_factory=list)
    y_labeled: list[int] = field(default_factory=list)
    labeled_ids: set[int] = field(default_factory=set)

    @property
    def n_labeled(self) -> int:
        return len(self.y_labeled)

    @property
    def n_unlabeled(self) -> int:
        return len(self.X_pool) - len(self.labeled_ids)

    def unlabeled_indices(self) -> list[int]:
        return [i for i in range(len(self.X_pool))
                if i not in self.labeled_ids]

    def annotate(self, indices: list[int]) -> list[tuple[int, int]]:
        """Simulate a human annotator labelling exactly these items.

        Returns ``(pool_index, label)`` pairs and moves the items into
        the labelled set. Asking for an already-labelled item is a
        caller bug and raises — labels are never double-counted.
        """
        out = []
        for idx in indices:
            if idx in self.labeled_ids:
                raise ValueError(f"pool item {idx} is already labelled")
            if not 0 <= idx < len(self.X_pool):
                raise IndexError(f"pool index {idx} out of range")
            self.labeled_ids.add(idx)
            self.X_labeled.append(list(self.X_pool[idx]))
            self.y_labeled.append(self.y_pool[idx])
            out.append((idx, self.y_pool[idx]))
        return out

    def seed(self, per_class: int = 3, seed: int = 42) -> None:
        """Take a small stratified seed set (``per_class`` per class)."""
        rng = random.Random(seed)
        by_class: dict[int, list[int]] = {}
        for i, label in enumerate(self.y_pool):
            by_class.setdefault(label, []).append(i)
        chosen: list[int] = []
        for label in sorted(by_class):
            members = list(by_class[label])
            rng.shuffle(members)
            chosen.extend(members[:per_class])
        self.annotate(sorted(chosen))

    def snapshot(self) -> dict:
        return {"n_labeled": self.n_labeled,
                "n_unlabeled": self.n_unlabeled,
                "class_counts": {str(k): v for k, v in
                                 self._labeled_class_counts().items()}}

    def _labeled_class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for label in self.y_labeled:
            counts[label] = counts.get(label, 0) + 1
        return counts


def make_pool(dataset: Dataset, seed_per_class: int = 3,
              seed: int = 42) -> LabelledPool:
    """Build a pool from a dataset with a stratified seed set."""
    pool = LabelledPool(X_pool=[list(r) for r in dataset.X],
                        y_pool=list(dataset.y))
    pool.seed(per_class=seed_per_class, seed=seed)
    return pool


def mean_uncertainty(probas: list[list[float]]) -> float:
    """Mean entropy of a batch of predicted class distributions."""
    if not probas:
        return 0.0
    total = 0.0
    for p in probas:
        total += -sum(pi * math.log(max(pi, 1e-15)) for pi in p if pi > 0)
    return total / len(probas)


def centroid_shift(X_a: list[list[float]],
                   X_b: list[list[float]]) -> float:
    """Euclidean distance between the feature centroids of two sets."""
    if not X_a or not X_b:
        return 0.0
    dim = len(X_a[0])
    ca = [sum(r[d] for r in X_a) / len(X_a) for d in range(dim)]
    cb = [sum(r[d] for r in X_b) / len(X_b) for d in range(dim)]
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(ca, cb)))
