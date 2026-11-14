"""Query strategies: which unlabelled items are worth a human label.

Every strategy takes the pool's unlabelled indices, the model's
predicted class probabilities for those items, and the raw features,
and returns exactly ``batch_size`` indices ranked most valuable first.
Ties break by pool index so runs are deterministic for a fixed seed.

Strategies
-----------
- ``least_confidence`` — lowest max-class probability.
- ``margin`` — smallest gap between the top two class probabilities.
- ``entropy`` — highest predictive entropy.
- ``random`` — uniform random sample; the baseline every strategy
  must beat (it also doubles as the control arm in evaluation).
- ``diversity`` — cluster-based: k-means over the pool features, then
  the most uncertain item from each cluster, so a batch never spends
  its whole budget on near-duplicates from one ambiguous region.
- ``hybrid`` — half the batch by margin, half by diversity.
"""

from __future__ import annotations

import math
import random

STRATEGIES = ("least_confidence", "margin", "entropy", "random",
              "diversity", "hybrid")


def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log(max(p, 1e-15)) for p in probs if p > 0)


def _least_confidence(probs: list[float]) -> float:
    return 1.0 - max(probs)


def _margin(probs: list[float]) -> float:
    ranked = sorted(probs, reverse=True)
    return ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]


def _uncertainty_scores(probs: list[list[float]],
                        strategy: str) -> list[float]:
    if strategy == "least_confidence":
        return [_least_confidence(p) for p in probs]
    if strategy == "margin":
        # Higher score = more valuable = smaller margin.
        return [1.0 - _margin(p) for p in probs]
    if strategy == "entropy":
        return [_entropy(p) for p in probs]
    raise ValueError(f"unknown uncertainty strategy {strategy!r}")


def _kmeans(X: list[list[float]], k: int, seed: int,
            iterations: int = 20) -> list[int]:
    """Deterministic k-means++-lite; returns a cluster id per row."""
    rng = random.Random(seed)
    n = len(X)
    dim = len(X[0])
    k = min(k, n)
    # Seeded farthest-point initialisation (deterministic given the seed).
    first = rng.randrange(n)
    centres = [list(X[first])]
    while len(centres) < k:
        dists = []
        for row in X:
            d = min(sum((a - b) ** 2 for a, b in zip(row, c))
                    for c in centres)
            dists.append(d)
        total = sum(dists)
        if total <= 0:
            centres.append(list(X[rng.randrange(n)]))
            continue
        pick = rng.random() * total
        acc = 0.0
        for i, d in enumerate(dists):
            acc += d
            if acc >= pick:
                centres.append(list(X[i]))
                break
        else:
            centres.append(list(X[-1]))
    labels = [0] * n
    for _ in range(iterations):
        changed = False
        for i, row in enumerate(X):
            best = min(range(k),
                       key=lambda c: sum((a - b) ** 2
                                         for a, b in zip(row, centres[c])))
            if best != labels[i]:
                labels[i] = best
                changed = True
        if not changed:
            break
        for c in range(k):
            members = [X[i] for i in range(n) if labels[i] == c]
            if members:
                centres[c] = [sum(r[d] for r in members) / len(members)
                              for d in range(dim)]
    return labels


def select_batch(indices: list[int], probas: list[list[float]],
                 X: list[list[float]], batch_size: int,
                 strategy: str, seed: int = 0) -> list[int]:
    """Pick ``batch_size`` pool indices to send for labelling.

    ``probas`` and ``X`` are aligned with ``indices`` (one entry per
    unlabelled item). Raises :class:`ValueError` on an unknown strategy
    or a batch larger than the available pool — callers must never ask
    for more labels than exist.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; "
                         f"expected one of {STRATEGIES}")
    if not indices:
        raise ValueError("cannot select from an empty pool")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if batch_size > len(indices):
        raise ValueError(
            f"batch_size {batch_size} exceeds unlabelled pool "
            f"of {len(indices)}")

    if strategy == "random":
        rng = random.Random(seed)
        chosen = list(indices)
        rng.shuffle(chosen)
        return sorted(chosen[:batch_size])

    scores = _uncertainty_scores(probas, strategy) \
        if strategy in ("least_confidence", "margin", "entropy") else None

    if strategy in ("least_confidence", "margin", "entropy"):
        assert scores is not None
        ranked = sorted(range(len(indices)),
                        key=lambda i: (-scores[i], indices[i]))
        return sorted(indices[i] for i in ranked[:batch_size])

    if strategy == "diversity":
        clusters = _kmeans(X, batch_size, seed)
        # Most uncertain item per cluster (entropy), then fill any
        # shortfall (empty clusters) with the next-most-uncertain.
        ent = [_entropy(p) for p in probas]
        by_cluster: dict[int, list[int]] = {}
        for pos, c in enumerate(clusters):
            by_cluster.setdefault(c, []).append(pos)
        picked: list[int] = []
        for c in sorted(by_cluster):
            members = sorted(by_cluster[c],
                             key=lambda p: (-ent[p], indices[p]))
            picked.append(members[0])
        if len(picked) < batch_size:
            rest = sorted((p for p in range(len(indices))
                           if p not in picked),
                          key=lambda p: (-ent[p], indices[p]))
            picked.extend(rest[: batch_size - len(picked)])
        return sorted(indices[p] for p in picked[:batch_size])

    # hybrid: half the batch by margin, half by diversity, no overlap.
    half = batch_size // 2
    margin_scores = _uncertainty_scores(probas, "margin")
    margin_ranked = sorted(range(len(indices)),
                           key=lambda i: (-margin_scores[i], indices[i]))
    picked_pos = set(margin_ranked[:half])
    diversity_batch = select_batch(indices, probas, X, batch_size,
                                   "diversity", seed)
    for idx in diversity_batch:
        pos = indices.index(idx)
        if len(picked_pos) < batch_size:
            picked_pos.add(pos)
    if len(picked_pos) < batch_size:
        rest = sorted((p for p in range(len(indices))
                       if p not in picked_pos),
                      key=lambda p: (-margin_scores[p], indices[p]))
        picked_pos.update(rest[: batch_size - len(picked_pos)])
    return sorted(indices[p] for p in picked_pos)


def strategy_report(indices: list[int], probas: list[list[float]],
                    strategy: str) -> dict:
    """Summarise what a strategy would pick, for logging/debugging."""
    if strategy == "random":
        return {"strategy": strategy, "mean_entropy": None}
    scores = _uncertainty_scores(probas, strategy) \
        if strategy in ("least_confidence", "margin", "entropy") else \
        [_entropy(p) for p in probas]
    return {"strategy": strategy,
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "max_score": round(max(scores), 4) if scores else 0.0,
            "n_candidates": len(indices)}
