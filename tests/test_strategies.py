"""Tests for the query strategies."""
import pytest

from activelearn.models import LogisticRegression, Standardizer
from activelearn.strategies import STRATEGIES, select_batch

N = 40


@pytest.fixture(scope="module")
def scored_pool(blobs):
    scaler = Standardizer().fit(blobs.X[:30])
    model = LogisticRegression(n_classes=3, epochs=80, seed=0)
    model.fit(scaler.transform(blobs.X[:30]), blobs.y[:30])
    X = blobs.X[30:30 + N]
    probas = model.predict_proba(scaler.transform(X))
    return list(range(30, 30 + N)), probas, X


def test_all_strategies_return_batch_size(scored_pool):
    indices, probas, X = scored_pool
    for strategy in STRATEGIES:
        chosen = select_batch(indices, probas, X, 10, strategy, seed=1)
        assert len(chosen) == 10
        assert len(set(chosen)) == 10
        assert all(i in indices for i in chosen)


def test_margin_picks_most_uncertain(scored_pool):
    indices, probas, X = scored_pool
    chosen = select_batch(indices, probas, X, 5, "margin", seed=1)

    def margin(p):
        ranked = sorted(p, reverse=True)
        return ranked[0] - ranked[1]

    chosen_margins = [margin(probas[indices.index(i)]) for i in chosen]
    rest = [margin(p) for j, p in enumerate(probas)
            if indices[j] not in chosen]
    assert max(chosen_margins) <= min(rest)


def test_least_confidence_picks_lowest_max_prob(scored_pool):
    indices, probas, X = scored_pool
    chosen = select_batch(indices, probas, X, 5, "least_confidence", seed=1)
    chosen_conf = [max(probas[indices.index(i)]) for i in chosen]
    rest_conf = [max(p) for j, p in enumerate(probas)
                 if indices[j] not in chosen]
    assert max(chosen_conf) <= min(rest_conf)


def test_entropy_picks_highest_entropy(scored_pool):
    import math
    indices, probas, X = scored_pool
    chosen = select_batch(indices, probas, X, 5, "entropy", seed=1)

    def ent(p):
        return -sum(v * math.log(max(v, 1e-15)) for v in p if v > 0)

    chosen_ent = [ent(probas[indices.index(i)]) for i in chosen]
    rest_ent = [ent(p) for j, p in enumerate(probas)
                if indices[j] not in chosen]
    assert min(chosen_ent) >= max(rest_ent)


def test_random_is_seed_deterministic(scored_pool):
    indices, probas, X = scored_pool
    a = select_batch(indices, probas, X, 8, "random", seed=7)
    b = select_batch(indices, probas, X, 8, "random", seed=7)
    c = select_batch(indices, probas, X, 8, "random", seed=8)
    assert a == b
    assert a != c


def test_diversity_spreads_across_space(scored_pool):
    indices, probas, X = scored_pool
    chosen = select_batch(indices, probas, X, 10, "diversity", seed=1)
    assert len(set(chosen)) == 10
    # A diversity batch is not the same as the pure-margin batch.
    margin = select_batch(indices, probas, X, 10, "margin", seed=1)
    assert set(chosen) != set(margin)


def test_hybrid_mixes_strategies(scored_pool):
    indices, probas, X = scored_pool
    chosen = select_batch(indices, probas, X, 10, "hybrid", seed=1)
    assert len(set(chosen)) == 10


def test_strategy_errors(scored_pool):
    indices, probas, X = scored_pool
    with pytest.raises(ValueError, match="unknown strategy"):
        select_batch(indices, probas, X, 5, "nope")
    with pytest.raises(ValueError, match="exceeds unlabelled pool"):
        select_batch(indices, probas, X, len(indices) + 1, "margin")
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        select_batch(indices, probas, X, 0, "margin")
    with pytest.raises(ValueError, match="empty pool"):
        select_batch([], [], [], 1, "margin")
