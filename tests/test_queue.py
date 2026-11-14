"""Tests for the annotation queue: assignment, agreement, adjudication."""
import pytest

from activelearn.queue import AnnotationQueue, cohens_kappa


@pytest.fixture()
def queue():
    return AnnotationQueue(["ann-a", "ann-b"], ["senior-1"],
                           double_label_fraction=0.2)


def _enqueue(queue, n=4):
    for i in range(n):
        queue.enqueue(i, [float(i)] * 3, model_prediction=i % 3,
                      model_confidence=0.6)


def test_enqueue_assign_label_happy_path(queue):
    _enqueue(queue, 1)
    assignee = queue.assign(0)
    assert assignee in ("ann-a", "ann-b")
    item = queue.submit_label(0, assignee, 2)
    assert item.final_label == 2
    assert item.status == "labelled"
    assert queue.labelled()[0].pool_index == 0


def test_double_label_agreement_no_adjudication(queue):
    _enqueue(queue, 1)
    a1 = queue.assign(0)
    a2 = queue.assign(0)
    assert a1 != a2
    queue.submit_label(0, a1, 1)
    item = queue.submit_label(0, a2, 1)
    assert not item.disagreement
    assert item.final_label == 1
    assert queue.needs_adjudication() == []


def test_disagreement_goes_to_adjudication(queue):
    _enqueue(queue, 1)
    a1 = queue.assign(0)
    a2 = queue.assign(0)
    queue.submit_label(0, a1, 0)
    item = queue.submit_label(0, a2, 2)
    assert item.disagreement
    assert item.final_label == 0  # first label stands until adjudicated
    assert queue.needs_adjudication() == [item]
    adjudicated = queue.adjudicate(0, 2)
    assert adjudicated.final_label == 2
    assert adjudicated.adjudicator == "senior-1"
    assert adjudicated.status == "adjudicated"
    assert queue.needs_adjudication() == []


def test_agreement_scores(queue):
    _enqueue(queue, 4)
    # Items 0,1 agree; items 2,3 disagree.
    for i in range(4):
        a1 = queue.assign(i)
        a2 = queue.assign(i)
        queue.submit_label(i, a1, 0)
        queue.submit_label(i, a2, 0 if i < 2 else 1)
    report = queue.agreement()
    assert report["n_double_labelled"] == 4
    assert report["percent_agreement"] == 0.5
    assert report["cohens_kappa"] == 0.0  # chance-level for this split


def test_cohens_kappa_perfect_and_chance():
    assert cohens_kappa([0, 1, 2, 0], [0, 1, 2, 0]) == 1.0
    assert cohens_kappa([0, 0, 0], [1, 1, 1]) == 0.0
    assert cohens_kappa([], []) == 0.0
    assert cohens_kappa([0, 1], [0, 1, 2]) == 0.0


def test_queue_rejects_bad_operations(queue):
    _enqueue(queue, 1)
    with pytest.raises(ValueError, match="already in queue"):
        queue.enqueue(0, [0.0] * 3)
    with pytest.raises(KeyError):
        queue.assign(99)
    queue.assign(0)
    with pytest.raises(ValueError, match="not assigned"):
        queue.submit_label(0, "ann-b", 1)
    with pytest.raises(ValueError, match="no disagreement"):
        queue.adjudicate(0, 1)


def test_summary_counts(queue):
    _enqueue(queue, 3)
    queue.assign(0)
    queue.submit_label(0, "ann-a", 1)
    summary = queue.summary()
    assert summary["n_items"] == 3
    assert summary["by_status"]["labelled"] == 1
    assert summary["n_disagreements"] == 0
