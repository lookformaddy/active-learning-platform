"""Human-in-the-loop annotation queue with adjudication.

The queue is the contract between the model and the annotators:

- Items are enqueued by the query strategy and assigned to annotators
  (round-robin by default; an annotator never receives the same item
  twice).
- Every label is recorded with its annotator, so agreement can be
  audited per annotator and per item.
- Items labelled by two annotators who disagree go to adjudication:
  a senior annotator's label is final and the disagreement is kept in
  the record — it is signal (ambiguous region), not noise to delete.
- :func:`cohens_kappa` scores inter-annotator agreement over the
  double-labelled items; a low kappa is a trigger to retrain the
  annotation guidelines, not to silently drop the hard items.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnnotationItem:
    pool_index: int
    features: list[float]
    model_prediction: int | None = None
    model_confidence: float | None = None
    status: str = "queued"  # queued | assigned | labelled | adjudicated
    labels: dict[str, int] = field(default_factory=dict)
    assignees: list[str] = field(default_factory=list)
    final_label: int | None = None
    adjudicator: str | None = None
    disagreement: bool = False


class AnnotationQueue:
    """Annotation work queue with double-labelling + adjudication."""

    def __init__(self, annotators: list[str],
                 senior_annotators: list[str],
                 double_label_fraction: float = 0.2) -> None:
        if not annotators:
            raise ValueError("at least one annotator is required")
        if not senior_annotators:
            raise ValueError("at least one senior annotator is required")
        if not 0.0 <= double_label_fraction <= 1.0:
            raise ValueError("double_label_fraction must be in [0, 1]")
        self.annotators = list(annotators)
        self.senior_annotators = list(senior_annotators)
        self.double_label_fraction = double_label_fraction
        self.items: dict[int, AnnotationItem] = {}
        self._next_annotator = 0
        self._next_senior = 0

    def enqueue(self, pool_index: int, features: list[float],
                model_prediction: int | None = None,
                model_confidence: float | None = None) -> AnnotationItem:
        if pool_index in self.items:
            raise ValueError(f"pool item {pool_index} already in queue")
        item = AnnotationItem(pool_index=pool_index,
                              features=list(features),
                              model_prediction=model_prediction,
                              model_confidence=model_confidence)
        self.items[pool_index] = item
        return item

    def _assign_next(self, item: AnnotationItem) -> str:
        # Round-robin, but never assign the same annotator twice to one
        # item (double-labelling needs two different humans).
        for _ in range(len(self.annotators)):
            name = self.annotators[
                self._next_annotator % len(self.annotators)]
            self._next_annotator += 1
            if name not in item.assignees:
                item.assignees.append(name)
                item.status = "assigned"
                return name
        raise ValueError(
            f"no unassigned annotator left for item {item.pool_index}")

    def assign(self, pool_index: int) -> str:
        item = self._get(pool_index)
        if item.status == "adjudicated":
            raise ValueError(
                f"item {pool_index} is adjudicated, cannot assign")
        if len(item.assignees) >= 2:
            raise ValueError(f"item {pool_index} already has two annotators")
        return self._assign_next(item)

    def submit_label(self, pool_index: int, annotator: str,
                     label: int) -> AnnotationItem:
        item = self._get(pool_index)
        if annotator not in item.assignees:
            raise ValueError(
                f"{annotator} is not assigned to item {pool_index}")
        if annotator in item.labels:
            raise ValueError(
                f"{annotator} already labelled item {pool_index}")
        item.labels[annotator] = label
        if len(item.labels) == 1:
            item.status = "labelled"
            item.final_label = label
        elif len(item.labels) == 2:
            values = list(item.labels.values())
            if values[0] == values[1]:
                item.final_label = values[0]
                item.status = "labelled"
            else:
                item.disagreement = True
                item.status = "labelled"  # pending adjudication
        return item

    def needs_adjudication(self) -> list[AnnotationItem]:
        return [it for it in self.items.values() if it.disagreement
                and it.adjudicator is None]

    def adjudicate(self, pool_index: int, label: int,
                   adjudicator: str | None = None) -> AnnotationItem:
        item = self._get(pool_index)
        if not item.disagreement:
            raise ValueError(
                f"item {pool_index} has no disagreement to adjudicate")
        if item.adjudicator is not None:
            raise ValueError(f"item {pool_index} already adjudicated")
        senior = adjudicator or self.senior_annotators[
            self._next_senior % len(self.senior_annotators)]
        self._next_senior += 1
        item.adjudicator = senior
        item.final_label = label
        item.status = "adjudicated"
        return item

    def labelled(self) -> list[AnnotationItem]:
        """Items with a final label (single, agreed, or adjudicated)."""
        return [it for it in self.items.values()
                if it.final_label is not None]

    def double_labelled_pairs(self) -> list[tuple[int, int]]:
        """(label_a, label_b) for every item labelled by two annotators."""
        pairs = []
        for it in self.items.values():
            if len(it.labels) == 2:
                values = list(it.labels.values())
                pairs.append((values[0], values[1]))
        return pairs

    def agreement(self) -> dict:
        """Percent agreement + Cohen's kappa over double-labelled items."""
        pairs = self.double_labelled_pairs()
        if not pairs:
            return {"n_double_labelled": 0, "percent_agreement": None,
                    "cohens_kappa": None}
        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        kappa = cohens_kappa([a for a, _ in pairs],
                             [b for _, b in pairs])
        return {"n_double_labelled": len(pairs),
                "percent_agreement": round(agree, 4),
                "cohens_kappa": round(kappa, 4)}

    def summary(self) -> dict:
        by_status: dict[str, int] = {}
        for it in self.items.values():
            by_status[it.status] = by_status.get(it.status, 0) + 1
        return {"n_items": len(self.items), "by_status": by_status,
                "n_disagreements": sum(1 for it in self.items.values()
                                       if it.disagreement),
                "n_adjudicated": sum(1 for it in self.items.values()
                                     if it.adjudicator is not None),
                "agreement": self.agreement()}

    def _get(self, pool_index: int) -> AnnotationItem:
        try:
            return self.items[pool_index]
        except KeyError:
            raise KeyError(f"pool item {pool_index} not in queue") from None


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two annotators' label lists.

    1.0 is perfect agreement, 0.0 is chance-level. Returns 0.0 when
    there is nothing to compare or chance agreement is 1.0.
    """
    if len(a) != len(b) or not a:
        return 0.0
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = 0.0
    for label in labels:
        pa = sum(1 for x in a if x == label) / n
        pb = sum(1 for y in b if y == label) / n
        expected += pa * pb
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)
