"""Retrain triggers: when is a new model version worth training?

Retraining on every new label is wasteful; never retraining lets the
model go stale. Three independent triggers, any one of which fires a
retrain — the reason is always recorded so the retrain history is
auditable:

- ``batch_size`` — enough new labels have arrived since the last
  retrain (the workhorse trigger).
- ``uncertainty_threshold`` — the mean predictive entropy on the
  remaining pool has risen above a threshold (the world got harder,
  or the model is degrading).
- ``drift_threshold`` — the feature centroid of newly labelled items
  has shifted away from the training centroid (the incoming data
  distribution moved).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrainState:
    labels_since_retrain: int = 0
    mean_pool_uncertainty: float = 0.0
    centroid_shift: float = 0.0
    rounds_since_retrain: int = 0


@dataclass
class RetrainPolicy:
    batch_size: int = 20
    uncertainty_threshold: float | None = None
    drift_threshold: float | None = None
    max_rounds_between_retrains: int | None = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def should_retrain(self, state: RetrainState, round_no: int) -> tuple[bool, list[str]]:
        """Evaluate all triggers; returns (fire, [reasons])."""
        reasons: list[str] = []
        if state.labels_since_retrain >= self.batch_size:
            reasons.append(
                f"batch_size: {state.labels_since_retrain} new labels "
                f">= {self.batch_size}")
        if (self.uncertainty_threshold is not None
                and state.mean_pool_uncertainty >= self.uncertainty_threshold):
            reasons.append(
                f"uncertainty: mean pool entropy "
                f"{state.mean_pool_uncertainty:.3f} >= "
                f"{self.uncertainty_threshold:.3f}")
        if (self.drift_threshold is not None
                and state.centroid_shift >= self.drift_threshold):
            reasons.append(
                f"drift: centroid shift {state.centroid_shift:.3f} >= "
                f"{self.drift_threshold:.3f}")
        if (self.max_rounds_between_retrains is not None
                and state.rounds_since_retrain >= self.max_rounds_between_retrains):
            reasons.append(
                f"staleness: {state.rounds_since_retrain} rounds since "
                f"last retrain >= {self.max_rounds_between_retrains}")
        fired = bool(reasons)
        if fired:
            self.history.append({"round": round_no, "reasons": reasons,
                                 "labels_since_retrain":
                                     state.labels_since_retrain})
        return fired, reasons

    def summary(self) -> dict:
        return {"batch_size": self.batch_size,
                "uncertainty_threshold": self.uncertainty_threshold,
                "drift_threshold": self.drift_threshold,
                "n_retrains": len(self.history),
                "history": self.history}
