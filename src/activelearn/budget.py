"""Labelling-budget management with fail-closed caps.

Labelling is the expensive resource in active learning, so the budget
is a hard control, not a suggestion: every request goes through
:meth:`LabelingBudget.request`, which either grants the full batch or
raises :class:`BudgetExceeded` — partial grants never happen, and the
run stops instead of silently overspending. The ledger records every
grant and every refused request so an audit can reconstruct exactly
where the budget went.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Raised when a labelling request would exceed a hard cap."""


@dataclass
class LabelingBudget:
    """Hard caps on labelling spend.

    - ``total_budget`` — maximum labels for the whole run.
    - ``per_round_cap`` — maximum labels in a single query round.
    - ``cost_per_label`` — accounting unit cost (default 1.0); the run
      stops when ``total_cost`` would exceed ``total_budget`` in cost
      terms too, when a ``cost_cap`` is set.
    - ``cost_cap`` — optional hard cap on total spend in cost units.
    """

    total_budget: int
    per_round_cap: int
    cost_per_label: float = 1.0
    cost_cap: float | None = None
    spent: int = 0
    total_cost: float = 0.0
    ledger: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_budget < 1:
            raise ValueError("total_budget must be >= 1")
        if self.per_round_cap < 1:
            raise ValueError("per_round_cap must be >= 1")
        if self.per_round_cap > self.total_budget:
            raise ValueError("per_round_cap cannot exceed total_budget")

    @property
    def remaining(self) -> int:
        return self.total_budget - self.spent

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def request(self, n_labels: int, round_no: int,
                purpose: str = "") -> int:
        """Grant a labelling batch or raise — never a partial grant.

        The check order is deliberate: per-round cap first (a runaway
        round is a configuration bug), then the total budget, then the
        cost cap. Every refusal is recorded in the ledger.
        """
        if n_labels < 1:
            raise ValueError("n_labels must be >= 1")
        if n_labels > self.per_round_cap:
            self.ledger.append({"round": round_no, "requested": n_labels,
                                "granted": 0, "purpose": purpose,
                                "refused_reason": "per_round_cap"})
            raise BudgetExceeded(
                f"round {round_no}: requested {n_labels} labels exceeds "
                f"per-round cap {self.per_round_cap}")
        if n_labels > self.remaining:
            self.ledger.append({"round": round_no, "requested": n_labels,
                                "granted": 0, "purpose": purpose,
                                "refused_reason": "total_budget"})
            raise BudgetExceeded(
                f"round {round_no}: requested {n_labels} labels exceeds "
                f"remaining budget {self.remaining}")
        cost = n_labels * self.cost_per_label
        if self.cost_cap is not None and self.total_cost + cost > self.cost_cap:
            self.ledger.append({"round": round_no, "requested": n_labels,
                                "granted": 0, "purpose": purpose,
                                "refused_reason": "cost_cap"})
            raise BudgetExceeded(
                f"round {round_no}: cost {self.total_cost + cost:.2f} "
                f"would exceed cost cap {self.cost_cap:.2f}")
        self.spent += n_labels
        self.total_cost += cost
        self.ledger.append({"round": round_no, "requested": n_labels,
                            "granted": n_labels, "purpose": purpose,
                            "refused_reason": None})
        return n_labels

    def summary(self) -> dict:
        return {"total_budget": self.total_budget,
                "spent": self.spent,
                "remaining": self.remaining,
                "total_cost": round(self.total_cost, 2),
                "cost_cap": self.cost_cap,
                "n_requests": len(self.ledger),
                "n_refusals": sum(1 for e in self.ledger
                                  if e["refused_reason"] is not None)}
