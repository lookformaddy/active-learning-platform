"""Tests for the fail-closed labelling budget."""
import pytest

from activelearn.budget import BudgetExceeded, LabelingBudget


def test_grant_and_remaining():
    budget = LabelingBudget(total_budget=50, per_round_cap=10)
    assert budget.request(10, round_no=1, purpose="margin") == 10
    assert budget.spent == 10
    assert budget.remaining == 40
    assert not budget.exhausted


def test_total_budget_is_fail_closed():
    budget = LabelingBudget(total_budget=15, per_round_cap=10)
    budget.request(10, round_no=1)
    with pytest.raises(BudgetExceeded, match="remaining budget 5"):
        budget.request(10, round_no=2)
    # The refused request left the ledger untouched in spend terms.
    assert budget.spent == 10
    assert budget.remaining == 5
    assert budget.summary()["n_refusals"] == 1


def test_per_round_cap_is_fail_closed():
    budget = LabelingBudget(total_budget=100, per_round_cap=10)
    with pytest.raises(BudgetExceeded, match="per-round cap 10"):
        budget.request(11, round_no=1)
    assert budget.spent == 0


def test_cost_cap_is_fail_closed():
    budget = LabelingBudget(total_budget=100, per_round_cap=50,
                            cost_per_label=2.0, cost_cap=30.0)
    budget.request(10, round_no=1)  # cost 20
    with pytest.raises(BudgetExceeded, match="cost cap 30.00"):
        budget.request(10, round_no=2)  # would reach cost 40
    assert budget.total_cost == 20.0


def test_exhausted_after_full_spend():
    budget = LabelingBudget(total_budget=10, per_round_cap=10)
    budget.request(10, round_no=1)
    assert budget.exhausted
    with pytest.raises(BudgetExceeded):
        budget.request(1, round_no=2)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        LabelingBudget(total_budget=0, per_round_cap=1)
    with pytest.raises(ValueError):
        LabelingBudget(total_budget=5, per_round_cap=10)
    with pytest.raises(ValueError):
        LabelingBudget(total_budget=10, per_round_cap=10).request(0, 1)


def test_ledger_records_grants_and_refusals():
    budget = LabelingBudget(total_budget=10, per_round_cap=5)
    budget.request(5, round_no=1, purpose="round-1")
    with pytest.raises(BudgetExceeded):
        budget.request(6, round_no=2)
    assert len(budget.ledger) == 2
    assert budget.ledger[0]["granted"] == 5
    assert budget.ledger[1]["refused_reason"] == "per_round_cap"
