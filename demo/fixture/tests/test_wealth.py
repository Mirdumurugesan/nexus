import pytest

from wealth.goals import goal_progress, years_to_goal
from wealth.portfolio import Holding, allocation, rebalance_orders
from wealth.sip import future_value, monthly_rate, required_monthly_sip


def test_monthly_rate():
    assert monthly_rate(12) == pytest.approx(0.01)


def test_sip_future_value_10k_12pct_10y():
    # Standard annuity-due formula with i = 1% per month, n = 120 months.
    assert future_value(10_000, 12, 10) == pytest.approx(2_323_390.76, abs=1)


def test_sip_zero_rate():
    assert future_value(5_000, 0, 2) == 120_000


def test_required_sip_round_trips():
    sip = required_monthly_sip(1_000_000, 12, 10)
    assert future_value(sip, 12, 10) == pytest.approx(1_000_000, rel=1e-3)


def test_allocation_and_rebalance():
    h = [Holding("NIFTYBEES", "equity", 70), Holding("GILT", "debt", 30)]
    assert allocation(h)["equity"] == pytest.approx(0.7)
    orders = rebalance_orders(h, {"equity": 0.6, "debt": 0.4})
    assert orders["equity"] == pytest.approx(-10)
    assert orders["debt"] == pytest.approx(10)


def test_goals():
    assert goal_progress(50, 100) == 0.5
    assert years_to_goal(100, 40, 20) == 3
