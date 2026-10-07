"""Systematic Investment Plan (SIP) maths."""


def monthly_rate(annual_rate_pct: float) -> float:
    """Convert an annual percentage rate into a monthly decimal rate."""
    return annual_rate_pct / 12 / 100


def future_value(monthly_amount: float, annual_rate_pct: float, years: int) -> float:
    """Future value of a monthly SIP with contributions at the start of each month."""
    r = annual_rate_pct / 100
    n = years * 12
    if r == 0:
        return round(monthly_amount * n, 2)
    return round(monthly_amount * ((1 + r) ** n - 1) / r * (1 + r), 2)


def required_monthly_sip(goal: float, annual_rate_pct: float, years: int) -> float:
    """Monthly SIP needed to reach `goal` in `years`."""
    per_rupee = future_value(1.0, annual_rate_pct, years)
    return round(goal / per_rupee, 2)
