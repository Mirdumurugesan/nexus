"""Portfolio allocation helpers."""
from dataclasses import dataclass


@dataclass
class Holding:
    symbol: str
    asset_class: str  # "equity" | "debt" | "gold"
    value: float


def allocation(holdings: list[Holding]) -> dict[str, float]:
    """Share of total value per asset class, as fractions that sum to 1."""
    total = sum(h.value for h in holdings)
    if total == 0:
        return {}
    out: dict[str, float] = {}
    for h in holdings:
        out[h.asset_class] = out.get(h.asset_class, 0.0) + h.value / total
    return out


def rebalance_orders(holdings: list[Holding], target: dict[str, float]) -> dict[str, float]:
    """Amount to buy (+) or sell (-) per asset class to hit the target mix."""
    total = sum(h.value for h in holdings)
    current = allocation(holdings)
    return {
        cls: round((target.get(cls, 0.0) - current.get(cls, 0.0)) * total, 2)
        for cls in set(current) | set(target)
    }
