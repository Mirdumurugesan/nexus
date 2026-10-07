"""Goal tracking for financial plans."""


def goal_progress(saved: float, goal: float) -> float:
    """Fraction of a goal already funded, capped at 1.0."""
    if goal <= 0:
        raise ValueError("goal must be positive")
    return min(saved / goal, 1.0)


def years_to_goal(goal: float, saved: float, yearly_saving: float) -> int:
    """Whole years of saving (no growth) needed to close the gap."""
    gap = max(goal - saved, 0.0)
    if gap == 0:
        return 0
    if yearly_saving <= 0:
        raise ValueError("yearly_saving must be positive when there is a gap")
    years = int(gap // yearly_saving)
    return years if gap % yearly_saving == 0 else years + 1
