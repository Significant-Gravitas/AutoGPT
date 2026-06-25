"""Shared CoPilot budget thresholds."""

# Minimum viable per-query budget (USD). A turn dispatched with less than this
# is almost certainly doomed: the median task cost is ~$5.37 (p50), and even a
# lightweight follow-up needs about $1 of headroom.
MIN_VIABLE_TASK_BUDGET_USD = 1.0
