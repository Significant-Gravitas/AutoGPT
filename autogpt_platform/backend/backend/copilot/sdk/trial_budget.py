from math import isfinite


def resolve_trial_sdk_budget(static_cap: float, remaining: float) -> float:
    if (
        not isfinite(static_cap)
        or not isfinite(remaining)
        or min(static_cap, remaining) <= 0
    ):
        raise ValueError("Trial budget is unavailable or exhausted; no turn can start")
    return min(static_cap, remaining)
