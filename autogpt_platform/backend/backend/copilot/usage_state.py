"""Whether the account can start a turn now, and which limit decides that.

The usage meters say how full each window is. This is the verdict the client
keys its banners on, worked out once on the server from the same status:
``exhausted`` when a turn would be refused now, ``low`` from the point the
meters turn amber, ``ok`` otherwise -- with the limit behind it and when that
limit refills.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

UsageState = Literal["ok", "low", "exhausted"]
LimitWindow = Literal["daily", "weekly", "trial"]

# Where the usage meters turn amber.
LOW_PERCENT_USED = 80.0


class WindowReading(BaseModel):
    """One capped window as the meters show it."""

    window: Literal["daily", "weekly"]
    percent_used: float
    exhausted: bool
    resets_at: datetime
    # False for a window that allows nothing (no plan, or the trial is over):
    # its reset brings no budget back.
    refills: bool = True


class UsageVerdict(BaseModel):
    state: UsageState = "ok"
    limiting_window: LimitWindow | None = None
    # None for a trial: its allowance doesn't refill.
    resets_at: datetime | None = None


def usage_verdict(
    readings: list[WindowReading], *, trial: bool, trial_spent: bool
) -> UsageVerdict:
    """The state of the account's allowance, from its capped windows.

    When several windows are used up, the one that refills last is the one
    the account waits for. A trial's weekly window stands for its allowance,
    so it is named as the trial, with no reset.
    """
    if trial_spent:
        return UsageVerdict(state="exhausted", limiting_window="trial")
    exhausted = [reading for reading in readings if reading.exhausted]
    if exhausted:
        return _verdict("exhausted", max(exhausted, key=_refill_order), trial)
    if not readings:
        return UsageVerdict()
    tightest = max(readings, key=lambda r: (r.percent_used, _refill_order(r)))
    if tightest.percent_used >= LOW_PERCENT_USED:
        return _verdict("low", tightest, trial)
    return UsageVerdict()


def _refill_order(reading: WindowReading) -> tuple[bool, datetime, bool]:
    # A window that never refills outlasts any reset; the weekly window wins
    # a tie on time, being the larger of the two.
    return not reading.refills, reading.resets_at, reading.window == "weekly"


def _verdict(state: UsageState, reading: WindowReading, trial: bool) -> UsageVerdict:
    if trial and reading.window == "weekly":
        return UsageVerdict(state=state, limiting_window="trial")
    return UsageVerdict(
        state=state,
        limiting_window=reading.window,
        resets_at=reading.resets_at if reading.refills else None,
    )
