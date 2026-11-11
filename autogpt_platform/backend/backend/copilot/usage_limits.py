"""Per-plan usage limits, adjustable from PostHog without a deploy.

The ``usage-plan-limits`` flag is a JSON object keyed by plan::

    {"PRO": {"daily": 3200000, "weekly": 8000000, "reserve": 250000,
             "extra_week_credits": 1250}}

Amounts are microdollars of AI cost (1 USD = 1,000,000); the extra week's
price is credits (1 credit = 1 cent). The defaults below are the approved
launch numbers, so nothing depends on the flag being set or reachable. A
plan's entry that fails validation is dropped whole and that plan keeps its
default; the other plans are unaffected. A plan with neither an entry nor a
default keeps the base-times-multiplier limits it has always had.
"""

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from backend.util.cache import cached
from backend.util.feature_flag import Flag, get_feature_flag_value

logger = logging.getLogger(__name__)

# The most days a weekly window can span (7 days 18 hours in a 31-day month).
MAX_WINDOW_DAYS = 8
# Plans the flag may set. NO_TIER and TRIAL are never read from it, so a typo
# cannot open the paywall or change a trial's accepted allowance.
CONFIGURABLE_PLANS = frozenset({"BASIC", "PRO", "MAX", "BUSINESS", "ENTERPRISE"})


class PlanLimits(BaseModel):
    """One plan's limits. ``reserve`` is the part of each day's spend that
    doesn't count against the week; it is inside the weekly limit."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    # The lower bound catches dollars typed where microdollars are meant.
    daily: int = Field(ge=10_000, le=10_000_000_000)
    weekly: int = Field(ge=10_000, le=10_000_000_000)
    reserve: int = Field(default=0, ge=0)
    # 0: this plan can't buy an extra week.
    extra_week_credits: int = Field(default=0, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def _fits_together(self) -> "PlanLimits":
        if self.daily > self.weekly:
            raise ValueError("daily is above weekly")
        if self.reserve > self.daily:
            raise ValueError("reserve is above daily")
        if self.reserve * MAX_WINDOW_DAYS > self.weekly:
            raise ValueError("a window's reserves would exceed weekly")
        if 0 < self.extra_week_credits < 100:
            raise ValueError("extra_week_credits is under one dollar")
        return self


# Max's daily limit is its weekly limit divided by 3, to the microdollar: the
# flat-out week in SECRT-3441 (22.67, 22.67, 13.67, then the reserve) only adds
# up to exactly $68 that way.
DEFAULT_PLAN_LIMITS: dict[str, PlanLimits] = {
    "PRO": PlanLimits(
        daily=3_200_000, weekly=8_000_000, reserve=250_000, extra_week_credits=1_250
    ),
    "MAX": PlanLimits(
        daily=22_666_667,
        weekly=68_000_000,
        reserve=2_000_000,
        extra_week_credits=8_000,
    ),
}


async def get_plan_limits() -> dict[str, PlanLimits]:
    """Every plan's own limits: the defaults, with valid flag entries on top."""
    try:
        override = await _fetch_plan_limits_flag()
    except Exception:
        logger.warning("Plan limits flag lookup failed; using defaults", exc_info=True)
        override = None
    return {**DEFAULT_PLAN_LIMITS, **(override or {})}


async def plan_limits_for(tier: str) -> PlanLimits | None:
    """The plan's own limits, or None for one on the base-times-multiplier
    formula (and for no plan or a trial)."""
    return (await get_plan_limits()).get(tier)


def parse_plan_limits(raw: Any) -> dict[str, PlanLimits]:
    """The valid plan entries of a flag payload; an invalid one is dropped."""
    if not isinstance(raw, dict):
        logger.error(f"usage-plan-limits is not a JSON object: {raw!r}")
        return {}
    parsed: dict[str, PlanLimits] = {}
    for plan, entry in raw.items():
        if plan not in CONFIGURABLE_PLANS:
            logger.warning(f"usage-plan-limits: ignoring entry for {plan!r}")
            continue
        try:
            parsed[plan] = PlanLimits.model_validate(entry)
        except ValidationError as error:
            logger.error(
                f"usage-plan-limits[{plan}] is invalid, keeping that plan's "
                f"default: {error.errors(include_url=False, include_input=False)}"
            )
    return parsed


@cached(ttl_seconds=60, maxsize=1, cache_none=False)
async def _fetch_plan_limits_flag() -> dict[str, PlanLimits] | None:
    raw = await get_feature_flag_value(Flag.USAGE_PLAN_LIMITS.value, "system", None)
    # None: no answer. False: the flag is off. Either way, the defaults.
    if raw is None or raw is False:
        return None
    return parse_plan_limits(raw) or None
