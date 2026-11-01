"""A trial user's subscription trial and spend counters, in place of the database and Redis.

Only the stored state is stubbed. Everything that reads or writes them runs as in
production: the executor's snapshot of which trial a turn is spent from, the
engines' budget lookups, the spend recording and the paywall.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import prisma.enums
import pytest
from pydantic import BaseModel

from backend.copilot import rate_limit, token_tracking, usage_activation
from backend.copilot.rate_limit import SubscriptionTier
from backend.data import db_accessors
from backend.data.pro_activation import UsageActivationState
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer

from .ledger import PlatformLedger

TRIAL_ID = "trial-1"
DAILY_LIMIT = 500_000
WEEKLY_LIMIT = 1_000_000
TOTAL_LIMIT = 1_000_000


class TrialCharge(BaseModel):
    turn: int
    microdollars: int
    trial_id: str | None


class TrialAccount:
    def __init__(self, user_id: str, *, spent_before: int) -> None:
        self.user_id = user_id
        self.turn = 0
        self.tier = SubscriptionTier.TRIAL
        self.trial = active_trial(user_id, spent=spent_before)
        self.charges: list[TrialCharge] = []
        self.counters = SpendCounters()

    def install(self, monkeypatch: pytest.MonkeyPatch, ledger: PlatformLedger) -> None:
        store = MagicMock(
            get_subscription_trial=AsyncMock(side_effect=self._read_trial),
            record_subscription_trial_cost=AsyncMock(side_effect=self._charge_trial),
        )
        monkeypatch.setattr(rate_limit, "credit_db", lambda: store)
        monkeypatch.setattr(db_accessors, "credit_db", lambda: store)
        monkeypatch.setattr(
            rate_limit, "_fetch_user_tier", AsyncMock(side_effect=self._read_tier)
        )
        usage = MagicMock(
            get_usage_activation_state=AsyncMock(side_effect=self._read_usage_state)
        )
        monkeypatch.setattr(usage_activation, "pro_activation_db", lambda: usage)
        monkeypatch.setattr(
            rate_limit, "get_redis_async", AsyncMock(return_value=self.counters)
        )
        # The flag service's overrides of the default limits; none are set.
        monkeypatch.setattr(
            rate_limit, "_fetch_cost_limits_flag", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            rate_limit, "_fetch_tier_multipliers_flag", AsyncMock(return_value=None)
        )

        # The ledger keeps each charge, then the real recording runs: that is
        # where a turn's spend is put on the trial.
        async def charge(user_id: str, cost_microdollars: int, **kwargs: Any) -> None:
            await ledger.record_spend(user_id, cost_microdollars, **kwargs)
            await rate_limit.record_cost_usage(user_id, cost_microdollars, **kwargs)

        monkeypatch.setattr(token_tracking, "record_cost_usage", charge)

    def charged_in(self, turn: int) -> list[TrialCharge]:
        return [charge for charge in self.charges if charge.turn == turn]

    def remaining_usd(self) -> float:
        """What the trial still allows today, as the budget lookups compute it.

        Worked out from this account's own numbers, not read through the
        production lookups, so the budget assertions check against an
        independent answer.
        """
        spent = self.counters.values
        daily = DAILY_LIMIT - spent.get(rate_limit._daily_key(self.user_id), 0)
        weekly = WEEKLY_LIMIT - spent.get(rate_limit._weekly_key(self.user_id), 0)
        total = TOTAL_LIMIT - self.trial.cost_microdollars
        return max(0, min(daily, weekly, total)) / 1_000_000

    def end_trial(self) -> None:
        self.trial = self.trial.model_copy(
            update={"ends_at": datetime.now(UTC) - timedelta(seconds=1)}
        )

    def convert_to_paid(self) -> None:
        self.tier = SubscriptionTier.PRO
        self.trial = self.trial.model_copy(
            update={"status": "converted", "converted_at": datetime.now(UTC)}
        )

    async def _read_trial(self, user_id: str) -> TrialState | None:
        assert user_id == self.user_id
        return self.trial

    async def _read_tier(self, user_id: str) -> SubscriptionTier:
        assert user_id == self.user_id
        return self.tier

    async def _read_usage_state(self, user_id: str) -> UsageActivationState:
        assert user_id == self.user_id
        return usage_state_of(user_id, self.tier, self.trial)

    async def _charge_trial(
        self, user_id: str, cost_microdollars: int, trial_id: str | None = None
    ) -> None:
        """``record_subscription_trial_cost``: it updates the rows that match, and
        an attributed charge that matches none is an error."""
        assert user_id == self.user_id
        if cost_microdollars <= 0:
            return
        matches = (
            self.trial.status == "trialing"
            if trial_id is None
            else self.trial.id == trial_id and self.trial.consumed_at is not None
        )
        if not matches:
            if trial_id is not None:
                raise ValueError("Trial cost attribution was not found for this user")
            return
        self.charges.append(
            TrialCharge(
                turn=self.turn, microdollars=cost_microdollars, trial_id=trial_id
            )
        )
        self.trial = self.trial.model_copy(
            update={
                "cost_microdollars": self.trial.cost_microdollars + cost_microdollars
            }
        )


class SpendCounters:
    """The Redis holding the daily and weekly spend counters.

    It answers the reads and the INCRBY + EXPIRE transactions the rate limiter
    issues, so the limiter's own Redis code runs; anything else it would call
    is missing and fails the test.
    """

    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key: str) -> bytes | None:
        value = self.values.get(key)
        return None if value is None else str(value).encode()

    def pipeline(self, transaction: bool) -> "_Transaction":
        assert transaction, "the spend counters are only written in a transaction"
        return _Transaction(self)


class _Transaction:
    def __init__(self, counters: SpendCounters) -> None:
        self._counters = counters
        self._queued: list[tuple[str, str, int]] = []

    def incrby(self, key: str, amount: int) -> None:
        self._queued.append(("incrby", key, amount))

    def expire(self, key: str, seconds: int) -> None:
        self._queued.append(("expire", key, seconds))

    async def execute(self) -> None:
        for command, key, value in self._queued:
            if command == "incrby":
                self._counters.values[key] = self._counters.values.get(key, 0) + value
            else:
                self._counters.ttls[key] = value
        self._queued.clear()


def usage_state_of(
    user_id: str, tier: SubscriptionTier, trial: TrialState | None
) -> UsageActivationState:
    """The usage state the database query reports for this tier and trial: the
    trial counts only while it is live, started and not converted."""
    live = (
        tier == SubscriptionTier.TRIAL
        and trial is not None
        and trial.active
        and trial.consumed_at is not None
        and trial.converted_at is None
    )
    return UsageActivationState(
        user_id=user_id,
        tier=prisma.enums.SubscriptionTier(tier.value),
        trial_id=trial.id if live and trial is not None else None,
        ready=True,
    )


def active_trial(
    user_id: str,
    *,
    spent: int = 0,
    daily_limit: int = DAILY_LIMIT,
    weekly_limit: int = WEEKLY_LIMIT,
    total_limit: int = TOTAL_LIMIT,
) -> TrialState:
    """A card-verified trial a week from its end, with these spend limits."""
    now = datetime.now(UTC)
    return TrialState(
        id=TRIAL_ID,
        user_id=user_id,
        customer_id="cus_1",
        offer=AcceptedTrialOffer(
            version="offer-v1",
            new_users_from=now - timedelta(days=1),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=daily_limit,
            weekly_cost_limit=weekly_limit,
            total_cost_limit=total_limit,
            onboarding_credit_amount=300,
            price_id="price_pro",
            unit_amount=2000,
            currency="usd",
        ),
        checkout_session_id="cs_1",
        subscription_id="sub_1",
        checkout_attempt=0,
        success_url="https://example.com",
        cancel_url="https://example.com",
        checkout_metadata={},
        status="trialing",
        card_verified_at=now,
        started_at=now,
        ends_at=now + timedelta(days=7),
        consumed_at=now,
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=spent,
    )
