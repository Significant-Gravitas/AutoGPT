"""A trial user's subscription trial and spend counters, in place of the database and Redis.

Only the rows are stubbed. Everything that reads or writes them runs as in
production: the executor's snapshot of which trial a turn is spent from, the
engines' budget lookups, the spend recording and the paywall.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from backend.copilot import rate_limit, token_tracking
from backend.copilot.rate_limit import SubscriptionTier
from backend.data import db_accessors
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
        self.trial = _trial(user_id, spent=spent_before)
        self.charges: list[TrialCharge] = []
        self._counters: dict[str, int] = {}

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
        redis = MagicMock(get=AsyncMock(side_effect=self._read_counter))
        monkeypatch.setattr(
            rate_limit, "get_redis_async", AsyncMock(return_value=redis)
        )
        monkeypatch.setattr(
            rate_limit, "_incr_counter_atomic", AsyncMock(side_effect=self._increment)
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
            await ledger._record_spend(user_id, cost_microdollars, **kwargs)
            await rate_limit.record_cost_usage(user_id, cost_microdollars, **kwargs)

        monkeypatch.setattr(token_tracking, "record_cost_usage", charge)

    def charged_in(self, turn: int) -> list[TrialCharge]:
        return [charge for charge in self.charges if charge.turn == turn]

    def remaining_usd(self) -> float:
        """What the trial still allows today, as the budget lookups compute it."""
        daily = DAILY_LIMIT - self._counters.get(rate_limit._daily_key(self.user_id), 0)
        weekly = WEEKLY_LIMIT - self._counters.get(
            rate_limit._weekly_key(self.user_id), 0
        )
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

    async def _charge_trial(
        self, user_id: str, cost_microdollars: int, trial_id: str | None = None
    ) -> None:
        """``record_subscription_trial_cost``: the same rows its update matches."""
        assert user_id == self.user_id
        self.charges.append(
            TrialCharge(
                turn=self.turn, microdollars=cost_microdollars, trial_id=trial_id
            )
        )
        matches = (
            self.trial.status == "trialing"
            if trial_id is None
            else self.trial.id == trial_id and self.trial.consumed_at is not None
        )
        if matches:
            self.trial = self.trial.model_copy(
                update={
                    "cost_microdollars": self.trial.cost_microdollars
                    + cost_microdollars
                }
            )

    async def _read_counter(self, key: str) -> int | None:
        return self._counters.get(key)

    async def _increment(self, _redis: object, key: str, delta: int, _ttl: int) -> None:
        self._counters[key] = self._counters.get(key, 0) + delta


def _trial(user_id: str, *, spent: int) -> TrialState:
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
            daily_cost_limit=DAILY_LIMIT,
            weekly_cost_limit=WEEKLY_LIMIT,
            total_cost_limit=TOTAL_LIMIT,
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
