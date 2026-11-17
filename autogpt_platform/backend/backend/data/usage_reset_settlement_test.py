"""Settling a card payment: the purchase is counted once, by whichever call
granted it, and an ungranted payment says why."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import usage_reset_settlement as settlement
from backend.data.usage_reset import ResetPurchase
from backend.data.usage_reset_card import CardGrant
from backend.data.usage_reset_settlement import settle_card_reset
from backend.util.posthog_events import PostHogEvent

NOW = 1_793_824_256
SESSION = "cs_test_a1b2c3"
PURCHASE = ResetPurchase(
    id="reset-1",
    user_id="user-1",
    tier="MAX",
    price_credits=8_000,
    paid_with="card",
    status="granted",
    week_start=NOW - 86_400,
    week_end=NOW + 86_400,
    sequence=1,
    granted_at=NOW,
)


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    capture = MagicMock()
    monkeypatch.setattr(settlement.posthog_client, "capture", capture)
    monkeypatch.setattr(settlement, "now_seconds", lambda: NOW)
    return capture


def _outcome(monkeypatch: pytest.MonkeyPatch, grant, ungranted=None) -> AsyncMock:
    outcome = AsyncMock(return_value=(grant, ungranted))
    monkeypatch.setattr(settlement, "card_reset_outcome", outcome)
    return outcome


@pytest.mark.asyncio
async def test_the_call_that_grants_counts_the_purchase(
    events: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    outcome = _outcome(monkeypatch, CardGrant(purchase=PURCHASE, granted_now=True))

    settled = await settle_card_reset(SESSION, user_id="user-1")

    outcome.assert_awaited_once_with(SESSION, NOW, user_id="user-1")
    assert settled.purchase == PURCHASE and settled.ungranted is None
    events.assert_called_once_with(
        "user-1",
        PostHogEvent.USAGE_RESET_PURCHASED,
        {"tier": "MAX", "price_credits": 8_000, "paid_with": "card"},
    )


@pytest.mark.asyncio
async def test_a_grant_already_made_is_not_counted_again(
    events: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    """The webhook and the customer's return both settle; one of them granted."""
    _outcome(monkeypatch, CardGrant(purchase=PURCHASE, granted_now=False))

    settled = await settle_card_reset(SESSION)

    assert settled.purchase == PURCHASE
    events.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("why", ["unknown_session", "not_paid", "plan_changed"])
async def test_an_ungranted_payment_says_why(
    events: MagicMock, monkeypatch: pytest.MonkeyPatch, why: str
):
    _outcome(monkeypatch, None, why)

    settled = await settle_card_reset(SESSION)

    assert settled.purchase is None and settled.ungranted == why
    events.assert_not_called()
