"""Settling a card payment: a granted one is answered with its purchase, an
ungranted one says why."""

from unittest.mock import AsyncMock

import pytest

from backend.data import usage_reset_settlement as settlement
from backend.data.usage_reset import ResetPurchase
from backend.data.usage_reset_card import CardGrant
from backend.data.usage_reset_settlement import settle_card_reset

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


@pytest.fixture(autouse=True)
def pinned_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settlement, "now_seconds", lambda: NOW)


def _outcome(monkeypatch: pytest.MonkeyPatch, grant, ungranted=None) -> AsyncMock:
    outcome = AsyncMock(return_value=(grant, ungranted))
    monkeypatch.setattr(settlement, "card_reset_outcome", outcome)
    return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("granted_now", [True, False])
async def test_a_granted_payment_is_answered_with_its_purchase(
    monkeypatch: pytest.MonkeyPatch, granted_now: bool
):
    """Granted by this call or an earlier one: the webhook and the customer's
    return both settle, and both get the purchase."""
    outcome = _outcome(
        monkeypatch, CardGrant(purchase=PURCHASE, granted_now=granted_now)
    )

    settled = await settle_card_reset(SESSION, user_id="user-1")

    outcome.assert_awaited_once_with(SESSION, NOW, user_id="user-1")
    assert settled.purchase == PURCHASE and settled.ungranted is None


@pytest.mark.asyncio
@pytest.mark.parametrize("why", ["unknown_session", "not_paid", "plan_changed"])
async def test_an_ungranted_payment_says_why(monkeypatch: pytest.MonkeyPatch, why: str):
    _outcome(monkeypatch, None, why)

    settled = await settle_card_reset(SESSION)

    assert settled.purchase is None and settled.ungranted == why
