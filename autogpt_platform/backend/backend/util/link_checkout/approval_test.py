import time

import pytest

from backend.util.link_checkout.approval import (
    ApprovalConflict,
    PendingApproval,
    approval_details,
    decide,
    open_approval,
    read_approval,
)

CHECKOUT = "c" * 32


def pending(**changes) -> PendingApproval:
    return PendingApproval.model_validate(
        {
            "checkout_id": CHECKOUT,
            "user_id": "owner",
            "session_id": "chat",
            "merchant_name": "Test store",
            "merchant_url": "https://shop.example/checkout",
            "context": "One paperback from the signed-in cart, as asked in this chat.",
            "amount": 100,
            "currency": "usd",
            "test_mode": True,
            "expires_at": time.time() + 60,
            **changes,
        }
    )


@pytest.mark.asyncio
async def test_approval_waits_for_the_customer_then_records_their_click(fake_redis):
    await open_approval(pending())
    assert (await read_approval(CHECKOUT)).state == "awaiting"

    view = await decide(
        CHECKOUT, "owner", "chat", approve=True, user_agent="Mozilla/5.0"
    )

    assert view is not None and view.state == "approved"
    details = approval_details(view)
    assert details.approval_method == "click"
    assert details.external_user_id == "owner"
    assert details.external_session_id == "chat"
    assert details.agent_log_id == CHECKOUT
    assert details.user_agent == "Mozilla/5.0"
    assert abs(details.approved_at - time.time()) < 5


@pytest.mark.asyncio
async def test_records_are_encrypted_at_rest(fake_redis):
    await open_approval(pending())
    await decide(CHECKOUT, "owner", "chat", approve=True, user_agent="Mozilla/5.0")
    stored = " ".join(fake_redis.values.values())
    assert "Test store" not in stored
    assert "Mozilla" not in stored


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id,session_id", [("mallory", "chat"), ("owner", "x")])
async def test_only_the_purchases_own_chat_can_decide(fake_redis, user_id, session_id):
    await open_approval(pending())
    assert (
        await decide(CHECKOUT, user_id, session_id, approve=True, user_agent=None)
        is None
    )
    assert (await read_approval(CHECKOUT)).state == "awaiting"


@pytest.mark.asyncio
async def test_a_decision_is_final(fake_redis):
    await open_approval(pending())
    await decide(CHECKOUT, "owner", "chat", approve=False, user_agent=None)

    again = await decide(CHECKOUT, "owner", "chat", approve=False, user_agent=None)
    assert again is not None and again.state == "declined"
    with pytest.raises(ApprovalConflict):
        await decide(CHECKOUT, "owner", "chat", approve=True, user_agent=None)
    with pytest.raises(ValueError):
        approval_details(again)


@pytest.mark.asyncio
async def test_an_expired_purchase_cannot_be_approved(fake_redis):
    await open_approval(pending(expires_at=time.time() - 1))
    assert (await read_approval(CHECKOUT)).state == "expired"
    with pytest.raises(ApprovalConflict):
        await decide(CHECKOUT, "owner", "chat", approve=True, user_agent=None)


@pytest.mark.asyncio
async def test_a_purchase_is_recorded_once(fake_redis):
    await open_approval(pending())
    with pytest.raises(RuntimeError):
        await open_approval(pending(amount=1))
    assert (await read_approval(CHECKOUT)).pending.amount == 100
