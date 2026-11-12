"""Payment notices follow collection state, not a missing retry timestamp."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import NotificationType

from backend.notifications import lifecycle
from backend.notifications.lifecycle_test import _invoice, _run, _User


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["past_due", "unpaid", "canceled"])
async def test_missing_retry_date_uses_subscription_state(status):
    attempted_at = int(datetime(2026, 10, 24, tzinfo=timezone.utc).timestamp())
    with patch(
        "backend.notifications.lifecycle.stripe_call",
        AsyncMock(return_value={"status": status}),
    ):
        calls = await _run(
            lambda: lifecycle.on_payment_failed(
                _invoice(subscription="sub_1", next_payment_attempt=None),
                attempted_at=attempted_at,
            ),
            _User(),
        )
    if status == "canceled":
        calls["notify"].assert_not_awaited()
        return
    queued = calls["notify"].await_args.args[0]
    if status == "unpaid":
        assert queued.type == NotificationType.PAYMENT_FINAL_NOTICE
        assert queued.data.last_attempt_label == "24 Oct 2026"
        assert queued.data.pauses_label is None
    else:
        assert queued.type == NotificationType.PAYMENT_FAILED
        assert queued.data.next_retry_label is None


def test_missing_billing_name_uses_neutral_greeting():
    user = _User()
    user.name = " "
    assert lifecycle._greeting_name(user) == "there"
