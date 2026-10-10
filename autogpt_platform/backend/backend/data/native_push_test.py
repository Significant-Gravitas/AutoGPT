from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.api.model import NotificationPayload
from backend.data.native_push import (
    NativePushMessage,
    build_native_push,
    deliver_native_push,
)
from backend.data.native_push_subscription import NativePushSubscriptionDTO


def test_push_only_contains_a_generic_alert_and_route():
    message = build_native_push(
        NotificationPayload(
            type="copilot_completion",
            event="session_completed",
            session_id="a/b?c",
            text="private conversation",
            status="completed",
        )
    )
    assert message is not None
    assert message.path == "/home?sessionId=a%2Fb%3Fc"
    assert "private" not in message.model_dump_json()


def test_attention_opens_the_prompt_inbox_and_unknown_events_are_ignored():
    message = build_native_push(NotificationPayload(type="attention", event="approval"))
    assert message is not None
    assert message.path == "/mobile?tab=attention"
    assert (
        build_native_push(NotificationPayload(type="onboarding", event="step")) is None
    )


@pytest.mark.asyncio
async def test_invalid_token_is_removed_but_transient_failure_is_retained():
    sub = NativePushSubscriptionDTO(
        id="binding",
        provider="apns",
        token="a" * 64,
        environment="sandbox",
        origin="https://platform.agpt.co",
    )
    message = NativePushMessage(
        body="A chat needs your attention.", path="/mobile?tab=attention"
    )
    db = AsyncMock()
    with patch(
        "backend.data.native_push.get_database_manager_async_client", return_value=db
    ):
        with patch(
            "backend.data.native_push.send_apns",
            AsyncMock(return_value=httpx.Response(410)),
        ):
            await deliver_native_push(sub, message)
        db.delete_native_push_subscription.assert_awaited_once_with("binding")
        db.reset_mock()
        with patch(
            "backend.data.native_push.send_apns",
            AsyncMock(return_value=httpx.Response(503)),
        ):
            await deliver_native_push(sub, message)
        db.delete_native_push_subscription.assert_not_awaited()
