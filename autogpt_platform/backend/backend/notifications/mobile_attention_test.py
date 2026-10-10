from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.notifications import mobile_attention


@pytest.mark.asyncio
async def test_duplicate_reviews_do_not_repeatedly_notify_and_failures_release_claim():
    redis = AsyncMock()
    redis.set.return_value = False
    config = SimpleNamespace(
        apns_private_key_path="/mounted/key", fcm_service_account_path=""
    )
    with (
        patch.object(
            mobile_attention, "Settings", return_value=SimpleNamespace(config=config)
        ),
        patch.object(
            mobile_attention, "get_redis_async", AsyncMock(return_value=redis)
        ),
        patch.object(
            mobile_attention.AsyncRedisNotificationEventBus, "publish", AsyncMock()
        ) as publish,
    ):
        await mobile_attention.notify_attention("user", "review-1", "chat-1")
        publish.assert_not_awaited()
        redis.set.return_value = True
        await mobile_attention.notify_attention("user", "review-1", "chat-1")
        assert publish.await_args.args[0].payload.session_id == "chat-1"
        publish.side_effect = RuntimeError("offline")
        await mobile_attention.notify_attention("user", "review-2", "chat-2")
        redis.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_native_providers_means_no_new_notification_side_effects():
    config = SimpleNamespace(apns_private_key_path="", fcm_service_account_path="")
    with (
        patch.object(
            mobile_attention, "Settings", return_value=SimpleNamespace(config=config)
        ),
        patch.object(mobile_attention, "get_redis_async", AsyncMock()) as redis,
    ):
        await mobile_attention.notify_attention("user", "question", "chat")
        redis.assert_not_awaited()
