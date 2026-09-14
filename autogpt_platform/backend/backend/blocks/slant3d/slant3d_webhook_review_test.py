import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request
from pydantic import ValidationError

from backend.blocks.slant3d._api import TEST_CREDENTIALS
from backend.data import integrations
from backend.integrations.webhooks import _base, slant3d
from backend.integrations.webhooks.slant3d import Slant3DWebhooksManager
from backend.util.exceptions import (
    WebhookRegistrationError,
    WebhookSetupUnavailableError,
)


@pytest.mark.parametrize("event", [None, "", 123, {}])
async def test_invalid_event_type_is_a_validation_error(event):
    payload = {"platform_id": "platform-1", "event_type": event}
    if event is None:
        payload.pop("event_type")
    request = Request(
        {"type": "http", "headers": []},
        receive=AsyncMock(
            return_value={
                "type": "http.request",
                "body": json.dumps(payload).encode(),
                "more_body": False,
            }
        ),
    )
    webhook = integrations.Webhook.model_construct(
        resource="platform-1", config={"api_version": 2}
    )
    with pytest.raises(ValidationError, match="event_type"):
        await Slant3DWebhooksManager.validate_payload(webhook, request, None)


@pytest.mark.parametrize("second_user", ["user-1", "user-2"])
async def test_registration_serializes_the_external_platform(second_user):
    lock = asyncio.Lock()
    started = asyncio.Event()
    release = asyncio.Event()
    created = []
    lock_keys = []
    platform_url = ""

    class Mutex:
        def __init__(self, redis):
            self.owned = False

        async def acquire(self, key):
            lock_keys.append(key)
            await lock.acquire()
            self.owned = True

        async def release(self, key):
            if self.owned:
                self.owned = False
                lock.release()

    async def find(**kwargs):
        return next(
            (
                w
                for w in created
                if w.user_id == kwargs["user_id"]
                and w.credentials_id == kwargs["credentials_id"]
                and w.resource == kwargs["resource"]
                and w.organization_id == kwargs.get("organization_id")
                and w.team_id == kwargs.get("team_id")
                and set(kwargs.get("events") or []).issubset(w.events)
            ),
            None,
        )

    async def platform_request(method, resource, credentials, **kwargs):
        nonlocal platform_url
        if method == "GET":
            snapshot = {"data": {"webhookURL": platform_url}}
            started.set()
            await release.wait()
            return snapshot
        assert method == "PATCH"
        platform_url = kwargs["json"]["webhookURL"]
        return {"data": {}}

    async def create(webhook):
        created.append(webhook)
        return webhook

    with patch.object(
        slant3d, "AsyncRedisKeyedMutex", Mutex, create=True
    ), patch.object(
        slant3d, "get_redis_async", AsyncMock(return_value=object()), create=True
    ), patch.object(
        _base.app_config, "platform_base_url", "https://example.com"
    ), patch.object(
        integrations,
        "find_webhook_by_credentials_and_props",
        AsyncMock(side_effect=find),
    ), patch.object(
        integrations,
        "find_webhook_by_credentials_and_props_any_tenant",
        AsyncMock(side_effect=find),
    ), patch.object(
        integrations, "create_webhook", AsyncMock(side_effect=create)
    ), patch.object(
        Slant3DWebhooksManager,
        "_platform_request",
        AsyncMock(side_effect=platform_request),
    ) as api:
        first = asyncio.create_task(
            Slant3DWebhooksManager().get_suitable_auto_webhook(
                "user-1", TEST_CREDENTIALS, "orders", "platform-1", ["order.shipped"]
            )
        )
        await asyncio.wait_for(started.wait(), 2)
        second = asyncio.create_task(
            Slant3DWebhooksManager().get_suitable_auto_webhook(
                second_user, TEST_CREDENTIALS, "orders", "platform-1", ["order.shipped"]
            )
        )
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.wait_for(
            asyncio.gather(first, second, return_exceptions=True), 2
        )
    assert len(created) == 1
    assert sum(c.args[0] == "PATCH" for c in api.await_args_list) == 1
    assert len(lock_keys) == 2 and lock_keys[0] == lock_keys[1]
    if second_user == "user-1":
        assert results[0].id == results[1].id
    else:
        assert isinstance(results[1], WebhookRegistrationError)
    assert not lock.locked()


@pytest.mark.parametrize("failure", ["redis", "acquire"])
async def test_lock_failure_prevents_registration(failure):
    mutex = Mock(
        acquire=AsyncMock(
            side_effect=(
                RuntimeError("Redis unavailable") if failure == "acquire" else None
            )
        ),
        release=AsyncMock(),
    )
    with patch.object(
        slant3d,
        "get_redis_async",
        AsyncMock(
            side_effect=(
                RuntimeError("Redis unavailable") if failure == "redis" else None
            )
        ),
        create=True,
    ), patch.object(
        slant3d, "AsyncRedisKeyedMutex", return_value=mutex, create=True
    ), patch.object(
        _base.BaseWebhooksManager, "get_suitable_auto_webhook", AsyncMock()
    ) as register:
        with pytest.raises(WebhookSetupUnavailableError):
            await Slant3DWebhooksManager().get_suitable_auto_webhook(
                "user-1", TEST_CREDENTIALS, "orders", "platform-1", []
            )
    register.assert_not_awaited()
    if failure == "acquire":
        mutex.release.assert_awaited_once()


async def test_release_failure_preserves_successful_registration():
    webhook = integrations.Webhook.model_construct(id="webhook-1")
    mutex = Mock(
        acquire=AsyncMock(),
        release=AsyncMock(side_effect=RuntimeError("Redis offline")),
    )
    with patch.object(
        slant3d, "get_redis_async", AsyncMock(return_value=object())
    ), patch.object(slant3d, "AsyncRedisKeyedMutex", return_value=mutex), patch.object(
        _base.BaseWebhooksManager,
        "get_suitable_auto_webhook",
        AsyncMock(return_value=webhook),
    ):
        result = await Slant3DWebhooksManager().get_suitable_auto_webhook(
            "user-1", TEST_CREDENTIALS, "orders", "platform-1", []
        )
    assert result is webhook
    mutex.release.assert_awaited_once()
