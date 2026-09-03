"""Tests for the credentials-changed bus — the cross-process half of invalidation."""

import contextlib
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.creds_events import (
    CREDS_CHANGED_CHANNEL,
    CredentialsChangedEvent,
    listen_creds_changed,
    publish_creds_changed,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager

_USER = "user-creds-events-test"


@pytest.mark.asyncio
async def test_publish_spublishes_on_the_broadcast_channel():
    cluster = MagicMock()
    cluster.execute_command = AsyncMock()

    with patch("backend.data.event_bus.redis.get_redis_async", return_value=cluster):
        await publish_creds_changed(_USER, "github")

    command, channel, message = cluster.execute_command.await_args[0]
    assert command == "SPUBLISH"
    assert channel == f"creds_changed/{CREDS_CHANGED_CHANNEL}"
    assert _USER in message and "github" in message


@pytest.mark.asyncio
async def test_credential_write_publishes_creds_changed():
    """Every write through the manager announces itself, hook or no hook."""
    manager = IntegrationCredentialsManager()
    manager.store = MagicMock()
    manager.store.update_creds = AsyncMock()
    manager._locked = lambda *args, **kwargs: contextlib.nullcontext()

    with patch(
        "backend.integrations.creds_manager.publish_creds_changed",
        new_callable=AsyncMock,
    ) as publish:
        await manager.update(_USER, _oauth_creds())

    publish.assert_awaited_once_with(_USER, "github")


def test_write_publishes_in_a_process_that_has_no_in_process_hook():
    """The regression: the OAuth callback runs in the API process, which never
    imports the copilot cache, so its in-process hook is None."""
    result = subprocess.run(
        [sys.executable, "-c", _API_PROCESS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "hook=None published=('user-1', 'github')" in result.stdout


def _has_live_cluster() -> bool:
    from backend.data import redis_client

    try:
        client = redis_client.connect()
    except Exception:
        return False
    with contextlib.suppress(Exception):
        client.close()
    return True


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip creds-changed round trip",
)
async def test_publish_reaches_a_subscriber_in_another_event_loop():
    import asyncio

    from backend.data import redis_client

    redis_client.get_redis.cache_clear()
    redis_client._async_clients.clear()

    received: list[CredentialsChangedEvent] = []

    async def consume() -> None:
        async for event in listen_creds_changed():
            received.append(event)
            return

    task = asyncio.create_task(consume())
    # Let SSUBSCRIBE settle; races drop the publish otherwise.
    await asyncio.sleep(0.3)
    try:
        await publish_creds_changed(_USER, "github")
        await asyncio.wait_for(task, timeout=5.0)
    finally:
        if not task.done():
            task.cancel()
        await redis_client.disconnect_async()

    assert received and received[0].user_id == _USER


def _oauth_creds(token: str = "tok") -> OAuth2Credentials:
    return OAuth2Credentials(
        id="creds-oauth2",
        provider="github",
        title="Test OAuth",
        access_token=SecretStr(token),
        refresh_token=SecretStr("test-refresh"),
        access_token_expires_at=None,
        refresh_token_expires_at=None,
        scopes=[],
    )


_API_PROCESS_SCRIPT = """
import asyncio
from unittest.mock import AsyncMock, patch

import backend.api.features.integrations.router  # noqa: F401 - API process import set
import backend.integrations.creds_manager as cm

with patch.object(cm, "publish_creds_changed", new_callable=AsyncMock) as publish:
    asyncio.run(cm._invoke_creds_changed_hook("user-1", "github"))

print(f"hook={cm._on_creds_changed} published={publish.await_args[0]}")
"""
