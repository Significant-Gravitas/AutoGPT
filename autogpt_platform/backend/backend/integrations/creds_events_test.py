"""Tests for the credentials-changed bus — the cross-process half of invalidation."""

import contextlib
import logging
import socket
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
async def test_a_dead_redis_does_not_fail_the_credential_write():
    """A write is durable before it is announced, so a broadcast failure must
    stay non-fatal — loudly, since the fallback is then the 60 s TTL."""
    manager = IntegrationCredentialsManager()
    manager.store = MagicMock()
    manager.store.update_creds = AsyncMock()
    manager._locked = lambda *args, **kwargs: contextlib.nullcontext()

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    bus_logger = logging.getLogger("backend.data.event_bus")
    bus_logger.addHandler(handler)
    try:
        with patch(
            "backend.data.event_bus.redis.get_redis_async",
            side_effect=ConnectionError("redis down"),
        ):
            await manager.update(_USER, _oauth_creds())
    finally:
        bus_logger.removeHandler(handler)

    manager.store.update_creds.assert_awaited_once()
    assert [r for r in records if r.levelno >= logging.ERROR and r.exc_info]


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


@pytest.mark.parametrize(
    "module", ["backend.api.rest_api", "backend.copilot.executor.manager"]
)
def test_every_process_holding_the_cache_broadcasts_its_writes(module: str):
    """The regression: the API server and the copilot executor each hold their
    own copy of the token cache, so the in-process hook only ever evicts the
    writer's copy and the OAuth callback could not reach the executor's."""
    result = subprocess.run(
        [sys.executable, "-c", _PROCESS_PROBE.replace("__MODULE__", module)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "cache=True hook=True published=('user-1', 'github')" in result.stdout


def _has_live_cluster() -> bool:
    """Probe the socket, not ``redis_client.connect()``: this runs at collection
    time and that helper retries a dead host for ~45 minutes before giving up."""
    from backend.data import redis_client

    try:
        socket.create_connection((redis_client.HOST, redis_client.PORT), 1).close()
    except OSError:
        return False
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


_PROCESS_PROBE = """
import asyncio
import sys
from unittest.mock import AsyncMock, patch

import __MODULE__  # noqa: F401 - the process under test loads this
import backend.integrations.creds_manager as cm

with patch.object(cm, "publish_creds_changed", new_callable=AsyncMock) as publish:
    asyncio.run(cm._invoke_creds_changed_hook("user-1", "github"))

cached = "backend.copilot.integration_creds" in sys.modules
print(
    "cache=" + str(cached)
    + " hook=" + str(cm._on_creds_changed is not None)
    + " published=" + str(publish.await_args[0])
)
"""
