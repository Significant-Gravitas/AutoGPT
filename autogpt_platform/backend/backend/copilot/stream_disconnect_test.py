"""The cluster-wide broadcast behind ``DELETE /sessions/{id}/stream``."""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from backend.copilot.stream_disconnect import (
    LISTENER_DISCONNECT_CHANNEL,
    ListenerDisconnect,
    ListenerDisconnects,
)


@pytest_asyncio.fixture(scope="session", loop_scope="session", name="server")
async def _server_noop() -> None:
    return None


@pytest_asyncio.fixture(
    scope="session", loop_scope="session", autouse=True, name="graph_cleanup"
)
async def _graph_cleanup_noop():
    yield


def _bus(*requests: ListenerDisconnect) -> MagicMock:
    async def listen(_channel: str) -> AsyncGenerator[ListenerDisconnect, None]:
        for request in requests:
            yield request
        await asyncio.Event().wait()

    return MagicMock(listen_events=listen, publish_event=AsyncMock())


@pytest.mark.asyncio
async def test_every_request_cancels_this_pods_listeners():
    cancel_local = AsyncMock(return_value=1)
    request = ListenerDisconnect(session_id="sess-1", sent_at=123.0)
    disconnects = ListenerDisconnects(cancel_local, bus=_bus(request))

    disconnects.ensure_subscribed()
    try:
        for _ in range(50):
            if cancel_local.await_count:
                break
            await asyncio.sleep(0.01)
    finally:
        await disconnects.close()

    cancel_local.assert_awaited_once_with("sess-1", 123.0)


@pytest.mark.asyncio
async def test_broadcast_publishes_on_the_shared_channel():
    bus = _bus()
    disconnects = ListenerDisconnects(AsyncMock(), bus=bus)

    await disconnects.broadcast("sess-1")

    request, channel = bus.publish_event.await_args.args
    assert channel == LISTENER_DISCONNECT_CHANNEL
    assert request.session_id == "sess-1"


@pytest.mark.asyncio
async def test_a_failed_subscription_is_retried():
    cancel_local = AsyncMock(return_value=0)
    calls = 0

    async def listen(_channel: str) -> AsyncGenerator[ListenerDisconnect, None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("shard moved")
        yield ListenerDisconnect(session_id="sess-1", sent_at=1.0)
        await asyncio.Event().wait()

    disconnects = ListenerDisconnects(
        cancel_local, bus=MagicMock(listen_events=listen), retry_delay_s=0.01
    )
    disconnects.ensure_subscribed()
    try:
        for _ in range(100):
            if cancel_local.await_count:
                break
            await asyncio.sleep(0.01)
    finally:
        await disconnects.close()

    assert calls == 2
    cancel_local.assert_awaited_once()


@pytest.mark.asyncio
async def test_subscribing_twice_keeps_one_subscription():
    disconnects = ListenerDisconnects(AsyncMock(), bus=_bus())
    disconnects.ensure_subscribed()
    first = disconnects._task
    disconnects.ensure_subscribed()
    try:
        assert disconnects._task is first
    finally:
        await disconnects.close()
