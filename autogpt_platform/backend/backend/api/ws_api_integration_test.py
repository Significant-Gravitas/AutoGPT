"""The WebSocket router's subscribe path over the live Redis cluster, with no
Redis mocks: what a client is told when Redis doesn't confirm a subscribe in
time, and that sending the request it names again subscribes it. Skips when
the cluster is unreachable."""

import asyncio
import json
from contextlib import aclosing
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import WebSocket, WebSocketDisconnect

import backend.data.redis_client as redis_client
from backend.api import conn_manager
from backend.api.conn_manager import (
    ConnectionManager,
    _notification_bus_channel,
    event_bus_channel,
)
from backend.api.conn_manager_integration_test import _graph_event_payload
from backend.api.ws_api import websocket_router
from backend.data.execution import graph_all_channel
from backend.data.sharded_pubsub import ShardedSubscription
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="local redis cluster not reachable; skip ws_api integration",
)

# Only a step that never comes keeps a wait going this long.
_HANG_SECONDS = 10.0
_EVENTS = {"graph_execution_event", "node_execution_event", "notification"}


class _Browser:
    """One socket's client as the router sees it: what it sends, and every
    message the server sent it, in order."""

    def __init__(self) -> None:
        self.websocket = AsyncMock(spec=WebSocket)
        self.received: list[dict[str, Any]] = []
        self._outgoing: asyncio.Queue[str | None] = asyncio.Queue()
        self._arrived = asyncio.Condition()
        self._replies_read = 0
        self.websocket.receive_text = AsyncMock(side_effect=self._next_outgoing)
        self.websocket.send_text = AsyncMock(side_effect=self._receive)

    def send(self, method: str, data: Any) -> None:
        self._outgoing.put_nowait(json.dumps({"method": method, "data": data}))

    def leave(self) -> None:
        self._outgoing.put_nowait(None)

    async def next_reply(self) -> dict[str, Any]:
        """The server's answer to the oldest request not read yet."""
        async with self._arrived:
            await self._arrived.wait_for(
                lambda: len(self._replies()) > self._replies_read
            )
            reply = self._replies()[self._replies_read]
            self._replies_read += 1
            return reply

    async def event(self, marker: str) -> None:
        async with self._arrived:
            await self._arrived.wait_for(lambda: marker in self.markers())

    def markers(self) -> list[str]:
        return [
            message["data"]["inputs"]["x"]
            for message in self.received
            if message["method"] == "graph_execution_event"
        ]

    def _replies(self) -> list[dict[str, Any]]:
        return [m for m in self.received if m["method"] not in _EVENTS]

    async def _next_outgoing(self) -> str:
        message = await self._outgoing.get()
        if message is None:
            raise WebSocketDisconnect()
        return message

    async def _receive(self, text: str) -> None:
        async with self._arrived:
            self.received.append(json.loads(text))
            self._arrived.notify_all()


def _runs_channel(user_id: str, graph_id: str) -> str:
    return event_bus_channel(graph_all_channel(user_id, graph_id))


def _graphs_on_the_notification_shard(user_id: str, count: int) -> list[str]:
    """Graphs whose runs channel shares a shard with the user's notification
    stream, which the socket opens on connect: Redis's confirmation of a
    subscribe to one is read by that stream's pump."""
    shard = redis_client.resolve_shard_for_channel(_notification_bus_channel(user_id))
    graphs: list[str] = []
    # A shard owns about a third of CI's slots, so a few tries each suffice.
    for _ in range(1000):
        graph_id = f"graph-{uuid4().hex[:8]}"
        channel = _runs_channel(user_id, graph_id)
        if redis_client.resolve_shard_for_channel(channel) == shard:
            graphs.append(graph_id)
        if len(graphs) == count:
            return graphs
    raise AssertionError(f"no {count} graphs on the shard {shard}")


def _hold_confirmation(
    monkeypatch: pytest.MonkeyPatch, channel: str
) -> tuple[asyncio.Event, asyncio.Event, asyncio.Event]:
    """Hold Redis's confirmation of ``channel``'s SSUBSCRIBE where the stream
    reads it, as a slow connection to the shard would: nothing behind it is
    read until it is released. Returns the event set once it is held, the
    event that releases it, and one set once the stream reads Redis's
    confirmation of a SUNSUBSCRIBE of the channel."""
    held, release, unsubscribed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    read = ShardedSubscription.messages

    async def messages(self: ShardedSubscription):
        async with aclosing(read(self)) as stream:
            async for message in stream:
                if message.get("channel") == channel:
                    if message["type"] == "ssubscribe":
                        held.set()
                        await release.wait()
                    elif message["type"] == "sunsubscribe":
                        unsubscribed.set()
                yield message

    monkeypatch.setattr(ShardedSubscription, "messages", messages)
    return held, release, unsubscribed


def _deadlines_on_demand(monkeypatch: pytest.MonkeyPatch) -> list[asyncio.Timeout]:
    """Make each subscribe's deadline one that never passes by itself, so the
    test expires it, through the real expiry path, once it is ready."""
    deadlines: list[asyncio.Timeout] = []

    def subscribe_deadline() -> asyncio.Timeout:
        deadlines.append(asyncio.timeout(None))
        return deadlines[-1]

    monkeypatch.setattr(conn_manager, "_subscribe_deadline", subscribe_deadline)
    return deadlines


def _serve(
    browser: _Browser, user_id: str, monkeypatch: pytest.MonkeyPatch
) -> asyncio.Task:
    """The real router and ConnectionManager, past authentication and the
    per-message account checks, which read the database."""
    monkeypatch.setattr(
        "backend.api.ws_api.authenticate_websocket", AsyncMock(return_value=user_id)
    )
    monkeypatch.setattr(
        "backend.api.ws_api.is_user_banned", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        "backend.api.ws_api.is_user_account_closed", AsyncMock(return_value=False)
    )
    return asyncio.create_task(websocket_router(browser.websocket, ConnectionManager()))


@pytest.mark.asyncio
async def test_a_subscribe_redis_did_not_confirm_in_time_names_itself_to_be_sent_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#16922: a subscribe whose confirmation timed out is answered with an
    error that names the request and says it may be sent again. Until the
    client sends it again, the channel's events never reach it; then they do."""
    real_deadline = conn_manager._subscribe_deadline
    user_id = f"user-ws-int-{uuid4().hex[:8]}"
    redis_client.get_redis.cache_clear()
    cluster = redis_client.get_redis()
    steady, flaky = _graphs_on_the_notification_shard(user_id, 2)
    held, release, taken_back = _hold_confirmation(
        monkeypatch, _runs_channel(user_id, flaky)
    )

    def publish(graph_id: str, marker: str) -> int:
        """The number of subscriptions Redis delivered it to."""
        payload = _graph_event_payload(
            user_id=user_id, graph_id=graph_id, graph_exec_id="e", marker=marker
        )
        return cluster.spublish(_runs_channel(user_id, graph_id), payload.decode())

    browser = _Browser()
    router = _serve(browser, user_id, monkeypatch)
    try:
        async with asyncio.timeout(_HANG_SECONDS):
            browser.send("subscribe_graph_executions", {"graph_id": steady})
            assert (await browser.next_reply())["success"] is True
            deadlines = _deadlines_on_demand(monkeypatch)
            browser.send("subscribe_graph_executions", {"graph_id": flaky})
            # Redis has confirmed it, but the stream hasn't read that yet:
            # the bound passes now, however slow the runner.
            await held.wait()
            [deadline] = deadlines
            deadline.reschedule(asyncio.get_running_loop().time())
            failure = await browser.next_reply()
        monkeypatch.setattr(conn_manager, "_subscribe_deadline", real_deadline)
        assert failure["method"] == "error"
        assert failure["success"] is False
        release.set()
        # Redis applied the SSUBSCRIBE, so the server takes it back.
        async with asyncio.timeout(_HANG_SECONDS):
            await taken_back.wait()

        assert publish(flaky, "missed") == 0
        # One shard, one connection, published in this order: once the steady
        # graph's event is in, the flaky graph's would have come before it.
        publish(steady, "after-the-failure")
        async with asyncio.timeout(_HANG_SECONDS):
            await browser.event("after-the-failure")
        assert browser.markers() == ["after-the-failure"]

        assert failure.get("request") == {
            "method": "subscribe_graph_executions",
            "data": {"graph_id": flaky},
        }, failure
        assert failure.get("retryable") is True, failure

        async with asyncio.timeout(_HANG_SECONDS):
            browser.send(failure["request"]["method"], failure["request"]["data"])
            assert (await browser.next_reply())["success"] is True
        publish(flaky, "after-the-retry")
        publish(steady, "steady-again")
        async with asyncio.timeout(_HANG_SECONDS):
            await browser.event("steady-again")
        assert browser.markers() == [
            "after-the-failure",
            "after-the-retry",
            "steady-again",
        ]
    finally:
        release.set()
        browser.leave()
        async with asyncio.timeout(_HANG_SECONDS):
            await router
        redis_client.disconnect()
