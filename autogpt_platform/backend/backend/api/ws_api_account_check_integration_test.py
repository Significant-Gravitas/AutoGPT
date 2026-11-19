"""The WebSocket router's per-message account check over the live Redis
cluster, with only the database faked: while one user's lookup waits on a
Redis shard that stopped answering, every other socket on the process is
still answered. Skips when the cluster is unreachable."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import redis.asyncio.cluster as async_cluster
from fastapi import WebSocket, WebSocketDisconnect

import backend.data.redis_client as redis_client
from backend.api import ws_api
from backend.api.conn_manager import ConnectionManager
from backend.api.ws_api import websocket_router
from backend.util import cache as cache_module
from backend.util.cache_test_helpers import cache_key, hold_cache_read
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
    answer the server sent it, in order."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.websocket = AsyncMock(spec=WebSocket)
        self._outgoing: asyncio.Queue[str | None] = asyncio.Queue()
        self._answers: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.websocket.receive_text = AsyncMock(side_effect=self._next_outgoing)
        self.websocket.send_text = AsyncMock(
            side_effect=lambda text: self._receive(json.loads(text))
        )
        self.websocket.send_json = AsyncMock(side_effect=self._receive)

    def heartbeat(self) -> None:
        self._outgoing.put_nowait(json.dumps({"method": "heartbeat", "data": "ping"}))

    def leave(self) -> None:
        self._outgoing.put_nowait(None)

    async def next_answer(self) -> dict[str, Any]:
        return await self._answers.get()

    async def _next_outgoing(self) -> str:
        message = await self._outgoing.get()
        if message is None:
            raise WebSocketDisconnect()
        return message

    async def _receive(self, message: dict[str, Any]) -> None:
        if message["method"] not in _EVENTS:
            self._answers.put_nowait(message)


def _serve(
    browsers: list[_Browser], monkeypatch: pytest.MonkeyPatch
) -> list[asyncio.Task]:
    """The real router and ConnectionManager, past authentication. The
    calling test's ``real_ban_lookup`` fixture provides the real cached
    account lookups; their database reads find open accounts."""
    users = {id(browser.websocket): browser.user_id for browser in browsers}

    async def authenticate(websocket: WebSocket) -> str:
        return users[id(websocket)]

    db = MagicMock()
    db.is_auth_user_banned = AsyncMock(return_value=False)
    db.is_account_closed = AsyncMock(return_value=False)
    monkeypatch.setattr(ws_api, "authenticate_websocket", authenticate)
    monkeypatch.setattr("backend.data.user.user_db", lambda: db)
    manager = ConnectionManager()
    return [
        asyncio.create_task(websocket_router(browser.websocket, manager))
        for browser in browsers
    ]


@pytest.mark.asyncio
async def test_other_sockets_are_answered_while_redis_holds_one_users_account_check(
    monkeypatch: pytest.MonkeyPatch, real_ban_lookup: None
) -> None:
    """#17064: the account check before each message reads the cache in
    Redis. A shard that stops answering made that read block the event loop,
    so no socket on the process got an answer until Redis did."""
    held_user, other = _Browser(f"user-held-{uuid4().hex[:8]}"), _Browser(
        f"user-other-{uuid4().hex[:8]}"
    )
    routers = _serve([held_user, other], monkeypatch)
    held = None
    try:
        async with asyncio.timeout(_HANG_SECONDS):
            for browser in (held_user, other):
                browser.heartbeat()
                assert (await browser.next_answer())["data"] == "pong"

        # Held until released below, so neither bound may end the wait first.
        monkeypatch.setattr(cache_module, "REDIS_OP_TIMEOUT_S", 3 * _HANG_SECONDS)
        monkeypatch.setattr(ws_api, "ACCOUNT_CHECK_TIMEOUT_S", 3 * _HANG_SECONDS)
        held = hold_cache_read(
            monkeypatch,
            cache_key("is_user_banned", user_id=held_user.user_id),
            _HANG_SECONDS,
        )
        held_user.heartbeat()
        async with asyncio.timeout(2 * _HANG_SECONDS):
            await held.started()
            other.heartbeat()
            assert (await other.next_answer())["data"] == "pong"
        assert not held.answered

        held.release()
        async with asyncio.timeout(_HANG_SECONDS):
            assert (await held_user.next_answer())["data"] == "pong"
    finally:
        if held is not None:
            held.release()
        for browser in (held_user, other):
            browser.leave()
        async with asyncio.timeout(2 * _HANG_SECONDS):
            await asyncio.gather(*routers, return_exceptions=True)


@pytest.mark.asyncio
async def test_each_message_reads_the_ban_and_the_closure_once(
    monkeypatch: pytest.MonkeyPatch, real_ban_lookup: None
) -> None:
    """A Redis shard that stops answering costs each of the check's cache
    reads the cache's bound, so it reads each answer once per message."""
    browser = _Browser(f"user-reads-{uuid4().hex[:8]}")
    routers = _serve([browser], monkeypatch)
    answers = {
        cache_key("is_user_banned", user_id=browser.user_id): "ban",
        cache_key("is_user_account_closed", user_id=browser.user_id): "closure",
    }
    reads: list[str] = []
    execute = async_cluster.RedisCluster.execute_command

    async def recorded_execute(self, *args, **kwargs):
        command, key = args[0], args[1] if len(args) > 1 else None
        key = key.decode() if isinstance(key, bytes) else key
        if command in ("GET", "GETEX") and key in answers:
            reads.append(answers[key])
        return await execute(self, *args, **kwargs)

    try:
        async with asyncio.timeout(_HANG_SECONDS):
            # The first message stores both answers.
            browser.heartbeat()
            assert (await browser.next_answer())["data"] == "pong"
            monkeypatch.setattr(
                async_cluster.RedisCluster, "execute_command", recorded_execute
            )
            browser.heartbeat()
            assert (await browser.next_answer())["data"] == "pong"
        assert reads == ["ban", "closure"]
    finally:
        browser.leave()
        async with asyncio.timeout(2 * _HANG_SECONDS):
            await asyncio.gather(*routers, return_exceptions=True)
