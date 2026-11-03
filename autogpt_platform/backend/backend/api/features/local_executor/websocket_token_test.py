import asyncio
import contextlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.local_executor import websocket as ws_module
from backend.copilot.tools.local_pc_relay import RedisShimRelay
from backend.copilot.tools.local_pc_relay_test_support import FakeRedis, FakeWebSocket
from backend.copilot.tools.local_pc_shim import ShimConnectionManager


def _websocket() -> MagicMock:
    websocket = MagicMock()
    websocket.headers = {"authorization": "Bearer agpt_xt_token"}
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture(autouse=True)
def _quick_rechecks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ws_module, "_TOKEN_RECHECK_SECONDS", 0.01)


@pytest.mark.asyncio
async def test_a_revoked_token_closes_the_connection_before_it_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The revocation push can miss a connection (Redis down, another
    worker); rechecking the token still ends it, without waiting an hour."""
    introspect = AsyncMock(return_value=SimpleNamespace(active=False))
    monkeypatch.setattr(ws_module, "introspect_token", introspect)
    websocket = _websocket()

    await ws_module._close_when_token_ends(websocket, int(time.time()) + 3600)

    websocket.close.assert_awaited_once_with(code=4428, reason="Token revoked")
    assert introspect.await_args.args[0] == "agpt_xt_token"


@pytest.mark.asyncio
async def test_an_expired_token_still_closes_as_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    introspect = AsyncMock()
    monkeypatch.setattr(ws_module, "introspect_token", introspect)
    websocket = _websocket()

    await ws_module._close_when_token_ends(websocket, int(time.time()) - 1)

    websocket.close.assert_awaited_once_with(code=4401, reason="Access token expired")
    introspect.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_recheck_keeps_the_connection_until_a_later_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lookup that fails says nothing about the token, so it must not close
    a live connection; the next check that sees the revocation does."""
    introspect = AsyncMock(
        side_effect=[
            RuntimeError("database unavailable"),
            SimpleNamespace(active=True),
            SimpleNamespace(active=False),
        ]
    )
    monkeypatch.setattr(ws_module, "introspect_token", introspect)
    websocket = _websocket()

    await ws_module._close_when_token_ends(websocket, int(time.time()) + 3600)

    assert introspect.await_count == 3
    websocket.close.assert_awaited_once_with(code=4428, reason="Token revoked")


@pytest.mark.asyncio
async def test_a_recheck_request_checks_the_token_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A revocation push asks for a check now, not at the next periodic one."""
    monkeypatch.setattr(ws_module, "_TOKEN_RECHECK_SECONDS", 3600)
    introspect = AsyncMock(return_value=SimpleNamespace(active=False))
    monkeypatch.setattr(ws_module, "introspect_token", introspect)
    websocket = _websocket()
    recheck = asyncio.Event()
    watcher = asyncio.create_task(
        ws_module._close_when_token_ends(websocket, int(time.time()) + 7200, recheck)
    )
    await asyncio.sleep(0)

    recheck.set()
    await asyncio.wait_for(watcher, 5)

    websocket.close.assert_awaited_once_with(code=4428, reason="Token revoked")


@pytest.mark.asyncio
async def test_a_recheck_keeps_a_connection_whose_token_is_still_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws_module, "_TOKEN_RECHECK_SECONDS", 3600)
    checked = asyncio.Event()

    async def introspect(*_args: object, **_kwargs: object) -> SimpleNamespace:
        checked.set()
        return SimpleNamespace(active=True)

    monkeypatch.setattr(ws_module, "introspect_token", introspect)
    websocket = _websocket()
    recheck = asyncio.Event()
    watcher = asyncio.create_task(
        ws_module._close_when_token_ends(websocket, int(time.time()) + 7200, recheck)
    )
    try:
        recheck.set()
        await asyncio.wait_for(checked.wait(), 5)
        await asyncio.sleep(0)

        assert not watcher.done()
        websocket.close.assert_not_awaited()
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher


class _SignedInWebSocket(FakeWebSocket):
    """A companion connection authenticated with its own access token."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self.headers = {"authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_revoking_one_computers_token_leaves_the_others_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each computer signs in on its own. Revoking one computer's token closes
    that computer's connection with 4428, and the user's other computer, whose
    token is still active, stays connected."""
    monkeypatch.setattr(ws_module, "_TOKEN_RECHECK_SECONDS", 3600)
    revoked = {"token-a"}
    introspected: set[str] = set()
    other_checked = asyncio.Event()

    async def introspect(token: str, **_kwargs: object) -> SimpleNamespace:
        introspected.add(token)
        if token == "token-b":
            other_checked.set()
        return SimpleNamespace(active=token not in revoked)

    monkeypatch.setattr(ws_module, "introspect_token", introspect)
    relay = RedisShimRelay(FakeRedis())
    manager = ShimConnectionManager(relay=relay)
    computers: dict[str, tuple[_SignedInWebSocket, asyncio.Task, asyncio.Task]] = {}
    for name in ("a", "b"):
        websocket = _SignedInWebSocket(f"token-{name}")
        recheck = asyncio.Event()
        serve = asyncio.create_task(
            relay.serve_websocket(
                f"session-{name}",
                websocket,
                hello={"machine_id": f"machine-{name}"},
                user_id="user-1",
                client_id="autogpt-local-executor",
                on_token_recheck=recheck.set,
            )
        )
        watch = asyncio.create_task(
            ws_module._close_when_token_ends(
                websocket, int(time.time()) + 7200, recheck
            )
        )
        computers[name] = (websocket, serve, watch)
        await relay.wait_for_presence(f"session-{name}", timeout=1)

    revoked_ws, revoked_serve, revoked_watch = computers["a"]
    other_ws, other_serve, other_watch = computers["b"]
    try:
        assert await manager.push_token_revocation("user-1", "autogpt-local-executor")

        await asyncio.wait_for(revoked_watch, 5)
        await asyncio.wait_for(revoked_serve, 5)
        assert revoked_ws.close_code == 4428

        await asyncio.wait_for(other_checked.wait(), 5)
        await asyncio.sleep(0.05)
        assert introspected == {"token-a", "token-b"}
        assert not other_ws.closed.is_set()
        assert not other_serve.done() and not other_watch.done()
        # The relay handles the recheck itself; no companion receives it.
        assert revoked_ws.outbound.empty() and other_ws.outbound.empty()
    finally:
        other_watch.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await other_watch
        await other_ws.close()
        await asyncio.wait_for(other_serve, 5)
