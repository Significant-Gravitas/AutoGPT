import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.local_executor import websocket as ws_module


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
