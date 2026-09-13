"""OAuth PKCE and localhost callback tests."""

from __future__ import annotations

import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogpt_local_executor.auth import OAuthFlow
from autogpt_local_executor.config import ShimConfig


def _config(tmp_path: Path, *, callback_port: int = 41899) -> ShimConfig:
    return ShimConfig(
        platform_url="https://platform.example.com",
        oauth_redirect_port=callback_port,
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_authorize_url_uses_supported_scope_and_explicit_state(tmp_path: Path) -> None:
    flow = OAuthFlow(_config(tmp_path), token_store=object())
    url = flow._build_auth_url(
        "challenge",
        state="expected-state",
        redirect_port=41903,
    )
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)

    assert parsed.path == "/auth/authorize"
    assert query["scope"] == ["USE_TOOLS"]
    assert query["state"] == ["expected-state"]
    assert query["redirect_uri"] == ["http://localhost:41903/callback"]


def test_callback_validates_state_without_logging_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flow = OAuthFlow(_config(tmp_path, callback_port=_free_port()), token_store=object())
    server, codes, errors, port = flow._bind_callback_server(expected_state="right-state")

    def send_callback() -> None:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/callback?code=the-code&state=right-state",
            timeout=2,
        ):
            pass

    thread = threading.Thread(target=send_callback)
    thread.start()
    assert flow._wait_for_callback(server, codes, errors) == "the-code"
    thread.join(timeout=2)
    captured = capsys.readouterr()
    assert "the-code" not in captured.out
    assert "the-code" not in captured.err


def test_callback_rejects_state_mismatch(tmp_path: Path) -> None:
    flow = OAuthFlow(_config(tmp_path, callback_port=_free_port()), token_store=object())
    server, codes, errors, port = flow._bind_callback_server(expected_state="right-state")

    def send_callback() -> None:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/callback?code=the-code&state=wrong-state",
                timeout=2,
            )
        except urllib.error.HTTPError:
            pass

    thread = threading.Thread(target=send_callback)
    thread.start()
    with pytest.raises(ValueError, match="state did not match"):
        flow._wait_for_callback(server, codes, errors)
    thread.join(timeout=2)


def test_callback_uses_next_port_when_first_is_busy(tmp_path: Path) -> None:
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    first_port = blocker.getsockname()[1]
    flow = OAuthFlow(_config(tmp_path, callback_port=first_port), token_store=object())

    server, _codes, _errors, bound_port = flow._bind_callback_server(expected_state="state")
    try:
        assert bound_port > first_port
        assert bound_port <= first_port + 11
    finally:
        server.server_close()
        blocker.close()


@pytest.mark.asyncio
async def test_code_exchange_posts_form_to_backend_token_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"access_token": "access", "refresh_token": "refresh"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url: str, *, data: dict[str, str]):
            captured["url"] = url
            captured["data"] = data
            return Response()

    monkeypatch.setattr("autogpt_local_executor.auth.httpx.AsyncClient", Client)
    flow = OAuthFlow(_config(tmp_path), token_store=object())

    result = await flow._exchange_code("code", "verifier", redirect_port=41907)

    assert result["access_token"] == "access"
    assert captured["url"] == "https://platform.example.com/api/oauth/token"
    assert captured["data"] == {
        "grant_type": "authorization_code",
        "code": "code",
        "redirect_uri": "http://localhost:41907/callback",
        "client_id": "autogpt-local-executor",
        "code_verifier": "verifier",
    }


class _TokenStore:
    def __init__(self) -> None:
        self.cleared = False

    async def get_access_token(self) -> str:
        return "access-token"

    async def get_refresh_token(self) -> str:
        return "refresh-token"

    def clear_tokens(self) -> None:
        self.cleared = True


@pytest.mark.asyncio
async def test_revoke_posts_access_and_refresh_before_clearing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[str, dict[str, str]]] = []

    class Response:
        def raise_for_status(self) -> None:
            pass

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url: str, *, json: dict[str, str]):
            captured.append((url, json))
            return Response()

    monkeypatch.setattr("autogpt_local_executor.auth.httpx.AsyncClient", Client)
    store = _TokenStore()
    flow = OAuthFlow(_config(tmp_path), token_store=store)  # type: ignore[arg-type]

    await flow.revoke_tokens()

    assert store.cleared is True
    assert [body["token_type_hint"] for _, body in captured] == [
        "access_token",
        "refresh_token",
    ]
    assert all(url == "https://platform.example.com/api/oauth/revoke" for url, _ in captured)
    assert all(body["client_id"] == "autogpt-local-executor" for _, body in captured)
    assert all(body["client_secret"] == "" for _, body in captured)


@pytest.mark.asyncio
async def test_revoke_failure_preserves_local_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_count = 0

    class Response:
        def __init__(self, should_fail: bool) -> None:
            self.should_fail = should_fail

        def raise_for_status(self) -> None:
            if self.should_fail:
                raise RuntimeError("platform unavailable")

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url: str, *, json: dict[str, str]):
            nonlocal call_count
            call_count += 1
            return Response(should_fail=call_count == 2)

    monkeypatch.setattr("autogpt_local_executor.auth.httpx.AsyncClient", Client)
    store = _TokenStore()
    flow = OAuthFlow(_config(tmp_path), token_store=store)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="platform unavailable"):
        await flow.revoke_tokens()

    assert store.cleared is False


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["exchange", "refresh", "revoke"])
async def test_insecure_platform_is_rejected_before_sending_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    config = _config(tmp_path).model_copy(update={"platform_url": "http://platform.example.com"})
    client = AsyncMock()
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = client
    monkeypatch.setattr("autogpt_local_executor.auth.httpx.AsyncClient", factory)
    store = _TokenStore()
    flow = OAuthFlow(config, token_store=store)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="TLS"):
        if operation == "exchange":
            await flow._exchange_code("code", "verifier")
        elif operation == "refresh":
            await flow.refresh_token()
        else:
            await flow.revoke_tokens()

    client.post.assert_not_awaited()
    assert store.cleared is False
