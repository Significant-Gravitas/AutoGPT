"""OAuth PKCE and localhost callback tests."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogpt_local_executor.auth import KeychainTokenStore, OAuthFlow
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


@pytest.fixture
def isolated_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    passwords: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(
        "autogpt_local_executor.auth.keyring.get_password",
        lambda service, user: passwords.get((service, user)),
    )
    monkeypatch.setattr(
        "autogpt_local_executor.auth.keyring.set_password",
        lambda service, user, value: passwords.__setitem__((service, user), value),
    )
    monkeypatch.setattr(
        "autogpt_local_executor.auth.keyring.delete_password",
        lambda service, user: passwords.pop((service, user), None),
    )
    return passwords


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"platform_url": "https://other.example.com"},
        {"platform_url": "https://platform.example.com/backend"},
        {"platform_ws_url": "wss://other.example.com/ws/local-executor"},
        {"platform_ws_url": "wss://platform.example.com/untrusted-path"},
        {"platform_oauth_url": "https://other.example.com/auth"},
        {"platform_oauth_url": "https://platform.example.com/other-auth"},
        {"oauth_client_id": "another-client"},
    ],
)
async def test_keychain_tokens_are_isolated_by_deployment_and_destinations(
    tmp_path: Path, isolated_keyring: dict, overrides: dict[str, str]
) -> None:
    config = _config(tmp_path)
    original = KeychainTokenStore(config)
    await original.store_tokens("original-access", "original-refresh")
    other = KeychainTokenStore(config.model_copy(update=overrides))

    assert await other.get_access_token() is None
    assert await other.get_refresh_token() is None
    await other.store_tokens("other-access", "other-refresh")
    other.clear_tokens()

    assert await original.get_access_token() == "original-access"
    assert await original.get_refresh_token() == "original-refresh"


@pytest.mark.asyncio
async def test_equivalent_endpoints_share_tokens_without_losing_backend_prefix(
    tmp_path: Path, isolated_keyring: dict
) -> None:
    original = _config(tmp_path).model_copy(
        update={
            "platform_url": "https://api.example.com/backend",
            "platform_oauth_url": "https://app.example.com/auth",
        }
    )
    store = KeychainTokenStore(original)
    await store.store_tokens("access", "refresh")
    equivalent = original.model_copy(
        update={
            "platform_url": "wss://API.EXAMPLE.COM:443/backend/",
            "platform_oauth_url": "https://APP.EXAMPLE.COM:443/auth/",
            "platform_ws_url": "wss://API.EXAMPLE.COM:443/backend/ws/local-executor/",
            "session_id": "new-child-session",
        }
    )

    assert await KeychainTokenStore(equivalent).get_access_token() == "access"
    assert await KeychainTokenStore(equivalent).get_refresh_token() == "refresh"
    different_prefix = original.model_copy(update={"platform_url": "https://api.example.com"})
    assert await KeychainTokenStore(different_prefix).get_access_token() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["access", "refresh", "revoke"])
async def test_legacy_credentials_are_never_forwarded_or_silently_deleted(
    tmp_path: Path,
    isolated_keyring: dict,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    isolated_keyring[("autogpt-local-executor", "access_token")] = "legacy-access-secret"
    isolated_keyring[("autogpt-local-executor", "refresh_token")] = "legacy-refresh-secret"
    saved = isolated_keyring.copy()
    config = _config(tmp_path)
    store = KeychainTokenStore(config)
    client = AsyncMock()
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = client
    monkeypatch.setattr("autogpt_local_executor.auth.httpx.AsyncClient", factory)

    with pytest.raises(ValueError, match="legacy.*autogpt-shim.*auth") as exc:
        if operation == "access":
            await store.get_access_token()
        elif operation == "refresh":
            await OAuthFlow(config, store).refresh_token()
        else:
            await OAuthFlow(config, store).revoke_tokens()

    assert "secret" not in str(exc.value)
    assert isolated_keyring == saved
    client.post.assert_not_awaited()
    await store.store_tokens("new-access", "new-refresh")
    assert await store.get_access_token() == "new-access"
    assert all(isolated_keyring[key] == value for key, value in saved.items())


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["metadata", "malformed", "missing-metadata"])
async def test_invalid_scope_metadata_fails_closed_without_token_disclosure(
    tmp_path: Path, isolated_keyring: dict, corruption: str
) -> None:
    store = KeychainTokenStore(_config(tmp_path))
    await store.store_tokens("access-secret", "refresh-secret")
    key = next(iter(isolated_keyring))
    record = json.loads(isolated_keyring[key])
    if corruption == "metadata":
        record["scope"]["client_id"] = "untrusted-client"
    elif corruption == "missing-metadata":
        del record["scope"]
    isolated_keyring[key] = (
        "invalid-json-access-secret" if corruption == "malformed" else json.dumps(record)
    )
    saved = isolated_keyring.copy()

    with pytest.raises(ValueError, match="credentials.*autogpt-shim.*auth") as exc:
        await store.get_access_token()

    assert "secret" not in str(exc.value)
    assert isolated_keyring == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["refresh", "revoke", "exchange"])
async def test_changed_flow_destination_cannot_receive_existing_credentials(
    tmp_path: Path,
    isolated_keyring: dict,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    config = _config(tmp_path)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-secret", "refresh-secret")
    flow = OAuthFlow(config, store)
    config.platform_url = "https://untrusted.example.com"
    client = AsyncMock()
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = client
    monkeypatch.setattr("autogpt_local_executor.auth.httpx.AsyncClient", factory)

    with pytest.raises(ValueError, match="configuration.*changed"):
        if operation == "refresh":
            await flow.refresh_token()
        elif operation == "revoke":
            await flow.revoke_tokens()
        else:
            await flow._exchange_code("code-secret", "verifier-secret")

    client.post.assert_not_awaited()
    assert await store.get_access_token() == "access-secret"


@pytest.mark.asyncio
async def test_refresh_and_revoke_only_use_the_current_deployment_record(
    tmp_path: Path, isolated_keyring: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _config(tmp_path)
    other = original.model_copy(update={"platform_url": "https://other.example.com/backend"})
    original_store = KeychainTokenStore(original)
    other_store = KeychainTokenStore(other)
    await original_store.store_tokens("original-access", "original-refresh")
    await other_store.store_tokens("other-access", "other-refresh")
    requests: list[tuple[str, dict[str, str]]] = []

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"access_token": "rotated-access", "refresh_token": "rotated-refresh"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, data=None, json=None):
            requests.append((url, data or json))
            return Response()

    monkeypatch.setattr("autogpt_local_executor.auth.httpx.AsyncClient", Client)
    flow = OAuthFlow(other, other_store)
    await flow.refresh_token()
    assert await other_store.get_access_token() == "rotated-access"
    assert await other_store.get_refresh_token() == "rotated-refresh"
    await flow.revoke_tokens()

    assert [url for url, _ in requests] == [
        "https://other.example.com/backend/api/oauth/token",
        "https://other.example.com/backend/api/oauth/revoke",
        "https://other.example.com/backend/api/oauth/revoke",
    ]
    assert requests[0][1]["refresh_token"] == "other-refresh"
    assert [body["token"] for _, body in requests[1:]] == ["rotated-access", "rotated-refresh"]
    assert await other_store.get_access_token() is None
    assert await original_store.get_access_token() == "original-access"
    assert await original_store.get_refresh_token() == "original-refresh"


def test_flow_rejects_a_store_for_a_different_deployment(tmp_path: Path) -> None:
    original = _config(tmp_path)
    store = KeychainTokenStore(original)
    other = original.model_copy(update={"platform_url": "https://other.example.com"})

    with pytest.raises(ValueError, match="different OAuth configuration"):
        OAuthFlow(other, store)


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
    with pytest.raises(ValueError, match="TLS"):
        flow = OAuthFlow(config, token_store=store)  # type: ignore[arg-type]
        if operation == "exchange":
            await flow._exchange_code("code", "verifier")
        elif operation == "refresh":
            await flow.refresh_token()
        else:
            await flow.revoke_tokens()

    client.post.assert_not_awaited()
    assert store.cleared is False
