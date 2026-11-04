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
    assert query["scope"] == ["CONNECT_LOCAL_EXECUTOR"]
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


def test_callback_wait_outlasts_requests_for_other_paths(tmp_path: Path) -> None:
    """A browser asks for /favicon.ico; that must not end the wait."""
    flow = OAuthFlow(_config(tmp_path, callback_port=_free_port()), token_store=object())
    server, codes, errors, port = flow._bind_callback_server(expected_state="right-state")
    statuses: list[int] = []

    def browse() -> None:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/favicon.ico", timeout=30)
        except urllib.error.HTTPError as exc:
            statuses.append(exc.code)
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/callback?code=the-code&state=right-state", timeout=30
        ) as response:
            statuses.append(response.status)

    thread = threading.Thread(target=browse)
    thread.start()
    try:
        assert flow._wait_for_callback(server, codes, errors, timeout=60) == "the-code"
    finally:
        thread.join(timeout=30)
    assert statuses == [404, 200]


def test_callback_wait_gives_up_at_its_deadline(tmp_path: Path) -> None:
    flow = OAuthFlow(_config(tmp_path, callback_port=_free_port()), token_store=object())
    server, codes, errors, _port = flow._bind_callback_server(expected_state="state")

    with pytest.raises(ValueError, match="timed out"):
        flow._wait_for_callback(server, codes, errors, timeout=0.2)


def test_a_callback_for_another_sign_in_is_refused_without_ending_the_wait(
    tmp_path: Path,
) -> None:
    """A request to /callback with another state, or none, ended the sign-in
    with an error, so any page the browser loaded could end it by requesting
    that URL, and a tab left from an earlier attempt ended the new one. Such
    a request is refused, and the wait goes on for this sign-in's redirect."""
    flow = OAuthFlow(_config(tmp_path, callback_port=_free_port()), token_store=object())
    server, codes, errors, port = flow._bind_callback_server(expected_state="right-state")
    statuses: list[int] = []

    def browse() -> None:
        for query in (
            "code=planted&state=wrong-state",
            "code=planted",
            "error=access_denied&state=wrong-state",
            "code=planted&state=%C3%A9",  # compare_digest refuses non-ASCII str
        ):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/callback?{query}", timeout=30)
            except urllib.error.HTTPError as exc:
                statuses.append(exc.code)
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/callback?code=the-code&state=right-state", timeout=30
        ) as response:
            statuses.append(response.status)

    thread = threading.Thread(target=browse)
    thread.start()
    try:
        assert flow._wait_for_callback(server, codes, errors, timeout=60) == "the-code"
    finally:
        thread.join(timeout=30)
    assert statuses == [400, 400, 400, 400, 200]
    assert errors == []


def test_a_denial_for_this_sign_in_still_ends_the_wait(tmp_path: Path) -> None:
    flow = OAuthFlow(_config(tmp_path, callback_port=_free_port()), token_store=object())
    server, codes, errors, port = flow._bind_callback_server(expected_state="right-state")

    def deny() -> None:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/callback?error=access_denied&state=right-state",
                timeout=30,
            )
        except urllib.error.HTTPError:
            pass

    thread = threading.Thread(target=deny)
    thread.start()
    try:
        with pytest.raises(ValueError, match="access_denied"):
            flow._wait_for_callback(server, codes, errors, timeout=60)
    finally:
        thread.join(timeout=30)


@pytest.mark.parametrize(("platform", "exclusive"), [("win32", True), ("linux", False)])
def test_the_callback_takes_its_port_exclusively_on_windows(
    monkeypatch: pytest.MonkeyPatch, platform: str, exclusive: bool
) -> None:
    """HTTPServer sets SO_REUSEADDR, with which a Windows socket binds a port
    another socket is already listening on: the fallback to the next port
    never ran, and the redirect could reach the other listener. Elsewhere it
    only lets a new sign-in bind over the last one's closed connections."""
    import sys
    from http.server import BaseHTTPRequestHandler

    import autogpt_local_executor.auth as auth_module

    server = auth_module._CallbackServer(
        ("127.0.0.1", 0), BaseHTTPRequestHandler, bind_and_activate=False
    )
    real_socket = server.socket
    fake_socket = MagicMock()
    fake_socket.getsockname.return_value = ("127.0.0.1", 41899)
    server.socket = fake_socket
    monkeypatch.setattr(socket, "SO_EXCLUSIVEADDRUSE", -7, raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    try:
        server.server_bind()
    finally:
        monkeypatch.undo()
        real_socket.close()

    options = [option.args[1] for option in fake_socket.setsockopt.call_args_list]
    assert (-7 in options) is exclusive
    assert (socket.SO_REUSEADDR in options) is not exclusive


def test_callback_uses_next_port_when_first_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogpt_local_executor.auth as auth_module

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    first_port = blocker.getsockname()[1]
    # A free port is rarely in the registered 41899-41910, so the range is
    # moved to start at it.
    monkeypatch.setattr(auth_module, "CALLBACK_LAST_PORT", first_port + 11)
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


def test_a_connection_that_sends_nothing_cannot_block_the_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server takes one connection at a time with no read timeout, so a
    browser preconnect that never sent a request held off the real redirect
    until the whole sign-in timed out."""
    import autogpt_local_executor.auth as auth_module

    monkeypatch.setattr(auth_module, "CALLBACK_READ_TIMEOUT_SECONDS", 0.5)
    flow = OAuthFlow(_config(tmp_path, callback_port=_free_port()), token_store=object())
    server, codes, errors, port = flow._bind_callback_server(expected_state="right-state")
    idle = socket.create_connection(("127.0.0.1", port), timeout=30)

    def redirect() -> None:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/callback?code=the-code&state=right-state", timeout=30
        ):
            pass

    thread = threading.Thread(target=redirect)
    thread.start()
    try:
        assert flow._wait_for_callback(server, codes, errors, timeout=20) == "the-code"
    finally:
        thread.join(timeout=30)
        idle.close()


def test_a_request_that_trickles_in_cannot_hold_the_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read timeout bounds each read, not the request, so a client
    sending a byte every so often kept the one-at-a-time server reading its
    headers, and the redirect waited past the sign-in's deadline."""
    import time

    import autogpt_local_executor.auth as auth_module

    monkeypatch.setattr(auth_module, "CALLBACK_READ_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(auth_module, "CALLBACK_REQUEST_SECONDS", 1.0, raising=False)
    flow = OAuthFlow(_config(tmp_path, callback_port=_free_port()), token_store=object())
    server, codes, errors, port = flow._bind_callback_server(expected_state="right-state")
    slow = socket.create_connection(("127.0.0.1", port), timeout=30)
    stop = threading.Event()

    def trickle() -> None:
        # Headers that never end, a byte at a time, well inside each read's
        # timeout, until the server hangs up (or a minute passes, so a server
        # that never does fails the test instead of hanging it).
        give_up = time.monotonic() + 60
        try:
            slow.sendall(b"GET /favicon.ico HTTP/1.1\r\nX-Slow: ")
            while not stop.wait(0.1) and time.monotonic() < give_up:
                slow.sendall(b"x")
        except OSError:
            pass

    def redirect() -> None:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/callback?code=the-code&state=right-state", timeout=90
        ):
            pass

    trickler = threading.Thread(target=trickle)
    trickler.start()
    redirector = threading.Thread(target=redirect)
    redirector.start()
    try:
        assert flow._wait_for_callback(server, codes, errors, timeout=20) == "the-code"
    finally:
        stop.set()
        trickler.join(timeout=90)
        redirector.join(timeout=90)
        slow.close()


@pytest.mark.parametrize(
    ("first", "tried"),
    [(41899, list(range(41899, 41911))), (41905, list(range(41905, 41911))), (50000, [50000])],
    ids=["default", "within-range", "custom"],
)
def test_the_callback_binds_only_registered_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: int, tried: list[int]
) -> None:
    """The fallback tried twelve ports from the configured one, so a start
    above 41899 could bind 41911 or higher, which the OAuth app doesn't list,
    and sign-in failed on the redirect. A port outside 41899-41910 is the
    only one known to be registered."""
    import autogpt_local_executor.auth as auth_module

    attempts: list[int] = []

    def busy(address, _handler):
        attempts.append(address[1])
        raise OSError("address in use")

    monkeypatch.setattr(auth_module, "_CallbackServer", busy)
    flow = OAuthFlow(_config(tmp_path, callback_port=first), token_store=object())

    with pytest.raises(RuntimeError, match="Could not bind"):
        flow._bind_callback_server(expected_state="state")

    assert attempts == tried
