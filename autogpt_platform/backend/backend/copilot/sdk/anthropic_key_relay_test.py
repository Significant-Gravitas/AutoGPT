"""The relay puts a user's key on a CLI launch's Messages API calls, and on
nothing else, for the holder of its token only."""

import datetime
import ipaddress
import ssl
from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from yarl import URL

from .anthropic_key_relay import AnthropicKeyRelay

_USER_KEY = "sk-ant-FAKE-the-users-own-key"
_REPLY = b'event: message_start\ndata: {"type": "message_start"}\n\n'


class _Upstream:
    """Stands in for the Anthropic API, and keeps each request it is sent."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, str], bytes]] = []
        self.url = ""

    async def handle(self, request: web.Request) -> web.StreamResponse:
        headers = {name.lower(): value for name, value in request.headers.items()}
        self.requests.append(
            (request.method, request.raw_path, headers, await request.read())
        )
        if request.headers.get("x-api-key") != _USER_KEY:
            return web.json_response({"type": "error"}, status=401)
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "request-id": "req_1"},
        )
        await response.prepare(request)
        await response.write(_REPLY)
        return response


@pytest_asyncio.fixture(loop_scope="function")
async def upstream() -> AsyncIterator[_Upstream]:
    upstream = _Upstream()
    runner = await _serve(upstream)
    yield upstream
    await runner.cleanup()


@pytest_asyncio.fixture(loop_scope="function")
async def client() -> AsyncIterator[aiohttp.ClientSession]:
    # A proxy configured in the environment is not for loopback.
    async with aiohttp.ClientSession(trust_env=False) as session:
        yield session


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
async def test_a_model_call_goes_upstream_on_the_users_key(
    path: str, upstream: _Upstream, client: aiohttp.ClientSession
):
    relay = AnthropicKeyRelay(f"{upstream.url}/anthropic/", _USER_KEY)
    env = await relay.start()
    try:
        async with client.post(
            f"{env['ANTHROPIC_BASE_URL']}{path}?beta=true",
            data=b'{"model": "claude-sonnet-4-6"}',
            headers={
                "x-api-key": env["ANTHROPIC_API_KEY"],
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "User-Agent": "claude-cli/2.1.284",
            },
        ) as response:
            status, headers, body = (
                response.status,
                response.headers,
                await response.read(),
            )
    finally:
        await relay.close()

    assert (status, body) == (200, _REPLY)
    assert headers["Content-Type"] == "text/event-stream"
    assert headers["request-id"] == "req_1"
    [(method, sent_path, sent, sent_body)] = upstream.requests
    assert (method, sent_path) == ("POST", f"/anthropic{path}?beta=true")
    assert sent_body == b'{"model": "claude-sonnet-4-6"}'
    assert sent["x-api-key"] == _USER_KEY
    assert sent["anthropic-version"] == "2023-06-01"
    assert sent["user-agent"] == "claude-cli/2.1.284"
    assert "authorization" not in sent
    assert not any(env["ANTHROPIC_API_KEY"] in value for value in sent.values())


@pytest.mark.asyncio(loop_scope="function")
async def test_the_path_upstream_is_the_relayed_route(
    upstream: _Upstream, client: aiohttp.ClientSession
):
    """A path written differently but matching a relayed route goes upstream
    as that route."""
    relay = AnthropicKeyRelay(upstream.url, _USER_KEY)
    env = await relay.start()
    try:
        async with client.post(
            URL(f"{env['ANTHROPIC_BASE_URL']}/v1/%6Dessages", encoded=True),
            data=b"{}",
            headers={"x-api-key": env["ANTHROPIC_API_KEY"]},
        ) as response:
            status = response.status
    finally:
        await relay.close()

    assert status == 200
    [(_, path, _, _)] = upstream.requests
    assert path == "/v1/messages"


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize(
    "sent_in", [("x-api-key", "Authorization"), ("Authorization",)]
)
async def test_the_key_goes_as_a_bearer_token_too_to_an_upstream_that_takes_either(
    sent_in: tuple[str, ...], upstream: _Upstream, client: aiohttp.ClientSession
):
    """OpenRouter: the CLI sends its token both ways, and either opens the
    relay."""
    relay = AnthropicKeyRelay(upstream.url, _USER_KEY, as_bearer_too=True)
    env = await relay.start()
    token = env["ANTHROPIC_AUTH_TOKEN"]
    credential = {"x-api-key": token, "Authorization": f"Bearer {token}"}
    try:
        async with client.post(
            f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
            data=b"{}",
            headers={name: credential[name] for name in sent_in},
        ) as response:
            status = response.status
    finally:
        await relay.close()

    assert status == 200
    assert token == env["ANTHROPIC_API_KEY"]
    [(_, _, sent, _)] = upstream.requests
    assert sent["x-api-key"] == _USER_KEY
    assert sent["authorization"] == f"Bearer {_USER_KEY}"


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"x-api-key": "not-the-token"},
        {"Authorization": "Bearer not-the-token"},
        # The key itself opens nothing: only the launch's token does.
        {"x-api-key": _USER_KEY},
        {"x-api-key": "café"},
    ],
)
async def test_a_call_without_the_launchs_token_is_refused(
    headers: dict[str, str], upstream: _Upstream, client: aiohttp.ClientSession
):
    relay = AnthropicKeyRelay(upstream.url, _USER_KEY)
    env = await relay.start()
    try:
        async with client.post(
            f"{env['ANTHROPIC_BASE_URL']}/v1/messages", data=b"{}", headers=headers
        ) as response:
            status, body = response.status, await response.json()
    finally:
        await relay.close()

    assert status == 401
    assert body["error"]["type"] == "authentication_error"
    assert upstream.requests == []


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/"),
        ("GET", "/v1/models"),
        ("POST", "/v1/complete"),
        ("GET", "/v1/messages"),
        ("POST", "/v1/messages/batches"),
    ],
)
async def test_only_the_messages_api_is_relayed(
    method: str, path: str, upstream: _Upstream, client: aiohttp.ClientSession
):
    relay = AnthropicKeyRelay(upstream.url, _USER_KEY)
    env = await relay.start()
    try:
        async with client.request(
            method,
            f"{env['ANTHROPIC_BASE_URL']}{path}",
            headers={"x-api-key": env["ANTHROPIC_API_KEY"]},
        ) as response:
            status = response.status
    finally:
        await relay.close()

    assert status in (404, 405)
    assert upstream.requests == []


@pytest.mark.asyncio(loop_scope="function")
async def test_an_upstream_error_comes_back_as_it_is(
    upstream: _Upstream, client: aiohttp.ClientSession
):
    """A key the provider refuses is reported as it was without the relay."""
    relay = AnthropicKeyRelay(upstream.url, "sk-ant-FAKE-revoked")
    env = await relay.start()
    try:
        async with client.post(
            f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
            data=b"{}",
            headers={"x-api-key": env["ANTHROPIC_API_KEY"]},
        ) as response:
            status, body = response.status, await response.json()
    finally:
        await relay.close()

    assert (status, body) == (401, {"type": "error"})


@pytest.mark.asyncio(loop_scope="function")
async def test_an_upstream_that_cannot_be_reached_is_an_api_error(
    client: aiohttp.ClientSession, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    relay = AnthropicKeyRelay("http://127.0.0.1:1", _USER_KEY)
    env = await relay.start()
    try:
        async with client.post(
            f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
            data=b"{}",
            headers={"x-api-key": env["ANTHROPIC_API_KEY"]},
        ) as response:
            status, body = response.status, await response.json()
    finally:
        await relay.close()

    assert status == 502
    assert body["error"]["type"] == "api_error"
    assert _USER_KEY not in body["error"]["message"]


@pytest.mark.asyncio(loop_scope="function")
async def test_a_redirect_comes_back_unfollowed(
    upstream: _Upstream, client: aiohttp.ClientSession
):
    """The key goes to the configured endpoint only: a redirect elsewhere is
    the CLI's to follow, back through the relay."""

    async def redirect(request: web.Request) -> web.StreamResponse:
        raise web.HTTPTemporaryRedirect(f"{upstream.url}/v1/messages")

    application = web.Application()
    application.router.add_route("*", "/{path:.*}", redirect)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    relay = AnthropicKeyRelay(f"http://127.0.0.1:{runner.addresses[0][1]}", _USER_KEY)
    env = await relay.start()
    try:
        async with client.post(
            f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
            data=b"{}",
            headers={"x-api-key": env["ANTHROPIC_API_KEY"]},
            allow_redirects=False,
        ) as response:
            status = response.status
    finally:
        await relay.close()
        await runner.cleanup()

    assert status == 307
    assert upstream.requests == []


@pytest.mark.asyncio(loop_scope="function")
async def test_upstream_is_trusted_as_the_cli_was_told_to_trust_it(
    client: aiohttp.ClientSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A deployment whose egress re-signs TLS names its CA to the CLI in
    ``NODE_EXTRA_CA_CERTS``; the relay now makes the CLI's calls."""
    certificate, private_key = _self_signed(tmp_path)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(certificate, private_key)
    upstream = _Upstream()
    runner = await _serve(upstream, tls)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")

    async def call() -> int:
        relay = AnthropicKeyRelay(upstream.url, _USER_KEY)
        env = await relay.start()
        try:
            async with client.post(
                f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
                data=b"{}",
                headers={"x-api-key": env["ANTHROPIC_API_KEY"]},
            ) as response:
                return response.status
        finally:
            await relay.close()

    try:
        monkeypatch.delenv("NODE_EXTRA_CA_CERTS", raising=False)
        assert await call() == 502
        monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(certificate))
        assert await call() == 200
        # Like node, the relay starts without a bundle it cannot read.
        monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(tmp_path / "missing.pem"))
        assert await call() == 502
    finally:
        await runner.cleanup()


async def _serve(
    upstream: _Upstream, tls: ssl.SSLContext | None = None
) -> web.AppRunner:
    application = web.Application()
    application.router.add_route("*", "/{path:.*}", upstream.handle)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0, ssl_context=tls).start()
    port = runner.addresses[0][1]
    upstream.url = f"{'https' if tls else 'http'}://127.0.0.1:{port}"
    return runner


def _self_signed(directory: Path) -> tuple[Path, Path]:
    """A certificate for 127.0.0.1 that is its own CA, and its key."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    address = x509.IPAddress(ipaddress.ip_address("127.0.0.1"))
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([address]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    certificate_path, key_path = directory / "upstream.crt", directory / "upstream.key"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path
