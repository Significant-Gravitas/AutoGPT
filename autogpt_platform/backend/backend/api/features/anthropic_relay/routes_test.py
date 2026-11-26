"""The relay route sends a Messages API call on with the user's own key only for
a live run token, and only for that user's own Anthropic credential."""

import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from aiohttp import web
from fastapi import FastAPI
from pydantic import SecretStr

from backend.api.features.anthropic_relay import routes
from backend.api.rest_api import app as rest_app
from backend.data.model import APIKeyCredentials
from backend.integrations.claude_code_relay import RELAY_PATH, RelayGrant
from backend.integrations.credentials_store import anthropic_credentials

_TOKEN = "relay-token-for-this-run"
_USER_KEY = "sk-ant-FAKE-the-users-own-key"
_REPLY = b'event: message_start\ndata: {"type": "message_start"}\n\n'
_GRANT = RelayGrant(user_id="user-a", credentials_id="cred-1")
_USERS_KEY = APIKeyCredentials(
    id="cred-1",
    provider="anthropic",
    api_key=SecretStr(_USER_KEY),
    title="My Anthropic key",
    expires_at=None,
)


class _Upstream:
    """Stands in for the Anthropic API, and keeps each request it is sent."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, list[tuple[str, str]], bytes]] = []
        self.url = ""

    async def handle(self, request: web.Request) -> web.StreamResponse:
        # Every header as sent, so a second one under the same name shows.
        headers = [(name.lower(), value) for name, value in request.headers.items()]
        self.requests.append((request.raw_path, headers, await request.read()))
        response = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)
        await response.write(_REPLY)
        return response


@pytest_asyncio.fixture
async def upstream() -> AsyncIterator[_Upstream]:
    upstream = _Upstream()
    application = web.Application()
    application.router.add_route("*", "/{path:.*}", upstream.handle)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    upstream.url = f"http://127.0.0.1:{runner.addresses[0][1]}"
    with patch.object(routes, "ANTHROPIC_API_URL", upstream.url):
        yield upstream
    await runner.cleanup()


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
@pytest.mark.parametrize(
    "credential",
    [{"x-api-key": _TOKEN}, {"authorization": f"Bearer {_TOKEN}"}],
    ids=["x-api-key", "bearer"],
)
async def test_a_call_with_a_live_token_goes_upstream_on_the_users_key(
    path: str, credential: dict[str, str], upstream: _Upstream
):
    grant = AsyncMock(return_value=_GRANT)
    get = AsyncMock(return_value=_USERS_KEY)
    with _lookups(grant, get):
        response = await _call(path, credential)

    assert (response.status_code, response.content) == (200, _REPLY)
    assert response.headers["content-type"] == "text/event-stream"
    grant.assert_awaited_once_with(_TOKEN)
    get.assert_awaited_once_with("user-a", "cred-1")
    [(sent_path, sent, body)] = upstream.requests
    assert (sent_path, body) == (f"{path}?beta=true", b'{"model": "claude"}')
    assert [value for name, value in sent if name == "x-api-key"] == [_USER_KEY]
    assert not any(_TOKEN in value for _, value in sent)


@pytest.mark.parametrize(
    "headers", [{}, {"x-api-key": "a-revoked-or-unknown-token"}], ids=["none", "dead"]
)
async def test_a_call_without_a_live_token_is_refused(
    headers: dict[str, str], upstream: _Upstream
):
    with _lookups(AsyncMock(return_value=None), AsyncMock(return_value=_USERS_KEY)):
        response = await _call("/v1/messages", headers)

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    assert upstream.requests == []


@pytest.mark.parametrize(
    "credentials",
    [
        None,
        _USERS_KEY.model_copy(update={"provider": "openai"}),
        _USERS_KEY.model_copy(update={"is_managed": True}),
        anthropic_credentials,
    ],
    ids=["gone", "other-provider", "managed", "platform-key"],
)
async def test_a_token_reaches_only_the_users_own_anthropic_key(
    credentials: APIKeyCredentials | None, upstream: _Upstream
):
    with _lookups(AsyncMock(return_value=_GRANT), AsyncMock(return_value=credentials)):
        response = await _call("/v1/messages", {"x-api-key": _TOKEN})

    assert response.status_code == 401
    assert upstream.requests == []


async def test_an_unreachable_anthropic_api_is_an_api_error():
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    port = closed.getsockname()[1]
    closed.close()
    with (
        patch.object(routes, "ANTHROPIC_API_URL", f"http://127.0.0.1:{port}"),
        _lookups(AsyncMock(return_value=_GRANT), AsyncMock(return_value=_USERS_KEY)),
    ):
        response = await _call("/v1/messages", {"x-api-key": _TOKEN})

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"


async def test_the_platform_serves_the_relay_where_sandboxes_are_sent():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=rest_app), base_url="http://platform"
    ) as client:
        response = await client.post(f"{RELAY_PATH}/v1/messages", content=b"{}")

    assert response.status_code == 401


@contextmanager
def _lookups(grant: AsyncMock, get: AsyncMock) -> Iterator[None]:
    with (
        patch.object(routes, "relay_grant", grant),
        patch.object(routes.creds_manager, "get", get),
    ):
        yield


async def _call(path: str, headers: dict[str, str]) -> httpx.Response:
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://platform"
    ) as client:
        return await client.post(
            f"/api/anthropic-relay{path}?beta=true",
            content=b'{"model": "claude"}',
            headers={"content-type": "application/json", **headers},
        )
