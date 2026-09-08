"""Round-trip regression for MCP server-URL normalization (SECRT-2592).

``/mcp/token`` writes ``metadata["mcp_server_url"]`` from the URL the user
typed; every lookup path re-derives that key from the same user input. If the
two sides ever normalize differently, the row is stored under a key no lookup
produces — 2xx, a green pill, and an agent that reports the server as not
connected. That is the contradiction this PR removes, and it is not visible
from either side alone, so this walks the whole path with only the database
faked:

    POST /mcp/token → credential store → auto_lookup_mcp_credential
                    → run_mcp_tool → MCPClient(authorization=...)

Nothing in between is stubbed. The behaviours these tests used to also cover —
the token reaching the client, the connect card, 401 invalidation, 403
retention, per-user isolation — landed on ``dev`` in #14075 or are covered by
``test_run_mcp_tool.py``, so only the normalization round trips remain here.
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import get_user_id

from backend.api.features.mcp.routes import router
from backend.data.model import Credentials

from ._test_data import make_session
from .run_mcp_tool import RunMCPToolTool

_USER_ID = "test-user-mcp-manual-token"
_SERVER_URL = "https://mcp.datafa.st/mcp"
_SERVER_URL_NO_SCHEME = "mcp.datafa.st/mcp"
_TOKEN = "dft_live_token_value"
# A bare token is stored (and sent) as a complete Bearer header.
_AUTHORIZATION = f"Bearer {_TOKEN}"

app = fastapi.FastAPI()
app.include_router(router)
app.dependency_overrides[get_user_id] = lambda: _USER_ID


class FakeCredentialsStore:
    """In-memory stand-in for ``IntegrationCredentialsStore``.

    Rows are keyed by ``user_id`` — the real store scopes every read to the
    owner, and a harness that ignored it would pass just as happily against
    code that served one user another's MCP token.

    Only the handful of methods this path touches are implemented; anything
    else should fail loudly rather than silently return a mock.
    """

    def __init__(self) -> None:
        self.by_user: dict[str, list[Credentials]] = {}

    @property
    def rows(self) -> list[Credentials]:
        return self.by_user.get(_USER_ID, [])

    async def add_creds(self, user_id: str, credentials: Credentials) -> None:
        self.by_user.setdefault(user_id, []).append(credentials)

    async def get_creds_by_provider(
        self, user_id: str, provider: str
    ) -> list[Credentials]:
        return [c for c in self.by_user.get(user_id, []) if c.provider == provider]

    async def get_creds_by_id(self, user_id: str, credentials_id: str):
        owned = self.by_user.get(user_id, [])
        return next((c for c in owned if c.id == credentials_id), None)

    async def delete_creds_by_id(self, user_id: str, credentials_id: str) -> None:
        owned = self.by_user.get(user_id, [])
        self.by_user[user_id] = [c for c in owned if c.id != credentials_id]


@pytest_asyncio.fixture
async def store():
    """Route every ``IntegrationCredentialsManager`` at one in-memory store.

    The routes module builds its manager at import time, so that instance is
    repointed directly; ``auto_lookup_mcp_credential`` builds a fresh manager
    per call, which the class-level patch covers.
    """
    fake = FakeCredentialsStore()
    with (
        patch(
            "backend.integrations.creds_manager.IntegrationCredentialsStore",
            return_value=fake,
        ),
        patch("backend.api.features.mcp.routes.creds_manager.store", fake),
        # ``invalidate_mcp_credential`` goes through ``mgr.delete``, which
        # takes a Redis lock, and every write publishes a creds-changed event
        # over Redis. Neither is what this test is about; only the store is
        # faked, everything between the route and it runs for real.
        patch(
            "backend.integrations.creds_manager.IntegrationCredentialsManager._locked",
            _noop_lock,
        ),
        patch(
            "backend.integrations.creds_manager._invoke_creds_changed_hook",
            new_callable=AsyncMock,
        ),
    ):
        yield fake


@contextlib.asynccontextmanager
async def _noop_lock(*_args, **_kwargs):
    yield


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _bypass_ssrf_validation():
    """Test URLs don't resolve; SSRF enforcement has its own tests."""
    with (
        patch(
            "backend.api.features.mcp.routes.validate_url_host",
            new_callable=AsyncMock,
        ),
        patch(
            "backend.copilot.tools.run_mcp_tool.validate_url_host",
            new_callable=AsyncMock,
        ),
    ):
        yield


def _mcp_client(tools: list | None = None):
    client = AsyncMock()
    client.initialize = AsyncMock(return_value={"protocolVersion": "2025-06-18"})
    client.list_tools = AsyncMock(return_value=tools or [])
    client.close = AsyncMock()
    return client


def _tool(name: str):
    t = MagicMock()
    t.name = name
    t.description = f"Description for {name}"
    t.input_schema = {"type": "object", "properties": {}, "required": []}
    return t


async def _store_token(client) -> None:
    with patch("backend.api.features.mcp.routes.MCPClient", return_value=_mcp_client()):
        response = await client.post(
            "/token", json={"server_url": _SERVER_URL, "token": _TOKEN}
        )
    assert response.status_code == 200, response.text


async def test_trailing_slash_variant_still_resolves(client, store):
    """The card and the agent can disagree on the trailing slash; the stored
    credential has to be found either way."""
    await _store_token(client)

    with patch(
        "backend.copilot.tools.run_mcp_tool.MCPClient",
        return_value=_mcp_client([_tool("get_analytics")]),
    ) as MockClient:
        await RunMCPToolTool()._execute(
            user_id=_USER_ID,
            session=make_session(_USER_ID),
            server_url=f"{_SERVER_URL}/",
        )

    assert MockClient.call_args.kwargs["authorization"] == _AUTHORIZATION


async def test_scheme_less_server_url_resolves_after_storing(client, store):
    """Storing and looking up must apply the same scheme rule.

    Users type ``mcp.example.com/mcp``. ``/token`` canonicalizes that to
    ``https://`` before writing ``metadata["mcp_server_url"]``, so every lookup
    has to canonicalize identically or the row is stored under a key no lookup
    ever produces — a 2xx and a green pill over a credential the agent cannot
    find, which is the exact contradiction this PR removes.
    """
    with patch("backend.api.features.mcp.routes.MCPClient", return_value=_mcp_client()):
        response = await client.post(
            "/token", json={"server_url": _SERVER_URL_NO_SCHEME, "token": _TOKEN}
        )
    assert response.status_code == 200, response.text

    with patch(
        "backend.copilot.tools.run_mcp_tool.MCPClient",
        return_value=_mcp_client([_tool("get_analytics")]),
    ) as MockClient:
        await RunMCPToolTool()._execute(
            user_id=_USER_ID,
            session=make_session(_USER_ID),
            server_url=_SERVER_URL_NO_SCHEME,
        )

    assert MockClient.call_args.kwargs["authorization"] == _AUTHORIZATION
