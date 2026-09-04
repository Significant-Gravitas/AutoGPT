"""End-to-end regression for manually-entered MCP API tokens (SECRT-2592).

The pieces of this path were each covered in isolation and each behaved
correctly on its own terms, yet composing them produced a server the UI
called "Connected" and the agent called unconnected.  So this test walks
the whole path with only the database faked:

    POST /mcp/token → credential store → auto_lookup_mcp_credential
                    → run_mcp_tool → MCPClient(auth_token=...)

Nothing in between is stubbed — in particular ``auto_lookup_mcp_credential``
and ``IntegrationCredentialsManager.refresh_if_needed`` run for real, which
is precisely where the token used to be dropped.
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
from backend.util.request import HTTPClientError

from ._test_data import make_session
from .models import MCPToolsDiscoveredResponse, SetupRequirementsResponse
from .run_mcp_tool import RunMCPToolTool

_USER_ID = "test-user-mcp-manual-token"
_SERVER_URL = "https://mcp.datafa.st/mcp"
_TOKEN = "dft_live_token_value"

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
        # takes a Redis lock. Only the store is faked here.
        patch(
            "backend.integrations.creds_manager.IntegrationCredentialsManager._locked",
            _noop_lock,
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


async def test_stored_token_reaches_the_mcp_client(client, store):
    """The token the user pasted must be the token the MCP server receives."""
    await _store_token(client)
    assert len(store.rows) == 1

    with patch(
        "backend.copilot.tools.run_mcp_tool.MCPClient",
        return_value=_mcp_client([_tool("get_analytics")]),
    ) as MockClient:
        response = await RunMCPToolTool()._execute(
            user_id=_USER_ID,
            session=make_session(_USER_ID),
            server_url=_SERVER_URL,
        )

    assert isinstance(response, MCPToolsDiscoveredResponse)
    MockClient.assert_called_once_with(_SERVER_URL, auth_token=_TOKEN)


async def test_connect_card_reports_connected_after_storing_a_token(client, store):
    """The agent's view and the UI's green pill must agree.

    ``surface_connect_card`` reporting "not connected" over a credential the
    user just stored is the contradiction this ticket is about.
    """
    await _store_token(client)

    with patch(
        "backend.copilot.tools.run_mcp_tool.MCPClient", return_value=_mcp_client()
    ):
        response = await RunMCPToolTool()._execute(
            user_id=_USER_ID,
            session=make_session(_USER_ID),
            server_url=_SERVER_URL,
            surface_connect_card=True,
        )

    assert isinstance(response, SetupRequirementsResponse)
    assert response.setup_info.user_readiness.has_all_credentials is True


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

    assert MockClient.call_args.kwargs["auth_token"] == _TOKEN


async def test_dead_token_is_invalidated_on_401(client, store):
    """The self-healing path this PR unblocks.

    Before the fix the lookup returned ``None``, so ``creds is not None`` was
    never true and the dead row survived every retry — which is why the UI
    could keep showing Connected forever.
    """
    await _store_token(client)
    assert len(store.rows) == 1

    dead = _mcp_client()
    dead.initialize = AsyncMock(
        side_effect=HTTPClientError("HTTP 401", status_code=401)
    )
    with patch("backend.copilot.tools.run_mcp_tool.MCPClient", return_value=dead):
        response = await RunMCPToolTool()._execute(
            user_id=_USER_ID,
            session=make_session(_USER_ID),
            server_url=_SERVER_URL,
            tool_name="get_analytics",
        )

    assert isinstance(response, SetupRequirementsResponse)
    assert store.rows == []


async def test_scope_level_403_keeps_the_credential(client, store):
    """A 403 is routinely "this token may not call *that tool*". Deleting on
    it forces a re-entry that fails identically."""
    await _store_token(client)

    forbidden = _mcp_client()
    forbidden.initialize = AsyncMock(
        side_effect=HTTPClientError("HTTP 403", status_code=403)
    )
    with patch("backend.copilot.tools.run_mcp_tool.MCPClient", return_value=forbidden):
        await RunMCPToolTool()._execute(
            user_id=_USER_ID,
            session=make_session(_USER_ID),
            server_url=_SERVER_URL,
            tool_name="get_analytics",
        )

    assert len(store.rows) == 1


async def test_another_users_token_is_not_resolved(client, store):
    """Credential lookup is scoped to the owner."""
    await _store_token(client)

    with patch(
        "backend.copilot.tools.run_mcp_tool.MCPClient",
        return_value=_mcp_client([_tool("get_analytics")]),
    ) as MockClient:
        await RunMCPToolTool()._execute(
            user_id="a-different-user",
            session=make_session("a-different-user"),
            server_url=_SERVER_URL,
        )

    assert MockClient.call_args.kwargs["auth_token"] is None
