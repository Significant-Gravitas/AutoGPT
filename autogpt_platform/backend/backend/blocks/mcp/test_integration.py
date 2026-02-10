"""
Integration tests for MCP client and MCPToolBlock against a real HTTP server.

These tests spin up a local MCP test server and run the full client/block flow
against it — no mocking, real HTTP requests.
"""

import asyncio
import json
import threading
from unittest.mock import patch

import pytest
from aiohttp import web
from pydantic import SecretStr

from backend.blocks.mcp.block import MCPToolBlock
from backend.blocks.mcp.client import MCPClient
from backend.blocks.mcp.test_server import create_test_mcp_app
from backend.data.model import OAuth2Credentials

MOCK_USER_ID = "test-user-integration"


class _MCPTestServer:
    """
    Run an MCP test server in a background thread with its own event loop.
    This avoids event loop conflicts with pytest-asyncio.
    """

    def __init__(self, auth_token: str | None = None):
        self.auth_token = auth_token
        self.url: str = ""
        self._runner: web.AppRunner | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start())
        self._started.set()
        self._loop.run_forever()

    async def _start(self):
        app = create_test_mcp_app(auth_token=self.auth_token)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.url = f"http://127.0.0.1:{port}/mcp"

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=5):
            raise RuntimeError("MCP test server failed to start within 5 seconds")
        return self

    def stop(self):
        if self._loop and self._runner:
            asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(
                timeout=5
            )
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)


@pytest.fixture(scope="module")
def mcp_server():
    """Start a local MCP test server in a background thread."""
    server = _MCPTestServer()
    server.start()
    yield server.url
    server.stop()


@pytest.fixture(scope="module")
def mcp_server_with_auth():
    """Start a local MCP test server with auth in a background thread."""
    server = _MCPTestServer(auth_token="test-secret-token")
    server.start()
    yield server.url, "test-secret-token"
    server.stop()


@pytest.fixture(autouse=True)
def _allow_localhost():
    """
    Allow 127.0.0.1 through SSRF protection for integration tests.

    The Requests class blocks private IPs by default. We patch the Requests
    constructor to always include 127.0.0.1 as a trusted origin so the local
    test server is reachable.
    """
    from backend.util.request import Requests

    original_init = Requests.__init__

    def patched_init(self, *args, **kwargs):
        trusted = list(kwargs.get("trusted_origins") or [])
        trusted.append("http://127.0.0.1")
        kwargs["trusted_origins"] = trusted
        original_init(self, *args, **kwargs)

    with patch.object(Requests, "__init__", patched_init):
        yield


def _make_client(url: str, auth_token: str | None = None) -> MCPClient:
    """Create an MCPClient for integration tests."""
    return MCPClient(url, auth_token=auth_token)


# ── MCPClient integration tests ──────────────────────────────────────


class TestMCPClientIntegration:
    """Test MCPClient against a real local MCP server."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_initialize(self, mcp_server):
        client = _make_client(mcp_server)
        result = await client.initialize()

        assert result["protocolVersion"] == "2025-03-26"
        assert result["serverInfo"]["name"] == "test-mcp-server"
        assert "tools" in result["capabilities"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_list_tools(self, mcp_server):
        client = _make_client(mcp_server)
        await client.initialize()
        tools = await client.list_tools()

        assert len(tools) == 3

        tool_names = {t.name for t in tools}
        assert tool_names == {"get_weather", "add_numbers", "echo"}

        # Check get_weather schema
        weather = next(t for t in tools if t.name == "get_weather")
        assert weather.description == "Get current weather for a city"
        assert "city" in weather.input_schema["properties"]
        assert weather.input_schema["required"] == ["city"]

        # Check add_numbers schema
        add = next(t for t in tools if t.name == "add_numbers")
        assert "a" in add.input_schema["properties"]
        assert "b" in add.input_schema["properties"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_call_tool_get_weather(self, mcp_server):
        client = _make_client(mcp_server)
        await client.initialize()
        result = await client.call_tool("get_weather", {"city": "London"})

        assert not result.is_error
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"

        data = json.loads(result.content[0]["text"])
        assert data["city"] == "London"
        assert data["temperature"] == 22
        assert data["condition"] == "sunny"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_call_tool_add_numbers(self, mcp_server):
        client = _make_client(mcp_server)
        await client.initialize()
        result = await client.call_tool("add_numbers", {"a": 3, "b": 7})

        assert not result.is_error
        data = json.loads(result.content[0]["text"])
        assert data["result"] == 10

    @pytest.mark.asyncio(loop_scope="session")
    async def test_call_tool_echo(self, mcp_server):
        client = _make_client(mcp_server)
        await client.initialize()
        result = await client.call_tool("echo", {"message": "Hello MCP!"})

        assert not result.is_error
        assert result.content[0]["text"] == "Hello MCP!"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_call_unknown_tool(self, mcp_server):
        client = _make_client(mcp_server)
        await client.initialize()
        result = await client.call_tool("nonexistent_tool", {})

        assert result.is_error
        assert "Unknown tool" in result.content[0]["text"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_auth_success(self, mcp_server_with_auth):
        url, token = mcp_server_with_auth
        client = _make_client(url, auth_token=token)
        result = await client.initialize()

        assert result["protocolVersion"] == "2025-03-26"

        tools = await client.list_tools()
        assert len(tools) == 3

    @pytest.mark.asyncio(loop_scope="session")
    async def test_auth_failure(self, mcp_server_with_auth):
        url, _ = mcp_server_with_auth
        client = _make_client(url, auth_token="wrong-token")

        with pytest.raises(Exception):
            await client.initialize()

    @pytest.mark.asyncio(loop_scope="session")
    async def test_auth_missing(self, mcp_server_with_auth):
        url, _ = mcp_server_with_auth
        client = _make_client(url)

        with pytest.raises(Exception):
            await client.initialize()


# ── MCPToolBlock integration tests ───────────────────────────────────


class TestMCPToolBlockIntegration:
    """Test MCPToolBlock end-to-end against a real local MCP server."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_full_flow_get_weather(self, mcp_server):
        """Full flow: discover tools, select one, execute it."""
        # Step 1: Discover tools (simulating what the frontend/API would do)
        client = _make_client(mcp_server)
        await client.initialize()
        tools = await client.list_tools()
        assert len(tools) == 3

        # Step 2: User selects "get_weather" and we get its schema
        weather_tool = next(t for t in tools if t.name == "get_weather")

        # Step 3: Execute the block — no credentials (public server)
        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url=mcp_server,
            selected_tool="get_weather",
            tool_input_schema=weather_tool.input_schema,
            tool_arguments={"city": "Paris"},
        )

        outputs = []
        async for name, data in block.run(input_data, user_id=MOCK_USER_ID):
            outputs.append((name, data))

        assert len(outputs) == 1
        assert outputs[0][0] == "result"
        result = outputs[0][1]
        assert result["city"] == "Paris"
        assert result["temperature"] == 22
        assert result["condition"] == "sunny"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_full_flow_add_numbers(self, mcp_server):
        """Full flow for add_numbers tool."""
        client = _make_client(mcp_server)
        await client.initialize()
        tools = await client.list_tools()
        add_tool = next(t for t in tools if t.name == "add_numbers")

        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url=mcp_server,
            selected_tool="add_numbers",
            tool_input_schema=add_tool.input_schema,
            tool_arguments={"a": 42, "b": 58},
        )

        outputs = []
        async for name, data in block.run(input_data, user_id=MOCK_USER_ID):
            outputs.append((name, data))

        assert len(outputs) == 1
        assert outputs[0][0] == "result"
        assert outputs[0][1]["result"] == 100

    @pytest.mark.asyncio(loop_scope="session")
    async def test_full_flow_echo_plain_text(self, mcp_server):
        """Verify plain text (non-JSON) responses work."""
        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url=mcp_server,
            selected_tool="echo",
            tool_input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            tool_arguments={"message": "Hello from AutoGPT!"},
        )

        outputs = []
        async for name, data in block.run(input_data, user_id=MOCK_USER_ID):
            outputs.append((name, data))

        assert len(outputs) == 1
        assert outputs[0][0] == "result"
        assert outputs[0][1] == "Hello from AutoGPT!"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_full_flow_unknown_tool_yields_error(self, mcp_server):
        """Calling an unknown tool should yield an error output."""
        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url=mcp_server,
            selected_tool="nonexistent_tool",
            tool_arguments={},
        )

        outputs = []
        async for name, data in block.run(input_data, user_id=MOCK_USER_ID):
            outputs.append((name, data))

        assert len(outputs) == 1
        assert outputs[0][0] == "error"
        assert "returned an error" in outputs[0][1]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_full_flow_with_auth(self, mcp_server_with_auth):
        """Full flow with authentication via credentials kwarg."""
        url, token = mcp_server_with_auth

        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url=url,
            selected_tool="echo",
            tool_input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            tool_arguments={"message": "Authenticated!"},
        )

        # Pass credentials via the standard kwarg (as the executor would)
        test_creds = OAuth2Credentials(
            id="test-cred",
            provider="mcp",
            access_token=SecretStr(token),
            refresh_token=SecretStr(""),
            scopes=[],
            title="Test MCP credential",
        )

        outputs = []
        async for name, data in block.run(
            input_data, user_id=MOCK_USER_ID, credentials=test_creds
        ):
            outputs.append((name, data))

        assert len(outputs) == 1
        assert outputs[0][0] == "result"
        assert outputs[0][1] == "Authenticated!"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_no_credentials_runs_without_auth(self, mcp_server):
        """Block runs without auth when no credentials are provided."""
        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url=mcp_server,
            selected_tool="echo",
            tool_input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            tool_arguments={"message": "No auth needed"},
        )

        outputs = []
        async for name, data in block.run(
            input_data, user_id=MOCK_USER_ID, credentials=None
        ):
            outputs.append((name, data))

        assert len(outputs) == 1
        assert outputs[0][0] == "result"
        assert outputs[0][1] == "No auth needed"
