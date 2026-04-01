"""
End-to-end tests against real public and authenticated MCP servers.

These tests hit live MCP servers and require network access:
  - OpenAI docs (https://developers.openai.com/mcp) — no auth required
  - Sentry (https://mcp.sentry.dev/mcp) — requires SENTRY_MCP_TOKEN env var
  - Linear (https://mcp.linear.app/mcp) — requires LINEAR_MCP_TOKEN env var

All tests are skipped unless the respective environment variables are set.
Mark: These are tagged with ``@pytest.mark.e2e`` so they can be run/skipped
independently of the rest of the test suite (they require network access).
"""

import json
import os

import pytest

from backend.blocks.mcp.client import MCPClient

# Public MCP server that requires no authentication
OPENAI_DOCS_MCP_URL = "https://developers.openai.com/mcp"

# Authenticated MCP servers
SENTRY_MCP_URL = "https://mcp.sentry.dev/mcp"
LINEAR_MCP_URL = "https://mcp.linear.app/mcp"

# Skip all tests in this module unless RUN_E2E env var is set
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_E2E"), reason="set RUN_E2E=1 to run e2e tests"
)


class TestRealMCPServer:
    """Tests against the live OpenAI docs MCP server."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_initialize(self):
        """Verify we can complete the MCP handshake with a real server."""
        client = MCPClient(OPENAI_DOCS_MCP_URL)
        result = await client.initialize()

        assert result["protocolVersion"] == "2025-03-26"
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "openai-docs-mcp"
        assert "tools" in result.get("capabilities", {})

    @pytest.mark.asyncio(loop_scope="session")
    async def test_list_tools(self):
        """Verify we can discover tools from a real MCP server."""
        client = MCPClient(OPENAI_DOCS_MCP_URL)
        await client.initialize()
        tools = await client.list_tools()

        assert len(tools) >= 3  # server has at least 5 tools as of writing

        tool_names = {t.name for t in tools}
        # These tools are documented and should be stable
        assert "search_openai_docs" in tool_names
        assert "list_openai_docs" in tool_names
        assert "fetch_openai_doc" in tool_names

        # Verify schema structure
        search_tool = next(t for t in tools if t.name == "search_openai_docs")
        assert "query" in search_tool.input_schema.get("properties", {})
        assert "query" in search_tool.input_schema.get("required", [])

    @pytest.mark.asyncio(loop_scope="session")
    async def test_call_tool_list_api_endpoints(self):
        """Call the list_api_endpoints tool and verify we get real data."""
        client = MCPClient(OPENAI_DOCS_MCP_URL)
        await client.initialize()
        result = await client.call_tool("list_api_endpoints", {})

        assert not result.is_error
        assert len(result.content) >= 1
        assert result.content[0]["type"] == "text"

        data = json.loads(result.content[0]["text"])
        assert "paths" in data or "urls" in data
        # The OpenAI API should have many endpoints
        total = data.get("total", len(data.get("paths", [])))
        assert total > 50

    @pytest.mark.asyncio(loop_scope="session")
    async def test_call_tool_search(self):
        """Search for docs and verify we get results."""
        client = MCPClient(OPENAI_DOCS_MCP_URL)
        await client.initialize()
        result = await client.call_tool(
            "search_openai_docs", {"query": "chat completions", "limit": 3}
        )

        assert not result.is_error
        assert len(result.content) >= 1

    @pytest.mark.asyncio(loop_scope="session")
    async def test_sse_response_handling(self):
        """Verify the client correctly handles SSE responses from a real server.

        This is the key test — our local test server returns JSON,
        but real MCP servers typically return SSE. This proves the
        SSE parsing works end-to-end.
        """
        client = MCPClient(OPENAI_DOCS_MCP_URL)
        # initialize() internally calls _send_request which must parse SSE
        result = await client.initialize()

        # If we got here without error, SSE parsing works
        assert isinstance(result, dict)
        assert "protocolVersion" in result

        # Also verify list_tools works (another SSE response)
        tools = await client.list_tools()
        assert len(tools) > 0
        assert all(hasattr(t, "name") for t in tools)


def _assert_has_relevant_tool(tool_names: set[str], keywords: list[str], server: str) -> None:
    """Assert that at least one tool name contains one of the expected keywords."""
    assert any(
        kw in name.lower() for name in tool_names for kw in keywords
    ), f"Expected a tool containing {keywords} on {server}, got: {sorted(tool_names)}"


class TestSentryMCPServer:
    """Tests against the live Sentry MCP server (https://mcp.sentry.dev/mcp).

    Requires SENTRY_MCP_TOKEN to be set to a valid Sentry OAuth access token.
    Skipped automatically when the token is not present.
    """

    @pytest.fixture
    def sentry_token(self) -> str:
        token = os.environ.get("SENTRY_MCP_TOKEN")
        if not token:
            pytest.skip("SENTRY_MCP_TOKEN not set")
        return token

    @pytest.mark.asyncio(loop_scope="session")
    async def test_initialize(self, sentry_token):
        """Verify we can complete the MCP handshake with Sentry."""
        client = MCPClient(SENTRY_MCP_URL, auth_token=sentry_token)
        result = await client.initialize()

        assert "protocolVersion" in result
        assert "serverInfo" in result
        assert "tools" in result.get("capabilities", {})

    @pytest.mark.asyncio(loop_scope="session")
    async def test_list_tools(self, sentry_token):
        """Verify we can discover tools from the Sentry MCP server."""
        client = MCPClient(SENTRY_MCP_URL, auth_token=sentry_token)
        await client.initialize()
        tools = await client.list_tools()

        assert len(tools) >= 1
        tool_names = {t.name for t in tools}
        _assert_has_relevant_tool(
            tool_names, ["issue", "event", "error"], "mcp.sentry.dev"
        )

        for tool in tools:
            assert tool.name
            assert isinstance(tool.input_schema, dict)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_sse_response_handling(self, sentry_token):
        """Verify the client handles SSE responses from the Sentry server."""
        client = MCPClient(SENTRY_MCP_URL, auth_token=sentry_token)
        result = await client.initialize()

        assert isinstance(result, dict)
        assert "protocolVersion" in result

        tools = await client.list_tools()
        assert len(tools) > 0
        assert all(hasattr(t, "name") for t in tools)


class TestLinearMCPServer:
    """Tests against the live Linear MCP server (https://mcp.linear.app/mcp).

    Requires LINEAR_MCP_TOKEN to be set to a valid Linear OAuth access token.
    Skipped automatically when the token is not present.
    """

    @pytest.fixture
    def linear_token(self) -> str:
        token = os.environ.get("LINEAR_MCP_TOKEN")
        if not token:
            pytest.skip("LINEAR_MCP_TOKEN not set")
        return token

    @pytest.mark.asyncio(loop_scope="session")
    async def test_initialize(self, linear_token):
        """Verify we can complete the MCP handshake with Linear."""
        client = MCPClient(LINEAR_MCP_URL, auth_token=linear_token)
        result = await client.initialize()

        assert "protocolVersion" in result
        assert "serverInfo" in result
        assert "tools" in result.get("capabilities", {})

    @pytest.mark.asyncio(loop_scope="session")
    async def test_list_tools(self, linear_token):
        """Verify we can discover tools from the Linear MCP server."""
        client = MCPClient(LINEAR_MCP_URL, auth_token=linear_token)
        await client.initialize()
        tools = await client.list_tools()

        assert len(tools) >= 1
        tool_names = {t.name for t in tools}
        _assert_has_relevant_tool(
            tool_names, ["issue", "project", "team"], "mcp.linear.app"
        )

        for tool in tools:
            assert tool.name
            assert isinstance(tool.input_schema, dict)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_sse_response_handling(self, linear_token):
        """Verify the client handles SSE responses from the Linear server."""
        client = MCPClient(LINEAR_MCP_URL, auth_token=linear_token)
        result = await client.initialize()

        assert isinstance(result, dict)
        assert "protocolVersion" in result

        tools = await client.list_tools()
        assert len(tools) > 0
        assert all(hasattr(t, "name") for t in tools)
