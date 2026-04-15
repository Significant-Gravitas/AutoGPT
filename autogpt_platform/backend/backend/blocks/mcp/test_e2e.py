"""
End-to-end tests against a real public MCP server.

These tests hit the OpenAI docs MCP server (https://developers.openai.com/mcp)
which is publicly accessible without authentication and returns SSE responses.

Mark: These are tagged with ``@pytest.mark.e2e`` so they can be run/skipped
independently of the rest of the test suite (they require network access).
"""

import json
import os

import pytest

from backend.blocks.mcp.client import MCPClient

# Public MCP server that requires no authentication
OPENAI_DOCS_MCP_URL = "https://developers.openai.com/mcp"

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
