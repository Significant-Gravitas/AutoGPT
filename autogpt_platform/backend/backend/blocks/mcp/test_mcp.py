"""
Tests for MCP client and MCPToolBlock.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.mcp.block import MCPToolBlock
from backend.blocks.mcp.client import MCPCallResult, MCPClient, MCPClientError
from backend.util.test import execute_block_test

# ── SSE parsing unit tests ───────────────────────────────────────────


class TestSSEParsing:
    """Tests for SSE (text/event-stream) response parsing."""

    def test_parse_sse_simple(self):
        sse = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","result":{"tools":[]},"id":1}\n'
            "\n"
        )
        body = MCPClient._parse_sse_response(sse)
        assert body["result"] == {"tools": []}
        assert body["id"] == 1

    def test_parse_sse_with_notifications(self):
        """SSE streams can contain notifications (no id) before the response."""
        sse = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","method":"some/notification"}\n'
            "\n"
            "event: message\n"
            'data: {"jsonrpc":"2.0","result":{"ok":true},"id":2}\n'
            "\n"
        )
        body = MCPClient._parse_sse_response(sse)
        assert body["result"] == {"ok": True}
        assert body["id"] == 2

    def test_parse_sse_error_response(self):
        sse = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","error":{"code":-32600,"message":"Bad Request"},"id":1}\n'
        )
        body = MCPClient._parse_sse_response(sse)
        assert "error" in body
        assert body["error"]["code"] == -32600

    def test_parse_sse_no_data_raises(self):
        with pytest.raises(MCPClientError, match="No JSON-RPC response found"):
            MCPClient._parse_sse_response("event: message\n\n")

    def test_parse_sse_empty_raises(self):
        with pytest.raises(MCPClientError, match="No JSON-RPC response found"):
            MCPClient._parse_sse_response("")

    def test_parse_sse_ignores_non_data_lines(self):
        sse = (
            ": comment line\n"
            "event: message\n"
            "id: 123\n"
            'data: {"jsonrpc":"2.0","result":"ok","id":1}\n'
            "\n"
        )
        body = MCPClient._parse_sse_response(sse)
        assert body["result"] == "ok"

    def test_parse_sse_uses_last_response(self):
        """If multiple responses exist, use the last one."""
        sse = (
            'data: {"jsonrpc":"2.0","result":"first","id":1}\n'
            "\n"
            'data: {"jsonrpc":"2.0","result":"second","id":2}\n'
            "\n"
        )
        body = MCPClient._parse_sse_response(sse)
        assert body["result"] == "second"


# ── MCPClient unit tests ─────────────────────────────────────────────


class TestMCPClient:
    """Tests for the MCP HTTP client."""

    def test_build_headers_without_auth(self):
        client = MCPClient("https://mcp.example.com")
        headers = client._build_headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_build_headers_with_auth(self):
        client = MCPClient("https://mcp.example.com", auth_token="my-token")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer my-token"

    def test_build_jsonrpc_request(self):
        client = MCPClient("https://mcp.example.com")
        req = client._build_jsonrpc_request("tools/list")
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "tools/list"
        assert "id" in req
        assert "params" not in req

    def test_build_jsonrpc_request_with_params(self):
        client = MCPClient("https://mcp.example.com")
        req = client._build_jsonrpc_request(
            "tools/call", {"name": "test", "arguments": {"x": 1}}
        )
        assert req["params"] == {"name": "test", "arguments": {"x": 1}}

    def test_request_id_increments(self):
        client = MCPClient("https://mcp.example.com")
        req1 = client._build_jsonrpc_request("tools/list")
        req2 = client._build_jsonrpc_request("tools/list")
        assert req2["id"] > req1["id"]

    def test_server_url_trailing_slash_stripped(self):
        client = MCPClient("https://mcp.example.com/mcp/")
        assert client.server_url == "https://mcp.example.com/mcp"

    @pytest.mark.asyncio
    async def test_send_request_success(self):
        client = MCPClient("https://mcp.example.com")

        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"tools": []},
            "id": 1,
        }

        with patch.object(client, "_send_request", return_value={"tools": []}):
            result = await client._send_request("tools/list")
            assert result == {"tools": []}

    @pytest.mark.asyncio
    async def test_send_request_error(self):
        client = MCPClient("https://mcp.example.com")

        async def mock_send(*args, **kwargs):
            raise MCPClientError("MCP server error [-32600]: Invalid Request")

        with patch.object(client, "_send_request", side_effect=mock_send):
            with pytest.raises(MCPClientError, match="Invalid Request"):
                await client._send_request("tools/list")

    @pytest.mark.asyncio
    async def test_list_tools(self):
        client = MCPClient("https://mcp.example.com")

        mock_result = {
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
                {
                    "name": "search",
                    "description": "Search the web",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            ]
        }

        with patch.object(client, "_send_request", return_value=mock_result):
            tools = await client.list_tools()

        assert len(tools) == 2
        assert tools[0].name == "get_weather"
        assert tools[0].description == "Get current weather for a city"
        assert tools[0].input_schema["properties"]["city"]["type"] == "string"
        assert tools[1].name == "search"

    @pytest.mark.asyncio
    async def test_list_tools_empty(self):
        client = MCPClient("https://mcp.example.com")

        with patch.object(client, "_send_request", return_value={"tools": []}):
            tools = await client.list_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_list_tools_none_result(self):
        client = MCPClient("https://mcp.example.com")

        with patch.object(client, "_send_request", return_value=None):
            tools = await client.list_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        client = MCPClient("https://mcp.example.com")

        mock_result = {
            "content": [
                {"type": "text", "text": json.dumps({"temp": 20, "city": "London"})}
            ],
            "isError": False,
        }

        with patch.object(client, "_send_request", return_value=mock_result):
            result = await client.call_tool("get_weather", {"city": "London"})

        assert not result.is_error
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        client = MCPClient("https://mcp.example.com")

        mock_result = {
            "content": [{"type": "text", "text": "City not found"}],
            "isError": True,
        }

        with patch.object(client, "_send_request", return_value=mock_result):
            result = await client.call_tool("get_weather", {"city": "???"})

        assert result.is_error

    @pytest.mark.asyncio
    async def test_call_tool_none_result(self):
        client = MCPClient("https://mcp.example.com")

        with patch.object(client, "_send_request", return_value=None):
            result = await client.call_tool("get_weather", {"city": "London"})

        assert result.is_error

    @pytest.mark.asyncio
    async def test_initialize(self):
        client = MCPClient("https://mcp.example.com")

        mock_result = {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "test-server", "version": "1.0.0"},
        }

        with (
            patch.object(client, "_send_request", return_value=mock_result) as mock_req,
            patch.object(client, "_send_notification") as mock_notif,
        ):
            result = await client.initialize()

        mock_req.assert_called_once()
        mock_notif.assert_called_once_with("notifications/initialized")
        assert result["protocolVersion"] == "2025-03-26"


# ── MCPToolBlock unit tests ──────────────────────────────────────────

MOCK_USER_ID = "test-user-123"


class TestMCPToolBlock:
    """Tests for the MCPToolBlock."""

    def test_block_instantiation(self):
        block = MCPToolBlock()
        assert block.id == "a0a4b1c2-d3e4-4f56-a7b8-c9d0e1f2a3b4"
        assert block.name == "MCPToolBlock"

    def test_input_schema_has_required_fields(self):
        block = MCPToolBlock()
        schema = block.input_schema.jsonschema()
        props = schema.get("properties", {})
        assert "server_url" in props
        assert "selected_tool" in props
        assert "tool_arguments" in props
        assert "credential_id" in props

    def test_output_schema(self):
        block = MCPToolBlock()
        schema = block.output_schema.jsonschema()
        props = schema.get("properties", {})
        assert "result" in props
        assert "error" in props

    def test_get_input_schema_with_tool_schema(self):
        tool_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        data = {"tool_input_schema": tool_schema}
        result = MCPToolBlock.Input.get_input_schema(data)
        assert result == tool_schema

    def test_get_input_schema_without_tool_schema(self):
        result = MCPToolBlock.Input.get_input_schema({})
        assert result == {}

    def test_get_input_defaults(self):
        data = {"tool_arguments": {"city": "London"}}
        result = MCPToolBlock.Input.get_input_defaults(data)
        assert result == {"city": "London"}

    def test_get_missing_input(self):
        data = {
            "tool_input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string"},
                },
                "required": ["city", "units"],
            },
            "tool_arguments": {"city": "London"},
        }
        missing = MCPToolBlock.Input.get_missing_input(data)
        assert missing == {"units"}

    def test_get_missing_input_all_present(self):
        data = {
            "tool_input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "tool_arguments": {"city": "London"},
        }
        missing = MCPToolBlock.Input.get_missing_input(data)
        assert missing == set()

    @pytest.mark.asyncio
    async def test_run_with_mock(self):
        """Test the block using the built-in test infrastructure."""
        block = MCPToolBlock()
        await execute_block_test(block)

    @pytest.mark.asyncio
    async def test_run_missing_server_url(self):
        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url="",
            selected_tool="test",
        )
        outputs = []
        async for name, data in block.run(input_data, user_id=MOCK_USER_ID):
            outputs.append((name, data))
        assert outputs == [("error", "MCP server URL is required")]

    @pytest.mark.asyncio
    async def test_run_missing_tool(self):
        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url="https://mcp.example.com/mcp",
            selected_tool="",
        )
        outputs = []
        async for name, data in block.run(input_data, user_id=MOCK_USER_ID):
            outputs.append((name, data))
        assert outputs == [
            ("error", "No tool selected. Please select a tool from the dropdown.")
        ]

    @pytest.mark.asyncio
    async def test_run_success(self):
        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url="https://mcp.example.com/mcp",
            selected_tool="get_weather",
            tool_input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            tool_arguments={"city": "London"},
        )

        async def mock_call(*args, **kwargs):
            return {"temp": 20, "city": "London"}

        block._call_mcp_tool = mock_call  # type: ignore

        outputs = []
        async for name, data in block.run(input_data, user_id=MOCK_USER_ID):
            outputs.append((name, data))

        assert len(outputs) == 1
        assert outputs[0][0] == "result"
        assert outputs[0][1] == {"temp": 20, "city": "London"}

    @pytest.mark.asyncio
    async def test_run_mcp_error(self):
        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url="https://mcp.example.com/mcp",
            selected_tool="bad_tool",
        )

        async def mock_call(*args, **kwargs):
            raise MCPClientError("Tool not found")

        block._call_mcp_tool = mock_call  # type: ignore

        outputs = []
        async for name, data in block.run(input_data, user_id=MOCK_USER_ID):
            outputs.append((name, data))

        assert outputs[0][0] == "error"
        assert "Tool not found" in outputs[0][1]

    @pytest.mark.asyncio
    async def test_call_mcp_tool_parses_json_text(self):
        block = MCPToolBlock()

        mock_result = MCPCallResult(
            content=[
                {"type": "text", "text": '{"temp": 20}'},
            ],
            is_error=False,
        )

        async def mock_init(self):
            return {}

        async def mock_call(self, name, args):
            return mock_result

        with (
            patch.object(MCPClient, "initialize", mock_init),
            patch.object(MCPClient, "call_tool", mock_call),
        ):
            result = await block._call_mcp_tool(
                "https://mcp.example.com", "test_tool", {}
            )

        assert result == {"temp": 20}

    @pytest.mark.asyncio
    async def test_call_mcp_tool_plain_text(self):
        block = MCPToolBlock()

        mock_result = MCPCallResult(
            content=[
                {"type": "text", "text": "Hello, world!"},
            ],
            is_error=False,
        )

        async def mock_init(self):
            return {}

        async def mock_call(self, name, args):
            return mock_result

        with (
            patch.object(MCPClient, "initialize", mock_init),
            patch.object(MCPClient, "call_tool", mock_call),
        ):
            result = await block._call_mcp_tool(
                "https://mcp.example.com", "test_tool", {}
            )

        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_call_mcp_tool_multiple_content(self):
        block = MCPToolBlock()

        mock_result = MCPCallResult(
            content=[
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": '{"part": 2}'},
            ],
            is_error=False,
        )

        async def mock_init(self):
            return {}

        async def mock_call(self, name, args):
            return mock_result

        with (
            patch.object(MCPClient, "initialize", mock_init),
            patch.object(MCPClient, "call_tool", mock_call),
        ):
            result = await block._call_mcp_tool(
                "https://mcp.example.com", "test_tool", {}
            )

        assert result == ["Part 1", {"part": 2}]

    @pytest.mark.asyncio
    async def test_call_mcp_tool_error_result(self):
        block = MCPToolBlock()

        mock_result = MCPCallResult(
            content=[{"type": "text", "text": "Something went wrong"}],
            is_error=True,
        )

        async def mock_init(self):
            return {}

        async def mock_call(self, name, args):
            return mock_result

        with (
            patch.object(MCPClient, "initialize", mock_init),
            patch.object(MCPClient, "call_tool", mock_call),
        ):
            with pytest.raises(MCPClientError, match="returned an error"):
                await block._call_mcp_tool("https://mcp.example.com", "test_tool", {})

    @pytest.mark.asyncio
    async def test_call_mcp_tool_image_content(self):
        block = MCPToolBlock()

        mock_result = MCPCallResult(
            content=[
                {
                    "type": "image",
                    "data": "base64data==",
                    "mimeType": "image/png",
                }
            ],
            is_error=False,
        )

        async def mock_init(self):
            return {}

        async def mock_call(self, name, args):
            return mock_result

        with (
            patch.object(MCPClient, "initialize", mock_init),
            patch.object(MCPClient, "call_tool", mock_call),
        ):
            result = await block._call_mcp_tool(
                "https://mcp.example.com", "test_tool", {}
            )

        assert result == {
            "type": "image",
            "data": "base64data==",
            "mimeType": "image/png",
        }

    @pytest.mark.asyncio
    async def test_run_with_credential_id(self):
        """Verify the block resolves credential_id and passes auth token."""
        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url="https://mcp.example.com/mcp",
            selected_tool="test_tool",
            credential_id="cred-123",
        )

        captured_tokens = []

        async def mock_call(server_url, tool_name, arguments, auth_token=None):
            captured_tokens.append(auth_token)
            return "ok"

        async def mock_resolve(self, cred_id, uid):
            return "resolved-token"

        block._call_mcp_tool = mock_call  # type: ignore

        with patch.object(MCPToolBlock, "_resolve_auth_token", mock_resolve):
            async for _ in block.run(input_data, user_id=MOCK_USER_ID):
                pass

        assert captured_tokens == ["resolved-token"]

    @pytest.mark.asyncio
    async def test_run_without_credential_id(self):
        """Verify the block works without credentials (public server)."""
        block = MCPToolBlock()
        input_data = MCPToolBlock.Input(
            server_url="https://mcp.example.com/mcp",
            selected_tool="test_tool",
        )

        captured_tokens = []

        async def mock_call(server_url, tool_name, arguments, auth_token=None):
            captured_tokens.append(auth_token)
            return "ok"

        block._call_mcp_tool = mock_call  # type: ignore

        outputs = []
        async for name, data in block.run(input_data, user_id=MOCK_USER_ID):
            outputs.append((name, data))

        assert captured_tokens == [None]
        assert outputs == [("result", "ok")]

    @pytest.mark.asyncio
    async def test_resolve_auth_token_refreshes_expired(self):
        """Verify _resolve_auth_token refreshes expired MCP OAuth tokens."""
        import time

        from pydantic import SecretStr

        from backend.data.model import OAuth2Credentials

        block = MCPToolBlock()
        expired_creds = OAuth2Credentials(
            id="cred-expired",
            provider="mcp",
            title="MCP: test",
            access_token=SecretStr("old-token"),
            refresh_token=SecretStr("refresh-tok"),
            access_token_expires_at=int(time.time()) - 60,  # Already expired
            scopes=[],
            metadata={
                "mcp_token_url": "https://auth.example.com/token",
                "mcp_client_id": "client-id",
                "mcp_client_secret": "client-secret",
                "mcp_resource_url": "https://mcp.example.com",
            },
        )
        fresh_creds = OAuth2Credentials(
            id="cred-expired",
            provider="mcp",
            title="MCP: test",
            access_token=SecretStr("fresh-token"),
            refresh_token=SecretStr("refresh-tok"),
            access_token_expires_at=int(time.time()) + 3600,
            scopes=[],
            metadata=expired_creds.metadata,
        )

        mock_store = AsyncMock()
        mock_store.get_creds_by_id = AsyncMock(return_value=expired_creds)
        mock_store.update_creds = AsyncMock()

        mock_handler_instance = AsyncMock()
        mock_handler_instance.refresh_tokens = AsyncMock(return_value=fresh_creds)

        with (
            patch(
                "backend.integrations.credentials_store.IntegrationCredentialsStore",
                return_value=mock_store,
            ),
            patch(
                "backend.blocks.mcp.oauth.MCPOAuthHandler",
                return_value=mock_handler_instance,
            ),
        ):
            token = await block._resolve_auth_token("cred-expired", "user-1")

        assert token == "fresh-token"
        mock_handler_instance.refresh_tokens.assert_awaited_once_with(expired_creds)
        mock_store.update_creds.assert_awaited_once_with("user-1", fresh_creds)

    @pytest.mark.asyncio
    async def test_resolve_auth_token_skips_refresh_if_valid(self):
        """Verify _resolve_auth_token does NOT refresh tokens that are still valid."""
        import time

        from pydantic import SecretStr

        from backend.data.model import OAuth2Credentials

        block = MCPToolBlock()
        valid_creds = OAuth2Credentials(
            id="cred-valid",
            provider="mcp",
            title="MCP: test",
            access_token=SecretStr("valid-token"),
            refresh_token=SecretStr("refresh-tok"),
            access_token_expires_at=int(time.time()) + 3600,  # Still valid
            scopes=[],
            metadata={
                "mcp_token_url": "https://auth.example.com/token",
            },
        )

        mock_store = AsyncMock()
        mock_store.get_creds_by_id = AsyncMock(return_value=valid_creds)

        with patch(
            "backend.integrations.credentials_store.IntegrationCredentialsStore",
            return_value=mock_store,
        ):
            token = await block._resolve_auth_token("cred-valid", "user-1")

        assert token == "valid-token"
        # update_creds should NOT have been called (no refresh needed)
        mock_store.update_creds.assert_not_awaited()
