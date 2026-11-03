import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolRequest, CallToolRequestParams

from backend.blocks.mcp.block import MCPToolBlock
from backend.blocks.mcp.client import MCPClient
from backend.blocks.mcp.test_integration import _MCPTestServer
from backend.util.request import Requests

ARRAY_PAYLOAD = [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [[], [{"type": "text", "text": ""}]])
@pytest.mark.parametrize("payload", [{}, {"base64Content": "SGVsbG8="}, ARRAY_PAYLOAD])
async def test_structured_content_reaches_block_output(content, payload):
    result = await _call_block(
        {"content": content, "structuredContent": payload},
        "fetch_blob",
        {"path": "/me/drive/items/test/content"},
    )
    assert result == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "value"),
    [
        ("add", {"a": 1, "b": 2}, 3),
        ("names", {}, ["Alice", "Bob"]),
        ("greet", {}, "hello"),
    ],
)
async def test_fastmcp_typed_result_stays_a_plain_value(tool_name, arguments, value):
    sent = await _fastmcp_result(tool_name, arguments)
    assert sent["structuredContent"] == {"result": value}
    assert await _call_block(sent, tool_name, arguments) == value


@pytest.mark.asyncio
@pytest.mark.parametrize("with_text", [True, False])
@pytest.mark.parametrize(
    ("block", "part"),
    [
        (
            {"type": "image", "data": "iVBORw0K", "mimeType": "image/png"},
            {"type": "image", "data": "iVBORw0K", "mimeType": "image/png"},
        ),
        (
            {"type": "resource", "resource": {"uri": "file:///a.txt", "text": "hi"}},
            {"uri": "file:///a.txt", "text": "hi"},
        ),
    ],
)
async def test_structured_content_keeps_media_blocks(with_text, block, part):
    payload = {"width": 1, "height": 1}
    text = [{"type": "text", "text": json.dumps(payload)}] if with_text else []
    sent = {"content": [*text, block], "structuredContent": payload}
    assert await _call_block(sent, "take_screenshot", {}) == [payload, part]


@pytest.mark.asyncio
@pytest.mark.parametrize("era", ["legacy", "modern"])
@pytest.mark.parametrize(
    "payload", [{"statusCode": 200, "base64Content": "SGVsbG8="}, ARRAY_PAYLOAD]
)
async def test_structured_only_result_crosses_real_http_and_block(era, payload):
    server = _MCPTestServer(era=era).start()
    original_init = Requests.__init__

    def local_requests(self, *args, **kwargs):
        kwargs["trusted_origins"] = ["http://127.0.0.1"]
        original_init(self, *args, **kwargs)

    try:
        with (
            patch.object(Requests, "__init__", local_requests),
            patch(
                "backend.blocks.mcp.test_server._tool_result",
                return_value={"content": [], "structuredContent": payload},
            ),
        ):
            result = await MCPToolBlock()._call_mcp_tool(
                server.url, "fetch_blob", {"path": "/fixture"}
            )
        assert result == payload
    finally:
        server.stop()


async def _call_block(
    sent: dict[str, Any], tool_name: str, arguments: dict[str, Any]
) -> Any:
    """Run the MCP block against a server that answers ``tools/call`` with *sent*."""
    with (
        patch.object(MCPClient, "initialize", new_callable=AsyncMock),
        patch.object(MCPClient, "close", new_callable=AsyncMock),
        patch.object(
            MCPClient, "_send_request", new_callable=AsyncMock, return_value=sent
        ),
    ):
        return await MCPToolBlock()._call_mcp_tool(
            "https://mcp.example.com/mcp", tool_name, arguments
        )


async def _fastmcp_result(tool_name: str, arguments: dict[str, Any]) -> dict:
    """The ``tools/call`` result a FastMCP server sends, as JSON."""
    server = FastMCP("typed-returns")

    @server.tool()
    def add(a: int, b: int) -> int:
        return a + b

    @server.tool()
    def names() -> list[str]:
        return ["Alice", "Bob"]

    @server.tool()
    def greet() -> str:
        return "hello"

    handler = server._mcp_server.request_handlers[CallToolRequest]
    result = await handler(
        CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool_name, arguments=arguments),
        )
    )
    return result.root.model_dump(mode="json", by_alias=True, exclude_none=True)
