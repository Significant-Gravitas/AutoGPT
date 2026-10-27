import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.mcp.block import MCPToolBlock
from backend.blocks.mcp.client import MCPClient
from backend.blocks.mcp.test_integration import _MCPTestServer
from backend.util.request import Requests

ARRAY_PAYLOAD = [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"base64Content": "SGVsbG8="}, ARRAY_PAYLOAD])
async def test_structured_content_reaches_block_output(payload):
    with (
        patch.object(MCPClient, "initialize", new_callable=AsyncMock),
        patch.object(MCPClient, "close", new_callable=AsyncMock),
        patch.object(
            MCPClient,
            "_send_request",
            new_callable=AsyncMock,
            return_value={"content": [], "structuredContent": payload},
        ),
    ):
        result = await MCPToolBlock()._call_mcp_tool(
            "https://workiq.svc.cloud.microsoft/mcp",
            "fetch_blob",
            {"path": "/me/drive/items/test/content"},
        )
    assert result == payload


@pytest.mark.asyncio
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
async def test_structured_content_keeps_media_blocks(block, part):
    payload = {"width": 1, "height": 1}
    with (
        patch.object(MCPClient, "initialize", new_callable=AsyncMock),
        patch.object(MCPClient, "close", new_callable=AsyncMock),
        patch.object(
            MCPClient,
            "_send_request",
            new_callable=AsyncMock,
            return_value={
                "content": [{"type": "text", "text": json.dumps(payload)}, block],
                "structuredContent": payload,
            },
        ),
    ):
        result = await MCPToolBlock()._call_mcp_tool(
            "https://mcp.example.com/mcp", "take_screenshot", {}
        )
    assert result == [payload, part]


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
