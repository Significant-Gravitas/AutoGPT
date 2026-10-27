from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.mcp.block import MCPToolBlock
from backend.blocks.mcp.client import MCPClient
from backend.blocks.mcp.test_integration import _MCPTestServer
from backend.util.request import Requests


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"base64Content": "SGVsbG8="}])
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
@pytest.mark.parametrize("era", ["legacy", "modern"])
async def test_structured_only_file_crosses_real_http_and_block(era):
    payload = {"statusCode": 200, "base64Content": "SGVsbG8="}
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
