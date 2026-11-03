from unittest.mock import AsyncMock

import pytest

from backend.blocks.mcp.client import MCPCallResult
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import MCPToolOutputResponse
from backend.copilot.tools.run_mcp_tool import RunMCPToolTool


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "payload", [{"base64Content": "SGVsbG8="}, [{"id": "1", "name": "Alice"}]]
)
async def test_copilot_receives_structured_only_result(payload):
    client = AsyncMock()
    client.call_tool.return_value = MCPCallResult(structured_content=payload)
    session = make_session("test-user")
    result = await RunMCPToolTool()._execute_tool(
        client,
        "https://workiq.svc.cloud.microsoft/mcp",
        "fetch_blob",
        {"path": "/fixture"},
        session.session_id,
        "test-user",
        session,
    )
    assert isinstance(result, MCPToolOutputResponse)
    assert result.result == payload
