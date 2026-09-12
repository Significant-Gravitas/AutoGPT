"""The seam for non-registry MCP tools refuses whenever the gate cannot decide."""

import json
from unittest.mock import AsyncMock, patch

from backend.copilot.gate import Decision
from backend.copilot.gate.mcp_seam import gate_non_registry_tool
from backend.copilot.model import ChatSession

_CHECK = "backend.copilot.gate.mcp_seam.check_action"


async def test_a_crashing_gate_refuses_rather_than_runs():
    session = ChatSession.new(user_id="u1", dry_run=False)
    with patch(_CHECK, new=AsyncMock(side_effect=RuntimeError("gate down"))):
        result = await gate_non_registry_tool(
            "write_file", {"path": "a"}, "u1", session
        )
    assert result is not None
    assert result["isError"] is True


async def test_a_parked_call_mounts_the_approval_card():
    session = ChatSession.new(user_id="u1", dry_run=False)
    parked = Decision(allowed=False, reason="needs you", review_id="r1")
    with patch(_CHECK, new=AsyncMock(return_value=parked)):
        result = await gate_non_registry_tool(
            "write_file", {"path": "a"}, "u1", session
        )
    assert result is not None
    payload = json.loads(result["content"][0]["text"])
    assert payload["review_id"] == "r1"
    assert payload["graph_exec_id"] == f"copilot-session-{session.session_id}"


async def test_an_allowed_call_proceeds():
    session = ChatSession.new(user_id="u1", dry_run=False)
    with patch(_CHECK, new=AsyncMock(return_value=Decision(allowed=True))):
        assert (
            await gate_non_registry_tool("read_file", {"p": 1}, "u1", session) is None
        )
