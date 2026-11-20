from unittest.mock import AsyncMock

import pytest

from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.ranking import ConnectionState
from backend.copilot.capabilities.registry import build_entries
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tools import TOOL_GROUPS, TOOL_REGISTRY, chat_platform

from ._test_data import make_session
from .chat_platform_hint import POST_TOOL, ChatPlatformLinks
from .find_capability import FindCapabilityTool
from .models import CapabilityListResponse

USER = "kind-filter-user"


@pytest.fixture(scope="module")
def index() -> CapabilityIndex:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(chat_platform, "_any_chat_platform_configured", lambda: True)
        return CapabilityIndex(build_entries(TOOL_REGISTRY, TOOL_GROUPS))


@pytest.fixture
def links(index, monkeypatch):
    target = "backend.copilot.tools.find_capability"
    monkeypatch.setattr(f"{target}.session_registry", AsyncMock(return_value=index))
    monkeypatch.setattr(
        f"{target}.load_connection_state", AsyncMock(return_value=ConnectionState())
    )
    reader = AsyncMock(
        return_value=ChatPlatformLinks(
            servers=frozenset({"discord", "slack", "telegram"}),
            dms=frozenset({"teams"}),
        )
    )
    monkeypatch.setattr(f"{target}.chat_platform_links", reader)
    return reader


@pytest.mark.parametrize("expert_id", [None, "kind-filter-expert"])
@pytest.mark.parametrize(
    "query",
    ["send a discord message", "post a report to slack", "send a telegram message"],
)
async def test_hidden_chat_tool_keeps_the_users_platform_links(links, query, expert_id):
    result = await FindCapabilityTool()._execute(
        USER, make_session(USER, expert_id=expert_id), query=query, kind="block"
    )

    assert isinstance(result, CapabilityListResponse)
    assert result.capabilities and all(
        c["kind"] == "block" for c in result.capabilities
    )
    tool = next(c for c in result.other_kinds if c["id"] == f"tool:{POST_TOOL}")
    assert tool["linked"] == ["discord", "slack", "telegram"]
    assert tool["dm_linked"] == ["teams"]
    assert "search again without kind" in result.message
    links.assert_awaited_once_with(USER)


@pytest.mark.parametrize(
    "context,permissions",
    [("graph", None), ("direct", CopilotPermissions(tools=[POST_TOOL]))],
)
async def test_hidden_chat_tool_still_respects_context_and_permissions(
    links, monkeypatch, context, permissions
):
    monkeypatch.setattr(
        "backend.copilot.tools.find_capability.get_current_permissions",
        lambda: permissions,
    )
    result = await FindCapabilityTool()._execute(
        USER,
        make_session(USER),
        query="send a discord message",
        kind="block",
        context=context,
    )

    assert isinstance(result, CapabilityListResponse)
    assert all(c["id"] != f"tool:{POST_TOOL}" for c in result.other_kinds)
    links.assert_not_awaited()
