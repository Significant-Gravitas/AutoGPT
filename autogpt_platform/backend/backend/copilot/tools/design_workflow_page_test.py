from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.gate.policy import Effect, effect_for
from backend.copilot.gate.subject import NO_OP
from backend.copilot.model import ChatSession

from . import flag_disabled_tools, get_available_tools
from .design_workflow_page import (
    DESIGN_WORKFLOW_PAGE_TOOL,
    DesignWorkflowPageTool,
    workflow_pages_disabled_tools,
)

_FLAG = "backend.copilot.tools.design_workflow_page.is_feature_enabled"
_VISUAL_ANSWERS_FLAG = "backend.copilot.tools.render_ui.is_feature_enabled"


def _session(**metadata: object) -> ChatSession:
    session = ChatSession.new(user_id="user-1", dry_run=False)
    return session.model_copy(
        update={"metadata": session.metadata.model_copy(update=metadata)}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag_on,metadata,hidden",
    [
        (False, {}, True),
        (True, {}, False),
        (True, {"source_platform": "discord"}, True),
        (True, {"text_only_result": True}, True),
    ],
    ids=["flag off", "flag on", "chat platform", "text-only caller"],
)
async def test_the_tool_is_offered_only_where_its_card_is_seen(
    flag_on: bool, metadata: dict[str, object], hidden: bool
) -> None:
    with patch(_FLAG, AsyncMock(return_value=flag_on)):
        disabled = await workflow_pages_disabled_tools(_session(**metadata))

    assert (DESIGN_WORKFLOW_PAGE_TOOL in disabled) is hidden


@pytest.mark.asyncio
async def test_a_preview_asks_nothing_and_a_save_is_a_platform_edit() -> None:
    tool = DesignWorkflowPageTool()
    session = _session()

    preview = await tool.gate_subject("user-1", session, {"page": {}})
    save = await tool.gate_subject("user-1", session, {"page": {}, "save": True})

    assert preview == NO_OP
    assert save is None
    assert effect_for(DESIGN_WORKFLOW_PAGE_TOOL) is Effect.PLATFORM


@pytest.mark.asyncio
async def test_with_the_flag_off_a_turn_declares_exactly_what_dev_declares() -> None:
    """The page tool costs a flag-off turn nothing: its tool list is the one
    the registry gives without the tool."""
    session = _session()
    with (
        patch(_FLAG, AsyncMock(return_value=False)),
        patch(_VISUAL_ANSWERS_FLAG, AsyncMock(return_value=False)),
    ):
        hidden = await flag_disabled_tools(session)
    without_the_tool = get_available_tools(
        disabled_tools=hidden | {DESIGN_WORKFLOW_PAGE_TOOL}
    )

    assert get_available_tools(disabled_tools=hidden) == without_the_tool
    with (
        patch(_FLAG, AsyncMock(return_value=True)),
        patch(_VISUAL_ANSWERS_FLAG, AsyncMock(return_value=False)),
    ):
        hidden_flag_on = await flag_disabled_tools(session)
    names = {
        t["function"]["name"]
        for t in get_available_tools(disabled_tools=hidden_flag_on)
    }
    assert DESIGN_WORKFLOW_PAGE_TOOL in names
