from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.gate.policy import Effect, effect_for
from backend.copilot.gate.subject import NO_OP
from backend.copilot.model import ChatSession

from .design_workflow_page import (
    DESIGN_WORKFLOW_PAGE_TOOL,
    DesignWorkflowPageTool,
    workflow_pages_disabled_tools,
)

_FLAG = "backend.copilot.tools.design_workflow_page.is_feature_enabled"


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
