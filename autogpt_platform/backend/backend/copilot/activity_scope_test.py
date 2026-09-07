from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.db import get_session_titles
from backend.copilot.model import ChatSession
from backend.copilot.tools.base import _record_activity
from backend.copilot.tools.models import ErrorResponse
from backend.data.activity_event import ActivityEventDraft
from backend.data.execution import ExecutionStatus
from backend.executor.activity_events import (
    handle_run_completed,
    log_node_integration_activity,
)


@pytest.mark.asyncio
async def test_tool_activity_uses_session_scope_even_if_draft_claims_another(mocker):
    session = ChatSession.new(user_id="actor", dry_run=False)
    session.organization_id = "org-live"
    session.team_id = "team-live"
    draft = ActivityEventDraft(
        category="FILE",
        event_type="file.created",
        title="report.md",
        organization_id="org-other",
        team_id="team-other",
    )
    tool = MagicMock(activity_event=MagicMock(return_value=draft))
    create = AsyncMock()
    mocker.patch(
        "backend.copilot.tools.base.activity_event_db",
        return_value=MagicMock(create_activity_event=create),
    )

    await _record_activity(tool, "actor", session, ErrorResponse(message="unused"), {})

    persisted = create.await_args.kwargs["draft"]
    assert create.await_args.kwargs["user_id"] == "actor"
    assert persisted.organization_id == "org-live"
    assert persisted.team_id == "team-live"


@pytest.mark.asyncio
async def test_session_titles_cannot_follow_an_event_into_another_workspace(mocker):
    lookup = AsyncMock(return_value=[])
    mocker.patch(
        "backend.copilot.db.PrismaChatSession.prisma",
        return_value=MagicMock(find_many=lookup),
    )

    assert (
        await get_session_titles(
            "actor", ["session-1"], organization_id="org-live", team_id="team-live"
        )
        == {}
    )

    assert lookup.await_args.kwargs["where"] == {
        "id": {"in": ["session-1"]},
        "userId": "actor",
        "organizationId": "org-live",
        "teamId": "team-live",
    }


@pytest.mark.asyncio
async def test_node_activity_uses_execution_context_scope():
    node = MagicMock(
        user_id="actor",
        graph_exec_id="run-1",
        node_exec_id="node-run-1",
        block_id="block-1",
        inputs={"credentials": {"id": "cred", "provider": "github"}},
        execution_context=MagicMock(
            dry_run=False, organization_id="org-live", team_id="team-live"
        ),
    )
    block = MagicMock()
    block.name = "Create issue"
    block.input_schema.get_credentials_fields.return_value = ["credentials"]
    create = AsyncMock()

    await log_node_integration_activity(
        node, block, MagicMock(create_activity_event=create)
    )

    draft = create.await_args.kwargs["draft"]
    assert draft.organization_id == "org-live"
    assert draft.team_id == "team-live"


def test_run_activity_uses_persisted_execution_scope():
    create = MagicMock()
    graph = MagicMock(user_id="actor", graph_exec_id="run-1", graph_id="graph-1")
    meta = MagicMock(
        status=ExecutionStatus.COMPLETED,
        expert_id=None,
        organization_id="org-live",
        team_id="team-live",
    )
    stats = MagicMock(is_dry_run=False, activity_status="Report created")

    handle_run_completed(MagicMock(create_activity_event=create), graph, meta, stats)

    draft = create.call_args.kwargs["draft"]
    assert draft.organization_id == "org-live"
    assert draft.team_id == "team-live"
