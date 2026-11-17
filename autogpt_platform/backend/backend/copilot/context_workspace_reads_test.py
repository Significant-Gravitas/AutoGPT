"""An expert's workspace and its file scope are read side by side, and a session
of another user still gets no workspace manager."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.copilot import context
from backend.copilot.context import set_execution_context
from backend.data.workspace_scope import WorkspaceAccessDeniedError, WorkspaceScope
from backend.util.testing import Rendezvous


def _expert_session(user_id: str) -> MagicMock:
    return MagicMock(user_id=user_id, expert_id="expert-1", session_id="session-1")


@pytest.mark.asyncio
async def test_an_experts_workspace_and_its_scope_are_read_side_by_side(
    mocker: MockerFixture,
) -> None:
    meeting = Rendezvous(2)
    workspace_db = SimpleNamespace(
        get_or_create_workspace=AsyncMock(
            side_effect=meeting.returning(SimpleNamespace(id="workspace-1"))
        ),
        resolve_expert_workspace_scope=AsyncMock(
            side_effect=meeting.returning(
                WorkspaceScope(expert_id="expert-1", session_ids=["earlier"])
            )
        ),
    )
    mocker.patch.object(context, "workspace_db", return_value=workspace_db)
    set_execution_context("user-1", _expert_session("user-1"))

    manager = await context.get_workspace_manager("user-1", "session-1")

    assert manager.workspace_id == "workspace-1"
    assert manager.scope is not None
    assert manager.scope.session_ids == ["earlier", "session-1"]
    workspace_db.resolve_expert_workspace_scope.assert_awaited_once_with(
        "user-1", "expert-1"
    )


@pytest.mark.asyncio
async def test_a_session_of_another_user_still_gets_no_workspace_manager(
    mocker: MockerFixture,
) -> None:
    workspace_db = SimpleNamespace(
        get_or_create_workspace=AsyncMock(return_value=SimpleNamespace(id="ws")),
        resolve_expert_workspace_scope=AsyncMock(),
    )
    mocker.patch.object(context, "workspace_db", return_value=workspace_db)
    set_execution_context("user-1", _expert_session("someone-else"))

    with pytest.raises(WorkspaceAccessDeniedError):
        await context.get_workspace_manager("user-1", "session-1")
    workspace_db.resolve_expert_workspace_scope.assert_not_awaited()
