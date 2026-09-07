"""``get_workspace_manager`` derives the expert scope from the executing
session placed in the context by the executor — never from arguments."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.context import get_workspace_manager, set_execution_context
from backend.copilot.model import ChatSession
from backend.data.workspace_scope import WorkspaceAccessDeniedError, WorkspaceScope


@pytest.fixture
def db():
    db = MagicMock()
    db.get_or_create_workspace = AsyncMock(return_value=MagicMock(id="ws-1"))
    db.resolve_expert_workspace_scope = AsyncMock(
        return_value=WorkspaceScope(
            expert_id="expert-a", session_ids=["older"], skill_names=["assigned"]
        )
    )
    with patch("backend.copilot.context.workspace_db", return_value=db):
        yield db
    set_execution_context(None, None)


async def test_expert_turn_gets_scoped_manager_including_current_session(db):
    session = ChatSession.new("user-1", dry_run=False, expert_id="expert-a")
    set_execution_context("user-1", session)
    manager = await get_workspace_manager("user-1", session.session_id)
    assert manager.scope is not None
    assert manager.scope.session_ids == ["older", session.session_id]
    db.resolve_expert_workspace_scope.assert_awaited_once_with("user-1", "expert-a")


async def test_scope_follows_the_executing_session_not_the_requested_one(db):
    session = ChatSession.new("user-1", dry_run=False, expert_id="expert-a")
    set_execution_context("user-1", session)
    manager = await get_workspace_manager("user-1", "some-other-session")
    assert manager.scope is not None
    assert manager.scope.expert_id == "expert-a"


async def test_personal_autopilot_turn_is_unrestricted(db):
    session = ChatSession.new("user-1", dry_run=False)
    set_execution_context("user-1", session)
    manager = await get_workspace_manager("user-1", session.session_id)
    assert manager.scope is None
    db.resolve_expert_workspace_scope.assert_not_awaited()


async def test_outside_a_turn_fails_closed_to_the_requested_session(db):
    set_execution_context(None, None)
    manager = await get_workspace_manager("user-1", "session-1")
    assert manager.scope is not None
    assert manager.scope.expert_id is None
    assert manager.scope.session_ids == ["session-1"]
    assert not manager.scope.allows_path("/sessions/other/file.txt")
    db.resolve_expert_workspace_scope.assert_not_awaited()


async def test_session_of_another_user_is_refused(db):
    session = ChatSession.new("someone-else", dry_run=False, expert_id="expert-a")
    set_execution_context("someone-else", session)
    with pytest.raises(WorkspaceAccessDeniedError):
        await get_workspace_manager("user-1", session.session_id)
