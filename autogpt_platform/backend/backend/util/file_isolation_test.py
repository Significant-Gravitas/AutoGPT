"""A ``workspace://`` block input is a second door into the workspace: it must
be held to the same expert grants as the session's own file tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.context import set_execution_context
from backend.copilot.model import ChatSession
from backend.data.execution import ExecutionContext
from backend.data.workspace_scope import WorkspaceAccessDeniedError, WorkspaceScope
from backend.util.file import store_media_file
from backend.util.type import MediaFileType
from backend.util.workspace_test import _make_workspace_file

OWN_SESSION = "expert-a-session"
FOREIGN_PATH = "/sessions/personal/private.txt"


@pytest.fixture
def scope_db():
    db = MagicMock()
    db.resolve_expert_workspace_scope = AsyncMock(
        return_value=WorkspaceScope(expert_id="expert-a", session_ids=[OWN_SESSION])
    )
    with patch("backend.copilot.context.workspace_db", return_value=db):
        yield db
    set_execution_context(None, None)


@pytest.fixture
def workspace_db_mock():
    db = MagicMock()
    db.get_workspace_file = AsyncMock(
        return_value=_make_workspace_file(path=FOREIGN_PATH)
    )
    db.get_workspace_file_by_path = AsyncMock(
        return_value=_make_workspace_file(path=FOREIGN_PATH)
    )
    storage = AsyncMock()
    storage.retrieve.return_value = b"secret"
    with (
        patch("backend.util.workspace.workspace_db", return_value=db),
        patch("backend.util.workspace.get_workspace_storage", return_value=storage),
        patch("backend.util.file.get_cloud_storage_handler", AsyncMock()),
        patch("backend.util.file.scan_content_safe", AsyncMock()),
    ):
        yield db, storage


def _context(session_id: str) -> ExecutionContext:
    return ExecutionContext(
        user_id="user-1",
        graph_exec_id="copilot-session-block",
        workspace_id="ws-1",
        session_id=session_id,
    )


def _start_expert_turn() -> None:
    session = ChatSession.new("user-1", dry_run=False, expert_id="expert-a")
    session.session_id = OWN_SESSION
    set_execution_context("user-1", session)


@pytest.mark.parametrize("ref", [f"workspace://{FOREIGN_PATH}", "workspace://file-id"])
async def test_expert_block_input_cannot_leave_the_experts_scope(
    scope_db, workspace_db_mock, ref: str
):
    _, storage = workspace_db_mock
    _start_expert_turn()
    with pytest.raises(WorkspaceAccessDeniedError):
        await store_media_file(
            MediaFileType(ref),
            _context(OWN_SESSION),
            return_format="for_external_api",
        )
    storage.retrieve.assert_not_awaited()


async def test_expert_block_input_reads_its_own_session_file(
    scope_db, workspace_db_mock
):
    db, _ = workspace_db_mock
    own = f"/sessions/{OWN_SESSION}/notes.txt"
    db.get_workspace_file_by_path.return_value = _make_workspace_file(path=own)
    _start_expert_turn()
    result = await store_media_file(
        MediaFileType(f"workspace://{own}"),
        _context(OWN_SESSION),
        return_format="for_external_api",
    )
    assert result.startswith("data:")


async def test_personal_autopilot_block_input_stays_unrestricted(
    scope_db, workspace_db_mock
):
    set_execution_context("user-1", ChatSession.new("user-1", dry_run=False))
    result = await store_media_file(
        MediaFileType(f"workspace://{FOREIGN_PATH}"),
        _context("personal"),
        return_format="for_external_api",
    )
    assert result.startswith("data:")
    scope_db.resolve_expert_workspace_scope.assert_not_awaited()
