"""Workspace tools cannot reach outside an expert session's scope through
``include_all_sessions``, explicit ``/sessions/<id>/`` paths, or file IDs."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.workspace_files import (
    DeleteWorkspaceFileTool,
    ListWorkspaceFilesTool,
    ReadWorkspaceFileTool,
    WorkspaceFileListResponse,
    WriteWorkspaceFileTool,
)
from backend.data.workspace_scope import WorkspaceScope
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_test import _make_workspace_file

SCOPE = WorkspaceScope(expert_id="expert-a", session_ids=["expert-a"])


@pytest.fixture
def db():
    db = MagicMock()
    db.get_workspace_file = AsyncMock(
        return_value=_make_workspace_file(path="/sessions/expert-b/private.txt")
    )
    db.get_workspace_file_by_path = AsyncMock(return_value=_make_workspace_file())
    db.list_workspace_files = AsyncMock(return_value=[])
    db.count_workspace_files = AsyncMock(return_value=0)
    db.soft_delete_workspace_file = AsyncMock()
    manager = WorkspaceManager("user-1", "ws-1", "expert-a", scope=SCOPE)
    with (
        patch("backend.util.workspace.workspace_db", return_value=db),
        patch(
            "backend.copilot.tools.workspace_files.get_workspace_manager",
            new=AsyncMock(return_value=manager),
        ),
    ):
        yield db


def _session() -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False, expert_id="expert-a")
    session.session_id = "expert-a"
    return session


async def test_include_all_sessions_is_confined_to_the_scope(db):
    result = await ListWorkspaceFilesTool()._execute(
        "user-1", _session(), include_all_sessions=True
    )
    assert isinstance(result, WorkspaceFileListResponse)
    assert (
        db.list_workspace_files.call_args.kwargs["allowed_path_prefixes"]
        == SCOPE.read_prefixes
    )


async def test_read_by_foreign_file_id_is_denied(db):
    result = await ReadWorkspaceFileTool()._execute(
        "user-1", _session(), file_id="foreign"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "access_denied"


async def test_read_by_explicit_foreign_path_is_denied(db):
    result = await ReadWorkspaceFileTool()._execute(
        "user-1", _session(), path="/sessions/expert-b/private.txt"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "access_denied"
    db.get_workspace_file_by_path.assert_not_awaited()


async def test_write_into_foreign_session_is_denied(db):
    result = await WriteWorkspaceFileTool()._execute(
        "user-1",
        _session(),
        filename="x.txt",
        content="x",
        path="/sessions/expert-b/x.txt",
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "access_denied"


async def test_delete_by_foreign_file_id_is_denied(db):
    result = await DeleteWorkspaceFileTool()._execute(
        "user-1", _session(), file_id="foreign"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "access_denied"
    db.soft_delete_workspace_file.assert_not_awaited()
