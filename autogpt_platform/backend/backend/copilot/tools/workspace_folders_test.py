"""Tests for the workspace folder CoPilot tools."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library.exceptions import FolderAlreadyExistsError
from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.workspace_folders import (
    CreateWorkspaceFolderTool,
    DeleteWorkspaceFolderTool,
    ListWorkspaceFoldersTool,
    MoveWorkspaceFilesToFolderTool,
    WorkspaceFilesMovedToFolderResponse,
    WorkspaceFolderCreatedResponse,
    WorkspaceFolderDeletedResponse,
    WorkspaceFolderListResponse,
)
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_folder import WorkspaceFolder

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
_USER = "user-123"


def _session() -> ChatSession:
    now = datetime.now(timezone.utc)
    return ChatSession(
        session_id=str(uuid.uuid4()),
        user_id=_USER,
        messages=[],
        usage=[],
        started_at=now,
        updated_at=now,
        successful_agent_runs={},
        successful_agent_schedules={},
    )


def _folder(
    id: str = "fld-1", name: str = "Invoices", file_count: int = 0
) -> WorkspaceFolder:
    return WorkspaceFolder(
        id=id,
        workspace_id="ws-123",
        name=name,
        icon=None,
        created_at=_NOW,
        updated_at=_NOW,
        file_count=file_count,
    )


def _file(id: str = "file-1") -> WorkspaceFile:
    return WorkspaceFile(
        id=id,
        workspace_id="ws-123",
        name="report.pdf",
        path="/report.pdf",
        storage_path=f"local://ws-123/{id}/report.pdf",
        mime_type="application/pdf",
        size_bytes=10,
        checksum="abc",
        metadata={},
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.fixture
def mock_folder_db():
    db = MagicMock()
    db.list_workspace_folders = AsyncMock(return_value=[])
    db.create_workspace_folder = AsyncMock(return_value=_folder())
    db.delete_workspace_folder = AsyncMock(return_value=None)
    db.bulk_move_files_to_folder = AsyncMock(return_value=[])
    return db


@pytest.fixture
def patched(mock_folder_db):
    workspace_db = MagicMock()
    workspace_db.get_or_create_workspace = AsyncMock(
        return_value=MagicMock(id="ws-123")
    )
    with (
        patch(
            "backend.copilot.tools.workspace_folders.workspace_folder_db",
            return_value=mock_folder_db,
        ),
        patch(
            "backend.copilot.tools.workspace_folders.workspace_db",
            return_value=workspace_db,
        ),
    ):
        yield mock_folder_db


# ---------------------------------------------------------------------------
# create_workspace_folder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_folder_returns_folder_info(patched):
    resp = await CreateWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="Invoices"
    )

    assert isinstance(resp, WorkspaceFolderCreatedResponse), resp.message
    assert resp.folder.name == "Invoices"
    assert resp.folder.folder_id == "fld-1"
    patched.create_workspace_folder.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_folder_requires_a_name(patched):
    resp = await CreateWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="   "
    )
    assert isinstance(resp, ErrorResponse)
    assert "folder name" in resp.message


@pytest.mark.asyncio
async def test_create_folder_rejects_overlong_name(patched):
    resp = await CreateWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="x" * 101
    )
    assert isinstance(resp, ErrorResponse)
    assert "too long" in resp.message
    patched.create_workspace_folder.assert_not_called()


@pytest.mark.asyncio
async def test_create_folder_duplicate_name_is_actionable(patched):
    patched.create_workspace_folder.side_effect = FolderAlreadyExistsError("dup")

    resp = await CreateWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="Invoices"
    )
    assert isinstance(resp, ErrorResponse)
    assert "already exists" in resp.message
    assert "list_workspace_folders" in resp.message


# ---------------------------------------------------------------------------
# delete_workspace_folder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_folder_by_name_reports_files_kept(patched):
    patched.list_workspace_folders.return_value = [_folder(file_count=3)]

    resp = await DeleteWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="Invoices"
    )

    assert isinstance(resp, WorkspaceFolderDeletedResponse), resp.message
    assert resp.files_moved_to_root == 3
    assert "moved to the workspace root" in resp.message
    patched.delete_workspace_folder.assert_awaited_once_with(
        folder_id="fld-1", workspace_id="ws-123"
    )


@pytest.mark.asyncio
async def test_delete_folder_name_lookup_is_case_insensitive(patched):
    patched.list_workspace_folders.return_value = [_folder()]

    resp = await DeleteWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="invoices"
    )
    assert isinstance(resp, WorkspaceFolderDeletedResponse), resp.message


@pytest.mark.asyncio
async def test_delete_folder_requires_an_identifier(patched):
    resp = await DeleteWorkspaceFolderTool()._execute(user_id=_USER, session=_session())
    assert isinstance(resp, ErrorResponse)
    assert "folder_id or name" in resp.message


@pytest.mark.asyncio
async def test_delete_folder_unknown_folder_errors(patched):
    patched.list_workspace_folders.return_value = []

    resp = await DeleteWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), folder_id="missing"
    )
    assert isinstance(resp, ErrorResponse)
    assert "not found" in resp.message
    patched.delete_workspace_folder.assert_not_called()


# ---------------------------------------------------------------------------
# list_workspace_folders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_folders_summarizes_counts(patched):
    patched.list_workspace_folders.return_value = [
        _folder(id="fld-1", name="Invoices", file_count=2),
        _folder(id="fld-2", name="Reports", file_count=0),
    ]

    resp = await ListWorkspaceFoldersTool()._execute(user_id=_USER, session=_session())

    assert isinstance(resp, WorkspaceFolderListResponse), resp.message
    assert [f.name for f in resp.folders] == ["Invoices", "Reports"]
    assert "Invoices (2 file(s)" in resp.message


@pytest.mark.asyncio
async def test_list_folders_empty_workspace(patched):
    resp = await ListWorkspaceFoldersTool()._execute(user_id=_USER, session=_session())
    assert isinstance(resp, WorkspaceFolderListResponse)
    assert resp.folders == []
    assert "No workspace folders yet" in resp.message


# ---------------------------------------------------------------------------
# move_workspace_files_to_folder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_files_into_folder_by_name(patched):
    patched.list_workspace_folders.return_value = [_folder()]
    patched.bulk_move_files_to_folder.return_value = [_file("file-1"), _file("file-2")]

    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER,
        session=_session(),
        file_ids=["file-1", "file-2"],
        folder_name="Invoices",
    )

    assert isinstance(resp, WorkspaceFilesMovedToFolderResponse), resp.message
    assert resp.folder_id == "fld-1"
    assert resp.file_ids == ["file-1", "file-2"]
    assert patched.bulk_move_files_to_folder.await_args.kwargs["folder_id"] == "fld-1"


@pytest.mark.asyncio
async def test_move_files_to_root(patched):
    patched.bulk_move_files_to_folder.return_value = [_file("file-1")]

    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER, session=_session(), file_ids=["file-1"], to_root=True
    )

    assert isinstance(resp, WorkspaceFilesMovedToFolderResponse), resp.message
    assert resp.folder_id is None
    assert patched.bulk_move_files_to_folder.await_args.kwargs["folder_id"] is None
    assert "workspace root" in resp.message


@pytest.mark.asyncio
async def test_move_files_reports_ids_dropped_as_out_of_workspace(patched):
    """bulk_move silently drops foreign IDs — say so rather than claiming success."""
    patched.list_workspace_folders.return_value = [_folder()]
    patched.bulk_move_files_to_folder.return_value = [_file("file-1")]

    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER,
        session=_session(),
        file_ids=["file-1", "someone-elses-file"],
        folder_id="fld-1",
    )

    assert isinstance(resp, WorkspaceFilesMovedToFolderResponse)
    assert "1 file ID(s) were not found" in resp.message


@pytest.mark.asyncio
async def test_move_files_unknown_folder_errors(patched):
    patched.list_workspace_folders.return_value = []

    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER,
        session=_session(),
        file_ids=["file-1"],
        folder_name="Nope",
    )
    assert isinstance(resp, ErrorResponse)
    assert "create_workspace_folder" in resp.message
    patched.bulk_move_files_to_folder.assert_not_called()


@pytest.mark.asyncio
async def test_move_files_requires_file_ids(patched):
    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER, session=_session(), file_ids=[]
    )
    assert isinstance(resp, ErrorResponse)
    assert "file_ids" in resp.message


@pytest.mark.asyncio
async def test_move_files_rejects_ambiguous_destination(patched):
    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER,
        session=_session(),
        file_ids=["file-1"],
        folder_id="fld-1",
        to_root=True,
    )
    assert isinstance(resp, ErrorResponse)
    assert "not both" in resp.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [
        CreateWorkspaceFolderTool(),
        DeleteWorkspaceFolderTool(),
        ListWorkspaceFoldersTool(),
        MoveWorkspaceFilesToFolderTool(),
    ],
)
async def test_all_folder_tools_require_authentication(tool, patched):
    resp = await tool._execute(
        user_id=None, session=_session(), name="X", file_ids=["file-1"]
    )
    assert isinstance(resp, ErrorResponse)
    assert "Authentication required" in resp.message
