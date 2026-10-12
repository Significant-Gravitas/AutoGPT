"""Tests for the workspace folder CoPilot tools."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library.exceptions import FolderAlreadyExistsError
from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.workspace_folders import (
    CreateWorkspaceFolderTool,
    DeleteWorkspaceFolderTool,
    MoveWorkspaceFilesToFolderTool,
    WorkspaceFilesMovedToFolderResponse,
    WorkspaceFolderCreatedResponse,
    WorkspaceFolderDeletedResponse,
)
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_folder import WorkspaceFolder
from backend.data.workspace_scope import WorkspaceScope
from backend.util.exceptions import NotFoundError
from backend.util.workspace import WorkspaceManager

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_USER = "user-123"


def _session() -> ChatSession:
    return ChatSession.new(_USER, dry_run=False)


def _folder(
    id: str = "fld-1",
    name: str = "Invoices",
    parent_id: str | None = None,
    file_count: int = 0,
) -> WorkspaceFolder:
    return WorkspaceFolder(
        id=id,
        workspace_id="ws-123",
        name=name,
        parent_id=parent_id,
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
    db.get_workspace_folder = AsyncMock(return_value=_folder())
    db.create_workspace_folder = AsyncMock(return_value=_folder())
    db.delete_workspace_folder = AsyncMock(return_value=0)
    db.bulk_move_files_to_folder = AsyncMock(return_value=[])
    return db


def _patch(mock_folder_db, scope: WorkspaceScope | None = None):
    manager = WorkspaceManager(_USER, "ws-123", "sess-1", scope=scope)
    return (
        patch(
            "backend.copilot.tools.workspace_folders.workspace_folder_db",
            return_value=mock_folder_db,
        ),
        patch(
            "backend.copilot.tools.workspace_folders.get_workspace_manager",
            AsyncMock(return_value=manager),
        ),
    )


@pytest.fixture
def patched(mock_folder_db):
    folder_db, manager = _patch(mock_folder_db)
    with folder_db, manager:
        yield mock_folder_db


@pytest.fixture
def as_expert(mock_folder_db):
    scope = WorkspaceScope(
        expert_id="exp-1", session_ids=["sess-1"], reads_user_files=True
    )
    folder_db, manager = _patch(mock_folder_db, scope)
    with folder_db, manager:
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
    assert (resp.folder.name, resp.folder.folder_id) == ("Invoices", "fld-1")
    patched.create_workspace_folder.assert_awaited_once_with(
        workspace_id="ws-123", name="Invoices", parent_id=None
    )


@pytest.mark.asyncio
async def test_create_folder_inside_a_parent(patched):
    patched.list_workspace_folders.return_value = [_folder("parent", "Invoices")]
    patched.create_workspace_folder.return_value = _folder(
        "child", "2026", parent_id="parent"
    )

    resp = await CreateWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="2026", parent_id="parent"
    )

    assert isinstance(resp, WorkspaceFolderCreatedResponse), resp.message
    assert resp.folder.parent_id == "parent"
    assert patched.create_workspace_folder.await_args.kwargs["parent_id"] == "parent"


@pytest.mark.asyncio
async def test_create_folder_under_an_unknown_parent_errors(patched):
    resp = await CreateWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="2026", parent_id="missing"
    )

    assert isinstance(resp, ErrorResponse)
    assert "Parent workspace folder not found" in resp.message
    patched.create_workspace_folder.assert_not_called()


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
async def test_create_folder_duplicate_sibling_is_refused_before_the_write(patched):
    """The check runs here because the DB's duplicate error is lost over RPC."""
    patched.list_workspace_folders.return_value = [_folder("fld-1", "Invoices")]

    resp = await CreateWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="Invoices"
    )

    assert isinstance(resp, ErrorResponse)
    assert "already exists" in resp.message
    assert "tool:list_workspace_files" in resp.message
    patched.create_workspace_folder.assert_not_called()


@pytest.mark.asyncio
async def test_create_folder_same_name_under_another_parent_is_allowed(patched):
    patched.list_workspace_folders.return_value = [
        _folder("a", "Invoices"),
        _folder("b", "2026", parent_id="a"),
    ]

    resp = await CreateWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="2026"
    )

    assert isinstance(resp, WorkspaceFolderCreatedResponse), resp.message


@pytest.mark.asyncio
async def test_create_folder_duplicate_race_is_still_actionable(patched):
    patched.create_workspace_folder.side_effect = FolderAlreadyExistsError("dup")

    resp = await CreateWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="Invoices"
    )

    assert isinstance(resp, ErrorResponse)
    assert "already exists" in resp.message


# ---------------------------------------------------------------------------
# delete_workspace_folder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_folder_reports_its_whole_subtree(patched):
    patched.list_workspace_folders.return_value = [
        _folder("inv", "Invoices", file_count=2),
        _folder("2026", "2026", parent_id="inv", file_count=5),
        _folder("q1", "Q1", parent_id="2026", file_count=7),
        _folder("photos", "Photos", file_count=1),
    ]
    patched.delete_workspace_folder.return_value = 14

    resp = await DeleteWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="invoices"
    )

    assert isinstance(resp, WorkspaceFolderDeletedResponse), resp.message
    assert resp.files_moved_to_root == 14
    assert resp.subfolders_deleted == 2
    assert "moved to the workspace root" in resp.message
    patched.delete_workspace_folder.assert_awaited_once_with(
        folder_id="inv", workspace_id="ws-123"
    )


@pytest.mark.asyncio
async def test_delete_folder_by_id(patched):
    patched.list_workspace_folders.return_value = [_folder("fld-1", "Invoices")]

    resp = await DeleteWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), folder_id="fld-1"
    )

    assert isinstance(resp, WorkspaceFolderDeletedResponse), resp.message
    assert "(it was empty)" in resp.message


@pytest.mark.asyncio
async def test_delete_folder_refuses_an_ambiguous_name(patched):
    """Nested folders may share a name; deleting one of them by guess is not ok."""
    patched.list_workspace_folders.return_value = [
        _folder("a", "2026", parent_id="inv"),
        _folder("b", "2026", parent_id="receipts"),
    ]

    resp = await DeleteWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="2026"
    )

    assert isinstance(resp, ErrorResponse)
    assert "pass folder_id" in resp.message
    patched.delete_workspace_folder.assert_not_called()


@pytest.mark.asyncio
async def test_delete_folder_name_prefers_the_exact_case(patched):
    """Siblings may differ only by case; the exact spelling settles it."""
    patched.list_workspace_folders.return_value = [
        _folder("upper", "Invoices"),
        _folder("lower", "invoices"),
    ]

    resp = await DeleteWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="invoices"
    )

    assert isinstance(resp, WorkspaceFolderDeletedResponse), resp.message
    assert resp.folder_id == "lower"


@pytest.mark.asyncio
async def test_delete_folder_requires_an_identifier(patched):
    resp = await DeleteWorkspaceFolderTool()._execute(user_id=_USER, session=_session())

    assert isinstance(resp, ErrorResponse)
    assert "folder_id or name" in resp.message


@pytest.mark.asyncio
async def test_delete_folder_unknown_folder_errors(patched):
    resp = await DeleteWorkspaceFolderTool()._execute(
        user_id=_USER, session=_session(), name="Nope"
    )

    assert isinstance(resp, ErrorResponse)
    assert "not found" in resp.message
    patched.delete_workspace_folder.assert_not_called()


# ---------------------------------------------------------------------------
# move_workspace_files_to_folder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_files_into_folder_by_name(patched):
    patched.list_workspace_folders.return_value = [_folder("fld-1", "Invoices")]
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
    assert patched.bulk_move_files_to_folder.await_args.kwargs == {
        "workspace_id": "ws-123",
        "file_ids": ["file-1", "file-2"],
        "folder_id": "fld-1",
    }


@pytest.mark.asyncio
async def test_move_files_by_folder_id_looks_up_only_that_folder(patched):
    patched.bulk_move_files_to_folder.return_value = [_file("file-1")]

    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER, session=_session(), file_ids=["file-1"], folder_id="fld-1"
    )

    assert isinstance(resp, WorkspaceFilesMovedToFolderResponse), resp.message
    patched.get_workspace_folder.assert_awaited_once_with("fld-1", "ws-123")
    patched.list_workspace_folders.assert_not_called()


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
    patched.bulk_move_files_to_folder.return_value = [_file("file-1")]

    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER,
        session=_session(),
        file_ids=["file-1", "foreign"],
        folder_id="fld-1",
    )

    assert isinstance(resp, WorkspaceFilesMovedToFolderResponse)
    assert "1 file ID(s) were not found" in resp.message


@pytest.mark.asyncio
async def test_move_files_with_no_file_in_the_workspace_is_an_error(patched):
    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER, session=_session(), file_ids=["foreign"], folder_id="fld-1"
    )

    assert isinstance(resp, ErrorResponse)
    assert "No files were moved" in resp.message


@pytest.mark.asyncio
async def test_move_files_unknown_folder_id_errors(patched):
    patched.get_workspace_folder.side_effect = NotFoundError("nope")

    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER, session=_session(), file_ids=["file-1"], folder_id="gone"
    )

    assert isinstance(resp, ErrorResponse)
    assert "tool:create_workspace_folder" in resp.message
    patched.bulk_move_files_to_folder.assert_not_called()


@pytest.mark.asyncio
async def test_move_files_unknown_folder_name_errors(patched):
    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER, session=_session(), file_ids=["file-1"], folder_name="Nope"
    )

    assert isinstance(resp, ErrorResponse)
    assert "tool:create_workspace_folder" in resp.message
    patched.bulk_move_files_to_folder.assert_not_called()


@pytest.mark.asyncio
async def test_move_files_requires_file_ids(patched):
    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER, session=_session(), file_ids=[], to_root=True
    )

    assert isinstance(resp, ErrorResponse)
    assert "file_ids" in resp.message


@pytest.mark.asyncio
async def test_move_files_requires_a_destination(patched):
    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER, session=_session(), file_ids=["file-1"]
    )

    assert isinstance(resp, ErrorResponse)
    assert "Provide a destination" in resp.message
    patched.bulk_move_files_to_folder.assert_not_called()


@pytest.mark.asyncio
async def test_move_files_rejects_folder_and_root_together(patched):
    resp = await MoveWorkspaceFilesToFolderTool()._execute(
        user_id=_USER,
        session=_session(),
        file_ids=["file-1"],
        folder_id="fld-1",
        to_root=True,
    )

    assert isinstance(resp, ErrorResponse)
    assert "not both" in resp.message


# ---------------------------------------------------------------------------
# Shared guards
# ---------------------------------------------------------------------------

_CALLS = [
    (CreateWorkspaceFolderTool, {"name": "Invoices"}),
    (DeleteWorkspaceFolderTool, {"folder_id": "fld-1"}),
    (MoveWorkspaceFilesToFolderTool, {"file_ids": ["file-1"], "to_root": True}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_cls, kwargs", _CALLS)
async def test_an_expert_session_cannot_change_the_owners_folders(
    as_expert, tool_cls, kwargs
):
    """Folders are the owner's, even for an expert that may read their files."""
    resp = await tool_cls()._execute(user_id=_USER, session=_session(), **kwargs)

    assert isinstance(resp, ErrorResponse)
    assert resp.error == "access_denied"
    as_expert.create_workspace_folder.assert_not_called()
    as_expert.delete_workspace_folder.assert_not_called()
    as_expert.bulk_move_files_to_folder.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_cls, kwargs", _CALLS)
async def test_all_folder_tools_require_authentication(patched, tool_cls, kwargs):
    resp = await tool_cls()._execute(user_id=None, session=_session(), **kwargs)

    assert isinstance(resp, ErrorResponse)
    assert "Authentication required" in resp.message
