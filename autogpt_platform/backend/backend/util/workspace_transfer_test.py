"""
Tests for server-side workspace file move and copy.
"""

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.errors import UniqueViolationError

from backend.data.workspace import WorkspaceFile
from backend.util.exceptions import NotFoundError
from backend.util.workspace import WorkspaceManager

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_workspace_file(
    id: str = "file-1",
    workspace_id: str = "ws-123",
    name: str = "report.pdf",
    path: str = "/report.pdf",
    storage_path: str = "local://ws-123/file-1/report.pdf",
    mime_type: str = "application/pdf",
    size_bytes: int = 2048,
    checksum: str = "abc123",
    folder_id: str | None = None,
) -> WorkspaceFile:
    return WorkspaceFile(
        id=id,
        workspace_id=workspace_id,
        name=name,
        path=path,
        storage_path=storage_path,
        mime_type=mime_type,
        size_bytes=size_bytes,
        checksum=checksum,
        folder_id=folder_id,
        metadata={},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _unique_violation() -> UniqueViolationError:
    return UniqueViolationError(
        {
            "user_facing_error": {
                "message": "Unique constraint failed on the fields: (`path`)",
            }
        }
    )


@pytest.fixture
def manager():
    return WorkspaceManager(user_id="user-123", workspace_id="ws-123")


@pytest.fixture
def session_manager():
    return WorkspaceManager(
        user_id="user-123", workspace_id="ws-123", session_id="sess-1"
    )


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.copy.return_value = "local://ws-123/new-uuid/summary.pdf"
    storage.delete = AsyncMock()
    return storage


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.get_workspace_file = AsyncMock(return_value=_make_workspace_file())
    db.get_workspace_file_by_path = AsyncMock(return_value=None)
    db.create_workspace_file = AsyncMock()
    db.update_workspace_file_location = AsyncMock()
    db.get_workspace_total_size = AsyncMock(return_value=0)
    return db


@contextmanager
def _patched(mock_db, mock_storage=None, storage_limit: int = 0, folder_valid=True):
    """Patch the transfer module's collaborators (mocked where used)."""
    folder_db = MagicMock()
    if folder_valid:
        folder_db.get_workspace_folder = AsyncMock(return_value=MagicMock())
    else:
        folder_db.get_workspace_folder = AsyncMock(
            side_effect=NotFoundError("Folder not found")
        )
    with ExitStack() as stack:
        stack.enter_context(
            patch("backend.util.workspace_transfer.workspace_db", return_value=mock_db)
        )
        stack.enter_context(
            patch(
                "backend.util.workspace_transfer.workspace_folder_db",
                return_value=folder_db,
            )
        )
        stack.enter_context(
            patch(
                "backend.util.workspace_transfer.get_workspace_storage_limit_bytes",
                AsyncMock(return_value=storage_limit),
            )
        )
        # The search index is fire-and-forget; keep it out of these assertions.
        stack.enter_context(
            patch(
                "backend.api.features.workspace.embeddings"
                ".schedule_workspace_file_embedding",
                MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "backend.util.workspace_transfer.get_workspace_storage",
                AsyncMock(return_value=mock_storage or AsyncMock()),
            )
        )
        yield


# ---------------------------------------------------------------------------
# move_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_file_rewrites_path_and_name_without_touching_storage(
    manager, mock_db, mock_storage
):
    """A move is metadata-only: no bytes are read, copied, or deleted."""
    mock_db.update_workspace_file_location.return_value = _make_workspace_file(
        name="summary.pdf", path="/reports/summary.pdf"
    )

    with _patched(mock_db, mock_storage):
        moved = await manager.move_file("file-1", "/reports/summary.pdf")

    assert moved.path == "/reports/summary.pdf"
    assert moved.name == "summary.pdf"
    kwargs = mock_db.update_workspace_file_location.call_args.kwargs
    assert kwargs["path"] == "/reports/summary.pdf"
    assert kwargs["name"] == "summary.pdf"
    assert kwargs["workspace_id"] == "ws-123"
    # The whole point of the ticket: no content ever moves.
    mock_storage.copy.assert_not_called()
    mock_storage.delete.assert_not_called()


@pytest.mark.asyncio
async def test_move_file_rejects_occupied_destination(manager, mock_db):
    """Destination taken + overwrite=False → actionable ValueError."""
    mock_db.get_workspace_file_by_path.return_value = _make_workspace_file(
        id="other-file", path="/reports/summary.pdf"
    )

    with _patched(mock_db):
        with pytest.raises(ValueError, match="already exists at path"):
            await manager.move_file("file-1", "/reports/summary.pdf")

    mock_db.update_workspace_file_location.assert_not_called()


@pytest.mark.asyncio
async def test_move_file_overwrite_deletes_occupant_first(manager, mock_db):
    """overwrite=True removes the destination file before relocating."""
    occupant = _make_workspace_file(id="other-file", path="/reports/summary.pdf")
    mock_db.get_workspace_file_by_path.return_value = occupant
    mock_db.update_workspace_file_location.return_value = _make_workspace_file(
        name="summary.pdf", path="/reports/summary.pdf"
    )

    with _patched(mock_db):
        with patch.object(
            WorkspaceManager, "delete_file", AsyncMock(return_value=True)
        ) as delete_file:
            moved = await manager.move_file(
                "file-1", "/reports/summary.pdf", overwrite=True
            )

    delete_file.assert_awaited_once_with("other-file")
    assert moved.path == "/reports/summary.pdf"


@pytest.mark.asyncio
async def test_move_file_to_same_path_is_a_noop(manager, mock_db):
    """Moving a file onto its current path short-circuits without a write."""
    with _patched(mock_db):
        result = await manager.move_file("file-1", "/report.pdf")

    assert result.path == "/report.pdf"
    mock_db.update_workspace_file_location.assert_not_called()


@pytest.mark.asyncio
async def test_move_file_preserves_folder_unless_overridden(manager, mock_db):
    """folder_id defaults to the source's folder, and is honoured when given."""
    mock_db.get_workspace_file.return_value = _make_workspace_file(folder_id="fld-1")
    mock_db.update_workspace_file_location.return_value = _make_workspace_file(
        path="/a.pdf", folder_id="fld-1"
    )

    with _patched(mock_db):
        await manager.move_file("file-1", "/a.pdf")
        assert mock_db.update_workspace_file_location.call_args.kwargs["folder_id"] == (
            "fld-1"
        )

        await manager.move_file("file-1", "/b.pdf", folder_id="fld-2")
        assert mock_db.update_workspace_file_location.call_args.kwargs["folder_id"] == (
            "fld-2"
        )


@pytest.mark.asyncio
async def test_move_file_rejects_unknown_folder(manager, mock_db):
    """A folder_id outside this workspace is rejected before any write."""
    with _patched(mock_db, folder_valid=False):
        with pytest.raises(ValueError, match="Workspace folder not found"):
            await manager.move_file("file-1", "/b.pdf", folder_id="foreign-folder")

    mock_db.update_workspace_file_location.assert_not_called()


@pytest.mark.asyncio
async def test_copy_file_rejects_unknown_folder(manager, mock_db, mock_storage):
    """A folder_id outside this workspace is rejected before any bytes are copied."""
    with _patched(mock_db, mock_storage, folder_valid=False):
        with pytest.raises(ValueError, match="Workspace folder not found"):
            await manager.copy_file("file-1", "/copy.pdf", folder_id="foreign-folder")

    mock_storage.copy.assert_not_called()
    mock_db.create_workspace_file.assert_not_called()


@pytest.mark.asyncio
async def test_move_file_resolves_paths_within_the_session(session_manager, mock_db):
    """Bare destinations are scoped to the session folder, like write_file."""
    mock_db.get_workspace_file.return_value = _make_workspace_file(
        path="/sessions/sess-1/report.pdf"
    )
    mock_db.update_workspace_file_location.return_value = _make_workspace_file(
        path="/sessions/sess-1/reports/summary.pdf"
    )

    with _patched(mock_db):
        await session_manager.move_file("file-1", "/reports/summary.pdf")

    assert mock_db.update_workspace_file_location.call_args.kwargs["path"] == (
        "/sessions/sess-1/reports/summary.pdf"
    )


@pytest.mark.asyncio
async def test_move_file_missing_source_raises(manager, mock_db):
    mock_db.get_workspace_file.return_value = None

    with _patched(mock_db):
        with pytest.raises(FileNotFoundError):
            await manager.move_file("nope", "/x.pdf")


@pytest.mark.asyncio
async def test_move_file_concurrent_conflict_becomes_value_error(manager, mock_db):
    """A racing writer taking the destination surfaces as a ValueError."""
    mock_db.update_workspace_file_location.side_effect = _unique_violation()

    with _patched(mock_db):
        with pytest.raises(ValueError, match="concurrent write conflict"):
            await manager.move_file("file-1", "/reports/summary.pdf")


# ---------------------------------------------------------------------------
# copy_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_copy_file_uses_server_side_storage_copy(manager, mock_db, mock_storage):
    """Copy delegates to the storage backend rather than reading bytes."""
    mock_db.create_workspace_file.return_value = _make_workspace_file(
        id="file-2", name="summary.pdf", path="/reports/summary.pdf"
    )

    with _patched(mock_db, mock_storage):
        copied = await manager.copy_file("file-1", "/reports/summary.pdf")

    assert copied.id == "file-2"
    mock_storage.copy.assert_awaited_once()
    args = mock_storage.copy.await_args.args
    assert args[0] == "local://ws-123/file-1/report.pdf"  # source blob
    assert args[1] == "ws-123"
    assert args[3] == "summary.pdf"
    # Content is never pulled into this process.
    mock_storage.retrieve.assert_not_called()

    create_kwargs = mock_db.create_workspace_file.call_args.kwargs
    assert create_kwargs["path"] == "/reports/summary.pdf"
    assert create_kwargs["size_bytes"] == 2048
    assert create_kwargs["mime_type"] == "application/pdf"
    assert create_kwargs["checksum"] == "abc123"
    assert create_kwargs["file_id"] != "file-1"


@pytest.mark.asyncio
async def test_copy_file_enforces_storage_quota(manager, mock_db, mock_storage):
    """A copy that would exceed the tier quota is rejected before any write."""
    mock_db.get_workspace_total_size.return_value = 900
    with _patched(mock_db, mock_storage, storage_limit=1000):
        with pytest.raises(ValueError, match="Storage limit exceeded"):
            await manager.copy_file("file-1", "/reports/summary.pdf")

    mock_storage.copy.assert_not_called()
    mock_db.create_workspace_file.assert_not_called()


@pytest.mark.asyncio
async def test_copy_file_rejects_occupied_destination(manager, mock_db, mock_storage):
    mock_db.get_workspace_file_by_path.return_value = _make_workspace_file(
        id="other-file", path="/reports/summary.pdf"
    )

    with _patched(mock_db, mock_storage):
        with pytest.raises(ValueError, match="already exists at path"):
            await manager.copy_file("file-1", "/reports/summary.pdf")

    mock_storage.copy.assert_not_called()


@pytest.mark.asyncio
async def test_copy_file_rejects_copy_onto_itself(manager, mock_db, mock_storage):
    with _patched(mock_db, mock_storage):
        with pytest.raises(ValueError, match="onto itself"):
            await manager.copy_file("file-1", "/report.pdf")

    mock_storage.copy.assert_not_called()


@pytest.mark.asyncio
async def test_copy_file_cleans_up_blob_when_db_record_fails(
    manager, mock_db, mock_storage
):
    """A failed DB insert must not leave an orphaned blob behind."""
    mock_db.create_workspace_file.side_effect = _unique_violation()

    with _patched(mock_db, mock_storage):
        with pytest.raises(ValueError, match="concurrent write conflict"):
            await manager.copy_file("file-1", "/reports/summary.pdf")

    mock_storage.delete.assert_awaited_once_with("local://ws-123/new-uuid/summary.pdf")


@pytest.mark.asyncio
async def test_copy_file_inherits_source_folder(manager, mock_db, mock_storage):
    mock_db.get_workspace_file.return_value = _make_workspace_file(folder_id="fld-1")
    mock_db.create_workspace_file.return_value = _make_workspace_file(id="file-2")

    with _patched(mock_db, mock_storage):
        await manager.copy_file("file-1", "/reports/summary.pdf")

    assert mock_db.create_workspace_file.call_args.kwargs["folder_id"] == "fld-1"
