"""
Tests for server-side workspace file move and copy.
"""

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.errors import UniqueViolationError

from backend.data.workspace import WorkspaceFile
from backend.data.workspace_scope import WorkspaceAccessDeniedError, WorkspaceScope
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
        # A replaced file's embedding is dropped over the DatabaseManager when
        # Prisma is not connected, which a unit test has no server for.
        stack.enter_context(
            patch(
                "backend.api.features.workspace.embeddings"
                ".delete_workspace_file_embedding",
                AsyncMock(),
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
async def test_move_file_overwrite_retires_the_occupant_with_the_move(
    manager, mock_db, mock_storage
):
    """The occupant's row goes in the move's transaction; its blob only after."""
    occupant = _make_workspace_file(
        id="other-file",
        path="/reports/summary.pdf",
        storage_path="local://ws-123/other-file/summary.pdf",
    )
    mock_db.get_workspace_file_by_path.return_value = occupant
    mock_db.update_workspace_file_location.return_value = _make_workspace_file(
        name="summary.pdf", path="/reports/summary.pdf"
    )

    with _patched(mock_db, mock_storage):
        moved = await manager.move_file(
            "file-1", "/reports/summary.pdf", overwrite=True
        )

    assert moved.path == "/reports/summary.pdf"
    kwargs = mock_db.update_workspace_file_location.call_args.kwargs
    assert kwargs["replace_file_id"] == "other-file"
    mock_storage.delete.assert_awaited_once_with(
        "local://ws-123/other-file/summary.pdf"
    )


@pytest.mark.asyncio
async def test_a_failed_overwrite_move_keeps_the_occupant(
    manager, mock_db, mock_storage
):
    occupant = _make_workspace_file(id="other-file", path="/reports/summary.pdf")
    mock_db.get_workspace_file_by_path.return_value = occupant
    mock_db.update_workspace_file_location.side_effect = RuntimeError("db down")

    with _patched(mock_db, mock_storage):
        with pytest.raises(RuntimeError):
            await manager.move_file("file-1", "/reports/summary.pdf", overwrite=True)

    mock_storage.delete.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_overwrite_copy_keeps_the_occupant(
    manager, mock_db, mock_storage
):
    """The new blob is discarded; the occupant's blob is never touched."""
    occupant = _make_workspace_file(
        id="other-file",
        path="/reports/summary.pdf",
        storage_path="local://ws-123/other-file/summary.pdf",
    )
    mock_db.get_workspace_file_by_path.return_value = occupant
    mock_db.create_workspace_file.side_effect = RuntimeError("db down")

    with _patched(mock_db, mock_storage):
        with pytest.raises(RuntimeError):
            await manager.copy_file("file-1", "/reports/summary.pdf", overwrite=True)

    assert mock_db.create_workspace_file.call_args.kwargs["replace_file_id"] == (
        "other-file"
    )
    mock_storage.delete.assert_awaited_once_with("local://ws-123/new-uuid/summary.pdf")


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
async def test_copy_file_overwrite_frees_the_occupant_from_the_quota(
    manager, mock_db, mock_storage
):
    """Overwrite subtracts the destination's current size before the limit check."""
    source = _make_workspace_file(size_bytes=1000)
    occupant = _make_workspace_file(
        id="occupant", path="/reports/summary.pdf", size_bytes=1000
    )
    mock_db.get_workspace_file.return_value = source
    mock_db.get_workspace_file_by_path.return_value = occupant
    mock_db.get_workspace_total_size.return_value = 2500
    mock_db.create_workspace_file.return_value = _make_workspace_file(
        id="new-copy", path="/reports/summary.pdf"
    )

    # 2500 + 1000 = 3500 > 3000 would fail; freeing the 1000-byte occupant
    # drops usage to 1500, so 1500 + 1000 = 2500 fits.
    with _patched(mock_db, mock_storage, storage_limit=3000):
        copied = await manager.copy_file(
            "file-1", "/reports/summary.pdf", overwrite=True
        )

    assert copied.id == "new-copy"
    mock_storage.copy.assert_awaited_once()
    kwargs = mock_db.create_workspace_file.call_args.kwargs
    assert kwargs["replace_file_id"] == "occupant"


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


# ---------------------------------------------------------------------------
# Expert scope and skill packages
# ---------------------------------------------------------------------------


@pytest.fixture
def expert_manager():
    """An expert session that reads the owner's files and writes only its own."""
    scope = WorkspaceScope(
        expert_id="exp-1", session_ids=["sess-1"], reads_user_files=True
    )
    return WorkspaceManager(
        user_id="user-123", workspace_id="ws-123", session_id="sess-1", scope=scope
    )


@pytest.mark.asyncio
async def test_an_expert_cannot_move_the_owners_file(expert_manager, mock_db):
    """Reading a user file is granted; taking it away from its place is not."""
    with _patched(mock_db):
        with pytest.raises(WorkspaceAccessDeniedError):
            await expert_manager.move_file("file-1", "/moved.pdf")

    mock_db.update_workspace_file_location.assert_not_called()


@pytest.mark.asyncio
async def test_an_expert_cannot_move_into_another_session(expert_manager, mock_db):
    mock_db.get_workspace_file.return_value = _make_workspace_file(
        path="/sessions/sess-1/report.pdf"
    )

    with _patched(mock_db):
        with pytest.raises(WorkspaceAccessDeniedError):
            await expert_manager.move_file("file-1", "/sessions/other/report.pdf")

    mock_db.update_workspace_file_location.assert_not_called()


@pytest.mark.asyncio
async def test_an_expert_copies_the_owners_file_into_its_chat_at_the_root(
    expert_manager, mock_db, mock_storage
):
    """The copy is the expert's file, so it does not join the owner's folder."""
    mock_db.get_workspace_file.return_value = _make_workspace_file(folder_id="fld-1")
    mock_db.create_workspace_file.return_value = _make_workspace_file(
        id="new-uuid", path="/sessions/sess-1/report.pdf"
    )

    with _patched(mock_db, mock_storage):
        await expert_manager.copy_file("file-1", "/report.pdf")

    kwargs = mock_db.create_workspace_file.call_args.kwargs
    assert kwargs["path"] == "/sessions/sess-1/report.pdf"
    assert kwargs["folder_id"] is None


@pytest.mark.asyncio
async def test_an_expert_cannot_file_a_copy_into_a_folder(
    expert_manager, mock_db, mock_storage
):
    with _patched(mock_db, mock_storage):
        with pytest.raises(WorkspaceAccessDeniedError):
            await expert_manager.copy_file("file-1", "/report.pdf", folder_id="fld-1")

    mock_storage.copy.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_path, new_path",
    [
        ("/skills/research/SKILL.md", "/research.md"),
        ("/notes.md", "/skills/research/notes.md"),
        ("/notes.md", "/experts/exp-1/skills/research/notes.md"),
    ],
)
async def test_neither_end_of_a_transfer_may_be_a_skill_package(
    manager, mock_db, mock_storage, source_path, new_path
):
    mock_db.get_workspace_file.return_value = _make_workspace_file(path=source_path)

    with _patched(mock_db, mock_storage):
        with pytest.raises(ValueError, match="Skill packages"):
            await manager.move_file("file-1", new_path)
        with pytest.raises(ValueError, match="Skill packages"):
            await manager.copy_file("file-1", new_path)

    mock_db.update_workspace_file_location.assert_not_called()
    mock_storage.copy.assert_not_called()
