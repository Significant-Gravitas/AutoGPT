"""
Tests for WorkspaceManager.write_file UniqueViolationError handling.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.errors import UniqueViolationError

from backend.data.workspace import WorkspaceFile
from backend.util.workspace import WorkspaceManager

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_workspace_file(
    id: str = "existing-file-id",
    workspace_id: str = "ws-123",
    name: str = "test.txt",
    path: str = "/test.txt",
    storage_path: str = "ws-123/existing-uuid/test.txt",
    mime_type: str = "text/plain",
    size_bytes: int = 5,
    checksum: str = "abc123",
) -> WorkspaceFile:
    """Create a mock WorkspaceFile with sensible defaults."""
    return WorkspaceFile(
        id=id,
        workspace_id=workspace_id,
        name=name,
        path=path,
        storage_path=storage_path,
        mime_type=mime_type,
        size_bytes=size_bytes,
        checksum=checksum,
        metadata={},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _unique_violation() -> UniqueViolationError:
    """Create a UniqueViolationError for testing."""
    data = {
        "user_facing_error": {
            "message": "Unique constraint failed on the fields: (`path`)",
        }
    }
    return UniqueViolationError(data)


@pytest.fixture
def manager():
    return WorkspaceManager(user_id="user-123", workspace_id="ws-123")


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.store.return_value = "ws-123/some-uuid/test.txt"
    storage.delete = AsyncMock()
    return storage


@pytest.fixture
def mock_db():
    """Create a mock workspace_db() return value."""
    db = MagicMock()
    db.create_workspace_file = AsyncMock()
    db.get_workspace_file_by_path = AsyncMock()
    db.get_workspace_file = AsyncMock()
    db.get_workspace_total_size = AsyncMock(return_value=0)
    return db


@pytest.mark.asyncio
async def test_write_file_no_overwrite_unique_violation_raises_and_cleans_up(
    manager, mock_storage, mock_db
):
    """overwrite=False + UniqueViolationError → ValueError + storage cleanup."""
    mock_db.get_workspace_file_by_path.return_value = None
    mock_db.create_workspace_file.side_effect = _unique_violation()

    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            return_value=mock_storage,
        ),
        patch("backend.util.workspace.workspace_db", return_value=mock_db),
        patch("backend.util.workspace.scan_content_safe", new_callable=AsyncMock),
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            return_value=250 * 1024 * 1024,
        ),
    ):
        with pytest.raises(ValueError, match="File already exists"):
            await manager.write_file(
                filename="test.txt", content=b"hello", overwrite=False
            )

    mock_storage.delete.assert_called_once()


@pytest.mark.asyncio
async def test_write_file_overwrite_conflict_then_retry_succeeds(
    manager, mock_storage, mock_db
):
    """overwrite=True + conflict → delete existing → retry succeeds."""
    created_file = _make_workspace_file()
    existing_file = _make_workspace_file(id="old-id")

    mock_db.get_workspace_file_by_path.return_value = existing_file
    mock_db.create_workspace_file.side_effect = [_unique_violation(), created_file]

    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            return_value=mock_storage,
        ),
        patch("backend.util.workspace.workspace_db", return_value=mock_db),
        patch("backend.util.workspace.scan_content_safe", new_callable=AsyncMock),
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            return_value=250 * 1024 * 1024,
        ),
        patch.object(manager, "delete_file", new_callable=AsyncMock) as mock_delete,
    ):
        result = await manager.write_file(
            filename="test.txt", content=b"hello", overwrite=True
        )

    assert result == created_file
    mock_delete.assert_called_once_with("old-id")
    mock_storage.delete.assert_not_called()


@pytest.mark.asyncio
async def test_write_file_overwrite_exhausted_retries_raises_and_cleans_up(
    manager, mock_storage, mock_db
):
    """overwrite=True + all retries exhausted → ValueError + cleanup."""
    existing_file = _make_workspace_file(id="old-id")

    mock_db.get_workspace_file_by_path.return_value = existing_file
    # Initial + 2 retries = 3 UniqueViolationErrors
    mock_db.create_workspace_file.side_effect = [
        _unique_violation(),
        _unique_violation(),
        _unique_violation(),
    ]

    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            return_value=mock_storage,
        ),
        patch("backend.util.workspace.workspace_db", return_value=mock_db),
        patch("backend.util.workspace.scan_content_safe", new_callable=AsyncMock),
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            return_value=250 * 1024 * 1024,
        ),
        patch.object(manager, "delete_file", new_callable=AsyncMock),
    ):
        with pytest.raises(ValueError, match="Unable to overwrite.*concurrent write"):
            await manager.write_file(
                filename="test.txt", content=b"hello", overwrite=True
            )

    mock_storage.delete.assert_called_once()


@pytest.mark.asyncio
async def test_write_file_quota_exceeded_raises_value_error(
    manager, mock_storage, mock_db
):
    """write_file raises ValueError when workspace storage quota is exceeded."""
    mock_db.get_workspace_file_by_path.return_value = None

    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            return_value=mock_storage,
        ),
        patch("backend.util.workspace.workspace_db", return_value=mock_db),
        patch(
            "backend.util.workspace.scan_content_safe", new_callable=AsyncMock
        ) as mock_scan,
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            return_value=250 * 1024 * 1024,  # 250 MB limit
        ),
    ):
        mock_db.get_workspace_total_size.return_value = 250 * 1024 * 1024  # at limit
        with pytest.raises(ValueError, match="Storage limit exceeded"):
            await manager.write_file(filename="test.txt", content=b"hello")

    # Quota rejection should short-circuit before expensive virus scan
    mock_scan.assert_not_called()
    # Storage should NOT have been written to
    mock_storage.store.assert_not_called()


@pytest.mark.asyncio
async def test_write_file_rejects_upload_when_usage_already_exceeds_downgraded_limit(
    manager, mock_storage, mock_db
):
    """Downgrading below current usage should block further uploads until usage drops."""
    mock_db.get_workspace_file_by_path.return_value = None

    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            return_value=mock_storage,
        ),
        patch("backend.util.workspace.workspace_db", return_value=mock_db),
        patch(
            "backend.util.workspace.scan_content_safe", new_callable=AsyncMock
        ) as mock_scan,
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            return_value=250 * 1024 * 1024,
        ),
    ):
        mock_db.get_workspace_total_size.return_value = 300 * 1024 * 1024
        with pytest.raises(ValueError, match="Storage limit exceeded"):
            await manager.write_file(filename="test.txt", content=b"hello")

    mock_scan.assert_not_called()
    mock_storage.store.assert_not_called()


@pytest.mark.asyncio
async def test_write_file_overwrite_not_double_counted(manager, mock_storage, mock_db):
    """Overwriting a file subtracts the old file size from usage check."""
    existing_file = _make_workspace_file(size_bytes=50)
    created_file = _make_workspace_file()
    mock_db.get_workspace_file_by_path.return_value = existing_file
    mock_db.create_workspace_file.return_value = created_file

    limit_bytes = 100
    current_usage = 90  # 90 bytes used, 50 of which is the file being replaced
    content = b"x" * 50  # replacing with same-size file — should succeed

    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            return_value=mock_storage,
        ),
        patch("backend.util.workspace.workspace_db", return_value=mock_db),
        patch("backend.util.workspace.scan_content_safe", new_callable=AsyncMock),
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            return_value=limit_bytes,
        ),
    ):
        mock_db.get_workspace_total_size.return_value = current_usage
        # Should NOT raise — net usage after overwrite is 90 - 50 + 50 = 90, under 100
        result = await manager.write_file(
            filename="test.txt", content=content, overwrite=True
        )
    assert result == created_file


@pytest.mark.asyncio
async def test_write_file_storage_check_routes_through_workspace_db_accessor(
    manager, mock_storage, mock_db
):
    """Storage-limit pre-check must go through ``workspace_db()`` so the
    executor (which has no Prisma connection) hits the database-manager RPC
    instead of raising ``ClientNotConnectedError``."""
    created_file = _make_workspace_file()
    mock_db.get_workspace_file_by_path.return_value = None
    mock_db.create_workspace_file.return_value = created_file
    mock_db.get_workspace_total_size.return_value = 100

    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            return_value=mock_storage,
        ),
        patch("backend.util.workspace.workspace_db", return_value=mock_db),
        patch("backend.util.workspace.scan_content_safe", new_callable=AsyncMock),
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            return_value=1_000_000,
        ),
    ):
        await manager.write_file(filename="test.txt", content=b"hello")

    mock_db.get_workspace_total_size.assert_awaited_once_with("ws-123")
