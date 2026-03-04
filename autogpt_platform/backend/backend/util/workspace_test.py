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
        patch.object(manager, "delete_file", new_callable=AsyncMock),
    ):
        with pytest.raises(ValueError, match="Unable to overwrite.*concurrent write"):
            await manager.write_file(
                filename="test.txt", content=b"hello", overwrite=True
            )

    mock_storage.delete.assert_called_once()
