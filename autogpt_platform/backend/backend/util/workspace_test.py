"""
Tests for WorkspaceManager.write_file UniqueViolationError handling.
"""

from unittest.mock import AsyncMock, patch

import pytest
from prisma.errors import UniqueViolationError

from backend.data.workspace import WorkspaceFile
from backend.util.workspace import WorkspaceManager


def _make_workspace_file(**overrides) -> WorkspaceFile:
    """Create a mock WorkspaceFile with sensible defaults."""
    defaults = dict(
        id="existing-file-id",
        workspaceId="ws-123",
        fileId="existing-uuid",
        name="test.txt",
        path="/test.txt",
        storagePath="ws-123/existing-uuid/test.txt",
        mimeType="text/plain",
        sizeBytes=5,
        checksum="abc123",
        metadata=None,
        createdAt=None,
        updatedAt=None,
    )
    defaults.update(overrides)
    return WorkspaceFile(**defaults)


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


@pytest.mark.asyncio
async def test_write_file_overwrite_false_unique_violation_raises_and_cleans_up(
    manager, mock_storage
):
    """overwrite=False + UniqueViolationError → ValueError + storage cleanup."""
    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            return_value=mock_storage,
        ),
        patch(
            "backend.util.workspace.get_workspace_file_by_path",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "backend.util.workspace.create_workspace_file",
            new_callable=AsyncMock,
            side_effect=_unique_violation(),
        ),
        patch("backend.util.workspace.scan_content_safe", new_callable=AsyncMock),
    ):
        with pytest.raises(ValueError, match="File already exists"):
            await manager.write_file(
                filename="test.txt", content=b"hello", overwrite=False
            )

    # Storage file should be cleaned up
    mock_storage.delete.assert_called_once()


@pytest.mark.asyncio
async def test_write_file_overwrite_true_conflict_then_retry_succeeds(
    manager, mock_storage
):
    """overwrite=True + conflict → delete existing → retry succeeds."""
    created_file = _make_workspace_file()
    existing_file = _make_workspace_file(id="old-id", fileId="old-uuid")

    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            return_value=mock_storage,
        ),
        patch(
            "backend.util.workspace.get_workspace_file_by_path",
            new_callable=AsyncMock,
            return_value=existing_file,
        ),
        patch(
            "backend.util.workspace.create_workspace_file",
            new_callable=AsyncMock,
            side_effect=[_unique_violation(), created_file],
        ),
        patch("backend.util.workspace.scan_content_safe", new_callable=AsyncMock),
        patch.object(manager, "delete_file", new_callable=AsyncMock) as mock_delete,
    ):
        result = await manager.write_file(
            filename="test.txt", content=b"hello", overwrite=True
        )

    assert result == created_file
    mock_delete.assert_called_once_with("old-id")
    # Storage should NOT be cleaned up — write succeeded
    mock_storage.delete.assert_not_called()


@pytest.mark.asyncio
async def test_write_file_overwrite_true_double_conflict_raises_and_cleans_up(
    manager, mock_storage
):
    """overwrite=True + double UniqueViolationError → ValueError + cleanup."""
    existing_file = _make_workspace_file(id="old-id")

    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            return_value=mock_storage,
        ),
        patch(
            "backend.util.workspace.get_workspace_file_by_path",
            new_callable=AsyncMock,
            return_value=existing_file,
        ),
        patch(
            "backend.util.workspace.create_workspace_file",
            new_callable=AsyncMock,
            side_effect=[_unique_violation(), _unique_violation()],
        ),
        patch("backend.util.workspace.scan_content_safe", new_callable=AsyncMock),
        patch.object(manager, "delete_file", new_callable=AsyncMock),
    ):
        with pytest.raises(ValueError, match="File already exists"):
            await manager.write_file(
                filename="test.txt", content=b"hello", overwrite=True
            )

    # Storage file should be cleaned up
    mock_storage.delete.assert_called_once()
