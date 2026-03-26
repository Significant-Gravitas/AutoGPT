"""
Tests for workspace data layer UniqueViolationError handling.

Verifies that create_workspace_file gracefully handles the race condition
where two concurrent requests try to create a file at the same
(workspaceId, path), returning the existing record instead of crashing.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.errors import UniqueViolationError

from backend.data.workspace import create_workspace_file


def _unique_violation() -> UniqueViolationError:
    """Create a UniqueViolationError for testing."""
    data = {
        "user_facing_error": {
            "message": "Unique constraint failed on (workspaceId, path)",
        }
    }
    return UniqueViolationError(data)


_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _fake_db_file():
    """Create a fake Prisma UserWorkspaceFile record."""
    f = MagicMock()
    f.id = "existing-id"
    f.workspaceId = "ws-123"
    f.createdAt = _NOW
    f.updatedAt = _NOW
    f.name = "report.pdf"
    f.path = "/docs/report.pdf"
    f.storagePath = "gs://bucket/ws-123/existing-id/report.pdf"
    f.mimeType = "application/pdf"
    f.sizeBytes = 1024
    f.checksum = "abc123"
    f.isDeleted = False
    f.deletedAt = None
    f.metadata = {}
    return f


@pytest.mark.asyncio
async def test_create_workspace_file_unique_violation_returns_existing():
    """When a UniqueViolationError occurs, the existing file is returned."""
    existing = _fake_db_file()

    mock_prisma = MagicMock()
    mock_prisma.create = AsyncMock(side_effect=_unique_violation())
    mock_prisma.find_first = AsyncMock(return_value=existing)

    with patch(
        "backend.data.workspace.UserWorkspaceFile.prisma",
        return_value=mock_prisma,
    ):
        result = await create_workspace_file(
            workspace_id="ws-123",
            file_id="new-id",
            name="report.pdf",
            path="/docs/report.pdf",
            storage_path="gs://bucket/ws-123/new-id/report.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
        )

    assert result.id == "existing-id"
    assert result.path == "/docs/report.pdf"
    mock_prisma.find_first.assert_called_once()


@pytest.mark.asyncio
async def test_create_workspace_file_unique_violation_reraises_if_not_found():
    """If the existing record vanishes after UniqueViolationError, re-raise."""
    mock_prisma = MagicMock()
    mock_prisma.create = AsyncMock(side_effect=_unique_violation())
    mock_prisma.find_first = AsyncMock(return_value=None)

    with patch(
        "backend.data.workspace.UserWorkspaceFile.prisma",
        return_value=mock_prisma,
    ):
        with pytest.raises(UniqueViolationError):
            await create_workspace_file(
                workspace_id="ws-123",
                file_id="new-id",
                name="report.pdf",
                path="/docs/report.pdf",
                storage_path="gs://bucket/ws-123/new-id/report.pdf",
                mime_type="application/pdf",
                size_bytes=1024,
            )


@pytest.mark.asyncio
async def test_create_workspace_file_success():
    """Normal creation works without hitting UniqueViolationError."""
    created = _fake_db_file()
    created.id = "new-id"

    mock_prisma = MagicMock()
    mock_prisma.create = AsyncMock(return_value=created)

    with patch(
        "backend.data.workspace.UserWorkspaceFile.prisma",
        return_value=mock_prisma,
    ):
        result = await create_workspace_file(
            workspace_id="ws-123",
            file_id="new-id",
            name="report.pdf",
            path="/docs/report.pdf",
            storage_path="gs://bucket/ws-123/new-id/report.pdf",
            mime_type="application/pdf",
            size_bytes=1024,
        )

    assert result.id == "new-id"
    mock_prisma.find_first.assert_not_called()
