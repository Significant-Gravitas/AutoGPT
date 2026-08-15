"""
Direct unit tests for workspace data-layer functions that other suites mock.
"""

from datetime import datetime, timezone

import pytest

from backend.data import workspace as ws


def _file_record(**overrides):
    """A lightweight stand-in for a Prisma UserWorkspaceFile record."""

    class _Rec:
        pass

    rec = _Rec()
    rec.id = overrides.get("id", "file-1")
    rec.workspaceId = overrides.get("workspaceId", "ws-1")
    rec.createdAt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rec.updatedAt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rec.name = overrides.get("name", "summary.pdf")
    rec.path = overrides.get("path", "/reports/summary.pdf")
    rec.storagePath = overrides.get("storagePath", "local://ws-1/file-1/report.pdf")
    rec.mimeType = overrides.get("mimeType", "application/pdf")
    rec.sizeBytes = overrides.get("sizeBytes", 2048)
    rec.checksum = overrides.get("checksum", "abc123")
    rec.isDeleted = overrides.get("isDeleted", False)
    rec.deletedAt = overrides.get("deletedAt", None)
    rec.folderId = overrides.get("folderId", None)
    rec.metadata = overrides.get("metadata", {})
    return rec


@pytest.mark.asyncio
async def test_update_location_is_scoped_to_the_workspace(mocker):
    """The write must match on workspaceId + isDeleted, not the file ID alone."""
    prisma = mocker.MagicMock()
    prisma.update_many = mocker.AsyncMock(return_value=1)
    prisma.find_first = mocker.AsyncMock(
        return_value=_file_record(
            name="summary.pdf", path="/reports/summary.pdf", folderId="fld-9"
        )
    )
    mocker.patch.object(
        ws.UserWorkspaceFile, "prisma", mocker.MagicMock(return_value=prisma)
    )

    result = await ws.update_workspace_file_location(
        file_id="file-1",
        workspace_id="ws-1",
        name="summary.pdf",
        path="reports/summary.pdf",
        folder_id="fld-9",
    )

    where = prisma.update_many.call_args.kwargs["where"]
    assert where["id"] == "file-1"
    assert where["workspaceId"] == "ws-1"
    assert where["isDeleted"] is False
    data = prisma.update_many.call_args.kwargs["data"]
    # A bare path is normalized with a leading slash before it is stored.
    assert data["path"] == "/reports/summary.pdf"
    assert data["folderId"] == "fld-9"
    assert result is not None
    assert result.path == "/reports/summary.pdf"
    assert result.folder_id == "fld-9"


@pytest.mark.asyncio
async def test_update_location_returns_none_when_nothing_matched(mocker):
    """A cross-workspace or deleted row updates zero rows and yields None."""
    prisma = mocker.MagicMock()
    prisma.update_many = mocker.AsyncMock(return_value=0)
    prisma.find_first = mocker.AsyncMock()
    mocker.patch.object(
        ws.UserWorkspaceFile, "prisma", mocker.MagicMock(return_value=prisma)
    )

    result = await ws.update_workspace_file_location(
        file_id="foreign-file",
        workspace_id="ws-1",
        name="x.pdf",
        path="/x.pdf",
    )

    assert result is None
    # No point looking up a row we never updated.
    prisma.find_first.assert_not_awaited()
