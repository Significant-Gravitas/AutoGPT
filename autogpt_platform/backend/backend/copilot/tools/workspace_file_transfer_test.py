"""Tests for the move_workspace_file / copy_workspace_file CoPilot tools."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.workspace_file_transfer import (
    CopyWorkspaceFileTool,
    MoveWorkspaceFileTool,
    WorkspaceFileCopiedResponse,
    WorkspaceFileMovedResponse,
)
from backend.data.workspace import WorkspaceFile

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


def _file(
    id: str = "file-1",
    name: str = "report.pdf",
    path: str = "/report.pdf",
    mime_type: str = "application/pdf",
    size_bytes: int = 2048,
) -> WorkspaceFile:
    return WorkspaceFile(
        id=id,
        workspace_id="ws-123",
        name=name,
        path=path,
        storage_path=f"local://ws-123/{id}/{name}",
        mime_type=mime_type,
        size_bytes=size_bytes,
        checksum="abc123",
        metadata={},
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.fixture
def mock_manager():
    manager = MagicMock()
    manager.get_file_info = AsyncMock(return_value=_file())
    manager.get_file_info_by_path = AsyncMock(return_value=_file())
    manager.move_file = AsyncMock(
        return_value=_file(name="summary.pdf", path="/reports/summary.pdf")
    )
    manager.copy_file = AsyncMock(
        return_value=_file(id="file-2", name="summary.pdf", path="/reports/summary.pdf")
    )
    return manager


def _patch_manager(mock_manager):
    return patch(
        "backend.copilot.tools.workspace_file_transfer.get_workspace_manager",
        AsyncMock(return_value=mock_manager),
    )


@pytest.mark.asyncio
async def test_move_tool_returns_old_and_new_path(mock_manager):
    with _patch_manager(mock_manager):
        resp = await MoveWorkspaceFileTool()._execute(
            user_id=_USER,
            session=_session(),
            file_id="file-1",
            new_path="/reports/summary.pdf",
        )

    assert isinstance(resp, WorkspaceFileMovedResponse), resp.message
    assert resp.previous_path == "/report.pdf"
    assert resp.path == "/reports/summary.pdf"
    assert resp.name == "summary.pdf"
    mock_manager.move_file.assert_awaited_once()


@pytest.mark.asyncio
async def test_copy_tool_returns_workspace_download_url(mock_manager):
    with _patch_manager(mock_manager):
        resp = await CopyWorkspaceFileTool()._execute(
            user_id=_USER,
            session=_session(),
            file_id="file-1",
            new_path="/reports/summary.pdf",
        )

    assert isinstance(resp, WorkspaceFileCopiedResponse), resp.message
    assert resp.source_file_id == "file-1"
    assert resp.file_id == "file-2"
    assert resp.download_url == "workspace://file-2#application/pdf"
    assert resp.source_path == "/report.pdf"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [MoveWorkspaceFileTool(), CopyWorkspaceFileTool()])
async def test_requires_authentication(tool, mock_manager):
    with _patch_manager(mock_manager):
        resp = await tool._execute(
            user_id=None, session=_session(), file_id="file-1", new_path="/x.pdf"
        )
    assert isinstance(resp, ErrorResponse)
    assert "Authentication required" in resp.message


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [MoveWorkspaceFileTool(), CopyWorkspaceFileTool()])
async def test_requires_a_target(tool, mock_manager):
    with _patch_manager(mock_manager):
        resp = await tool._execute(user_id=_USER, session=_session(), new_path="/x.pdf")
    assert isinstance(resp, ErrorResponse)
    assert "file_id or path" in resp.message


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [MoveWorkspaceFileTool(), CopyWorkspaceFileTool()])
async def test_requires_new_path(tool, mock_manager):
    with _patch_manager(mock_manager):
        resp = await tool._execute(user_id=_USER, session=_session(), file_id="file-1")
    assert isinstance(resp, ErrorResponse)
    assert "new_path" in resp.message


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [MoveWorkspaceFileTool(), CopyWorkspaceFileTool()])
async def test_missing_file_surfaces_not_found(tool, mock_manager):
    mock_manager.get_file_info.return_value = None
    with _patch_manager(mock_manager):
        resp = await tool._execute(
            user_id=_USER,
            session=_session(),
            file_id="nope",
            new_path="/reports/summary.pdf",
        )
    assert isinstance(resp, ErrorResponse)
    assert "File not found" in resp.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_path,dest_path",
    [
        ("/report.pdf", "/skills/my-skill/SKILL.md"),
        ("/skills/my-skill/SKILL.md", "/report.pdf"),
        ("/report.pdf", "skills/my-skill/SKILL.md"),
        ("/report.pdf", "/Skills/my-skill/SKILL.md"),
    ],
)
async def test_skills_registry_paths_are_rejected(source_path, dest_path, mock_manager):
    """The skills registry owns its own tooling; raw moves would bypass it."""
    with _patch_manager(mock_manager):
        resp = await MoveWorkspaceFileTool()._execute(
            user_id=_USER,
            session=_session(),
            path=source_path,
            new_path=dest_path,
        )
    assert isinstance(resp, ErrorResponse)
    assert "skills registry" in resp.message
    mock_manager.move_file.assert_not_called()


@pytest.mark.asyncio
async def test_skills_registry_blocked_when_targeted_by_file_id(mock_manager):
    """A file_id only reveals its path after lookup, so re-check post-resolve."""
    mock_manager.get_file_info.return_value = _file(path="/skills/my-skill/SKILL.md")

    with _patch_manager(mock_manager):
        resp = await MoveWorkspaceFileTool()._execute(
            user_id=_USER,
            session=_session(),
            file_id="file-1",
            new_path="/elsewhere.md",
        )

    assert isinstance(resp, ErrorResponse)
    assert "skills registry" in resp.message
    mock_manager.move_file.assert_not_called()


@pytest.mark.asyncio
async def test_move_collision_error_is_surfaced_to_the_agent(mock_manager):
    mock_manager.move_file.side_effect = ValueError(
        "File already exists at path: /reports/summary.pdf. Pass overwrite=true"
    )
    with _patch_manager(mock_manager):
        resp = await MoveWorkspaceFileTool()._execute(
            user_id=_USER,
            session=_session(),
            file_id="file-1",
            new_path="/reports/summary.pdf",
        )
    assert isinstance(resp, ErrorResponse)
    assert "already exists at path" in resp.message


@pytest.mark.asyncio
async def test_copy_quota_error_includes_remediation_hint(mock_manager):
    mock_manager.copy_file.side_effect = ValueError(
        "Storage limit exceeded. You've used 1 GB of your 1 GB quota."
    )
    with _patch_manager(mock_manager):
        resp = await CopyWorkspaceFileTool()._execute(
            user_id=_USER,
            session=_session(),
            file_id="file-1",
            new_path="/reports/summary.pdf",
        )
    assert isinstance(resp, ErrorResponse)
    assert "delete_workspace_file to free space" in resp.message


@pytest.mark.asyncio
async def test_overwrite_and_folder_id_are_forwarded(mock_manager):
    with _patch_manager(mock_manager):
        await MoveWorkspaceFileTool()._execute(
            user_id=_USER,
            session=_session(),
            file_id="file-1",
            new_path="/reports/summary.pdf",
            folder_id="fld-1",
            overwrite=True,
        )

    kwargs = mock_manager.move_file.await_args.kwargs
    assert kwargs["overwrite"] is True
    assert kwargs["folder_id"] == "fld-1"


@pytest.mark.asyncio
async def test_move_reports_a_noop_when_already_at_destination(mock_manager):
    """A same-path move is not an error, but the message must not claim a move."""
    mock_manager.move_file.return_value = _file()

    with _patch_manager(mock_manager):
        resp = await MoveWorkspaceFileTool()._execute(
            user_id=_USER,
            session=_session(),
            file_id="file-1",
            new_path="/report.pdf",
        )

    assert isinstance(resp, WorkspaceFileMovedResponse)
    assert "already at" in resp.message
