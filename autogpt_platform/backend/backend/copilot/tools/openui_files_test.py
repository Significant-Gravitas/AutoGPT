"""write_workspace_file / read_workspace_file on OpenUI view files."""

import base64
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.models import ChatSession, ChatUIDraft, UserWorkspaceFile

from backend.copilot.db import create_chat_session
from backend.copilot.tools import openui_files
from backend.copilot.tools._test_data import make_session, setup_test_data
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.openui_files import OpenUIFileContentResponse, read_view_file
from backend.copilot.tools.workspace_files import (
    DeleteWorkspaceFileTool,
    ReadWorkspaceFileTool,
    WorkspaceFileContentResponse,
    WorkspaceWriteResponse,
    WriteWorkspaceFileTool,
)
from backend.data.workspace import OPENUI_MIME_TYPE, WorkspaceFile
from backend.util.json import SafeJson

# Re-export so pytest discovers the session-scoped fixture
setup_test_data = setup_test_data

SOURCE = 'root = Workspace("Revenue", "From the uploaded report", [Metrics([Metric("Revenue", "$42,000", "Q3", "positive")])], "revenue")'


@pytest.mark.asyncio(loop_scope="session")
async def test_view_files_are_saved_revised_in_place_and_read_with_edits(
    setup_test_data,
):
    user = setup_test_data["user"]
    session = make_session(user.id)
    write, read = WriteWorkspaceFileTool(), ReadWorkspaceFileTool()
    written = await write._execute(
        user_id=user.id,
        session=session,
        filename="revenue",
        mime_type=OPENUI_MIME_TYPE,
        content=SOURCE,
    )
    assert isinstance(written, WorkspaceWriteResponse), written.message
    try:
        assert written.name == "revenue.openui"
        assert written.mime_type == OPENUI_MIME_TYPE
        record = await UserWorkspaceFile.prisma().find_unique(
            where={"id": written.file_id}
        )
        assert record is not None and isinstance(record.metadata, dict)
        openui = record.metadata["openui"]
        assert isinstance(openui, dict)
        assert openui["viewId"] == "revenue"
        assert openui["originSessionId"] == session.session_id
        assert "Revenue: $42,000 (Q3)" in str(openui["summary"])

        revised = await write._execute(
            user_id=user.id,
            session=session,
            filename="revenue-v2.openui",
            content=SOURCE.replace("$42,000", "$43,000"),
        )
        assert isinstance(revised, WorkspaceWriteResponse), revised.message
        assert (revised.file_id, revised.name) == (written.file_id, "revenue.openui")

        edits = {"revenue": {"note": {"value": "Checked", "componentType": "Field"}}}
        await ChatUIDraft.prisma().create(
            data={
                "fileId": written.file_id,
                "viewId": "revenue",
                "sequence": 0,
                "source": SOURCE,
                "state": SafeJson(edits),
                "revision": 1,
            }
        )
        result = await read._execute(
            user_id=user.id, session=session, file_id=written.file_id
        )
        assert isinstance(result, OpenUIFileContentResponse), result.message
        assert "$43,000" in result.source
        assert "Revenue: $43,000 (Q3)" in result.summary
        assert result.view_id == "revenue"
        assert result.user_edits == edits
        assert f"workspace://{written.file_id}" in result.message

        ranged = await read._execute(
            user_id=user.id, session=session, file_id=written.file_id, length=10
        )
        assert isinstance(ranged, WorkspaceFileContentResponse)
        assert base64.b64decode(ranged.content_base64).decode() == SOURCE[:10]
    finally:
        await ChatUIDraft.prisma().delete_many(where={"fileId": written.file_id})
        await DeleteWorkspaceFileTool()._execute(
            user_id=user.id, session=session, file_id=written.file_id
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_new_view_file_takes_over_the_chats_edits(setup_test_data):
    user = setup_test_data["user"]
    session = make_session(user.id)
    await create_chat_session(session.session_id, user.id)
    edits = {"revenue": {"note": {"value": "Checked", "componentType": "Field"}}}
    draft = await ChatUIDraft.prisma().create(
        data={
            "sessionId": session.session_id,
            "viewId": "revenue",
            "toolCallId": "view-one",
            "sequence": 1,
            "source": SOURCE,
            "state": SafeJson(edits),
            "revision": 2,
        }
    )
    write = WriteWorkspaceFileTool()
    written = await write._execute(
        user_id=user.id, session=session, filename="revenue.openui", content=SOURCE
    )
    assert isinstance(written, WorkspaceWriteResponse), written.message
    try:
        moved = await ChatUIDraft.prisma().find_unique(where={"id": draft.id})
        assert moved is not None
        assert (moved.sessionId, moved.fileId) == (None, written.file_id)
        assert (moved.state, moved.revision) == (edits, 2)

        revised = await write._execute(
            user_id=user.id,
            session=session,
            filename="revenue.openui",
            content=SOURCE.replace("$42,000", "$43,000"),
        )
        assert isinstance(revised, WorkspaceWriteResponse), revised.message
        assert revised.file_id == written.file_id
        rows = await ChatUIDraft.prisma().find_many(where={"viewId": "revenue"})
        assert [row.id for row in rows if row.fileId == written.file_id] == [draft.id]
        assert not [row for row in rows if row.sessionId == session.session_id]
    finally:
        await ChatUIDraft.prisma().delete_many(where={"fileId": written.file_id})
        await ChatSession.prisma().delete(where={"id": session.session_id})
        await DeleteWorkspaceFileTool()._execute(
            user_id=user.id, session=session, file_id=written.file_id
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, issue",
    [
        ("# Trip notes\n- Pack", "root = Workspace"),
        (SOURCE.replace(', "revenue")', ")"), "stable viewId"),
        (
            'root = Workspace("Test", "Data", [MadeUpComponent("hi")], "test")',
            "write_workspace_file",
        ),
    ],
    ids=["markdown", "no-view-id", "unknown-component"],
)
async def test_view_files_must_be_valid_programs(content, issue):
    manager = AsyncMock()
    with patch(
        "backend.copilot.tools.workspace_files.get_workspace_manager",
        AsyncMock(return_value=manager),
    ):
        result = await WriteWorkspaceFileTool()._execute(
            user_id="owner",
            session=make_session("owner"),
            filename="notes.openui",
            content=content,
        )
    assert isinstance(result, ErrorResponse)
    assert issue in result.message
    manager.write_file.assert_not_awaited()
    manager.replace_file_content.assert_not_awaited()


def _stored_view(openui: dict) -> WorkspaceFile:
    return WorkspaceFile(
        id="file-1",
        workspace_id="ws-1",
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        name="revenue.openui",
        path="/revenue.openui",
        storage_path="local://revenue.openui",
        mime_type=OPENUI_MIME_TYPE,
        size_bytes=len(SOURCE),
        metadata={"openui": openui},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "openui, validated",
    [
        ({"viewId": "revenue", "summary": "Stored summary"}, False),
        ({"viewId": "revenue"}, True),
    ],
    ids=["stored", "older-file"],
)
async def test_reading_a_view_uses_its_stored_summary(monkeypatch, openui, validated):
    db = MagicMock()
    db.get_openui_file_state = AsyncMock(return_value=None)
    monkeypatch.setattr(openui_files, "openui_files_db", MagicMock(return_value=db))
    validator = AsyncMock(wraps=openui_files.validate_openui_source)
    monkeypatch.setattr(openui_files, "validate_openui_source", validator)
    result = await read_view_file(
        "owner", "chat", _stored_view(openui), SOURCE.encode()
    )
    if validated:
        validator.assert_awaited_once_with(SOURCE)
        assert "Revenue: $42,000 (Q3)" in result.summary
    else:
        validator.assert_not_awaited()
        assert result.summary == "Stored summary"
