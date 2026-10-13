"""write_workspace_file / read_workspace_file on OpenUI view files."""

import base64
from unittest.mock import AsyncMock, patch

import pytest
from prisma.models import ChatUIDraft, UserWorkspaceFile

from backend.copilot.tools._test_data import make_session, setup_test_data
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.openui_files import OpenUIFileContentResponse
from backend.copilot.tools.workspace_files import (
    DeleteWorkspaceFileTool,
    ReadWorkspaceFileTool,
    WorkspaceFileContentResponse,
    WorkspaceWriteResponse,
    WriteWorkspaceFileTool,
)
from backend.data.workspace import OPENUI_MIME_TYPE
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
        assert record.metadata["openui"] == {
            "viewId": "revenue",
            "originSessionId": session.session_id,
        }

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
