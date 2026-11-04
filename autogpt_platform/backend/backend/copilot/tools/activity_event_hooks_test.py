"""Tests for the activity-event drafts side-effecting tools report."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from backend.blocks.airtable.schema import AirtableCreateTableBlock
from backend.copilot.model import ChatSession
from backend.copilot.tools.base import BaseTool, _record_activity
from backend.copilot.tools.models import (
    BlockOutputResponse,
    ErrorResponse,
    ToolResponseBase,
)
from backend.copilot.tools.run_block import RunBlockTool
from backend.copilot.tools.workspace_files import (
    WorkspaceWriteResponse,
    WriteWorkspaceFileTool,
)
from backend.data.activity_event import ActivityEventDraft


def _make_session(expert_id: str | None = None) -> ChatSession:
    return ChatSession(
        session_id="test-session",
        user_id="test-user",
        title=None,
        messages=[],
        usage=[],
        credentials={},
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        expert_id=expert_id,
    )


def _write_response(path: str, is_internal: bool = False) -> WorkspaceWriteResponse:
    return WorkspaceWriteResponse(
        message="File written",
        file_id="file-1",
        name="draft.md",
        is_internal=is_internal,
        path=path,
        mime_type="text/markdown",
        size_bytes=42,
        download_url="workspace://file-1#text/markdown",
    )


def test_write_workspace_file_reports_file_event() -> None:
    result = _write_response("/sessions/s1/draft.md")

    draft = WriteWorkspaceFileTool().activity_event(
        session=_make_session(), result=result, args={"filename": "draft.md"}
    )

    assert draft is not None
    assert draft.category == "FILE"
    assert draft.event_type == "file.created"
    assert draft.title == "draft.md"
    assert draft.object_id == "file-1"
    assert draft.data["mime_type"] == "text/markdown"


def test_write_with_overwrite_reports_update() -> None:
    result = _write_response("/draft.md")

    draft = WriteWorkspaceFileTool().activity_event(
        session=_make_session(),
        result=result,
        args={"filename": "draft.md", "overwrite": True},
    )

    assert draft is not None
    assert draft.event_type == "file.updated"


def test_writing_a_working_file_is_not_reported() -> None:
    result = _write_response("/sessions/s1/build_state.json", is_internal=True)

    draft = WriteWorkspaceFileTool().activity_event(
        session=_make_session(),
        result=result,
        args={"filename": "build_state.json"},
    )

    assert draft is None
    # The model and the client never see the flag.
    assert "is_internal" not in result.model_dump()


def test_error_response_reports_nothing() -> None:
    draft = WriteWorkspaceFileTool().activity_event(
        session=_make_session(),
        result=ErrorResponse(message="boom", session_id="test-session"),
        args={"filename": "draft.md"},
    )

    assert draft is None


def test_run_block_reports_integration_action_only_with_provider() -> None:
    tool = RunBlockTool()
    session = _make_session()
    with_provider = BlockOutputResponse(
        message="Block executed",
        block_id="block-1",
        block_name="Send Email",
        outputs={},
        provider="google",
        session_id="test-session",
    )
    without_provider = with_provider.model_copy(update={"provider": None})
    dry_run = with_provider.model_copy(update={"is_dry_run": True})

    draft = tool.activity_event(session=session, result=with_provider, args={})
    assert draft is not None
    assert draft.category == "INTEGRATION"
    assert draft.provider == "google"
    assert draft.title == "Send Email"

    assert (
        tool.activity_event(session=session, result=without_provider, args={}) is None
    )
    assert tool.activity_event(session=session, result=dry_run, args={}) is None


def test_run_block_does_not_report_llm_credential_use() -> None:
    tool = RunBlockTool()
    session = _make_session()
    model_call = BlockOutputResponse(
        message="Block executed",
        block_id="block-1",
        block_name="AITextGeneratorBlock",
        outputs={},
        provider="openai",
        session_id="test-session",
    )

    assert tool.activity_event(session=session, result=model_call, args={}) is None


def test_run_block_names_the_action_by_its_block() -> None:
    block = AirtableCreateTableBlock()
    result = BlockOutputResponse(
        message="Block executed",
        block_id=block.id,
        block_name=block.name,
        outputs={},
        provider="airtable",
        session_id="test-session",
    )

    draft = RunBlockTool().activity_event(
        session=_make_session(), result=result, args={}
    )

    assert draft is not None
    assert draft.title == "Airtable Create Table"


class _ResultArgTool(BaseTool):
    """A tool whose own argument is named ``result``, like the call's result."""

    @property
    def name(self) -> str:
        return "record_result"

    def activity_event(
        self,
        session: ChatSession,
        result: ToolResponseBase,
        args: dict[str, Any],
    ) -> ActivityEventDraft | None:
        return ActivityEventDraft(
            category="FILE",
            event_type="file.created",
            title=str(args["result"]),
        )


async def test_a_tool_argument_named_result_still_records_its_event() -> None:
    db = MagicMock()
    db.create_activity_event = AsyncMock()
    with patch("backend.copilot.tools.base.activity_event_db", return_value=db):
        await _record_activity(
            _ResultArgTool(),
            "test-user",
            _make_session(expert_id="expert-1"),
            ErrorResponse(message="ok", session_id="test-session"),
            {"result": "summary.md"},
        )

    db.create_activity_event.assert_awaited_once()
    draft = db.create_activity_event.await_args.kwargs["draft"]
    assert draft.title == "summary.md"
    assert draft.session_id == "test-session"
    assert draft.expert_id == "expert-1"
