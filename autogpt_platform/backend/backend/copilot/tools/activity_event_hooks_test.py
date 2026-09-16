"""Tests for the activity-event drafts side-effecting tools report."""

from datetime import UTC, datetime

from backend.copilot.model import ChatSession
from backend.copilot.tools.models import BlockOutputResponse, ErrorResponse
from backend.copilot.tools.run_block import RunBlockTool
from backend.copilot.tools.schedule_followup import (
    ScheduleCreatedResponse,
    ScheduleFollowupTool,
)
from backend.copilot.tools.workspace_files import (
    WorkspaceWriteResponse,
    WriteWorkspaceFileTool,
)


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


def _write_response(path: str) -> WorkspaceWriteResponse:
    return WorkspaceWriteResponse(
        message="File written",
        file_id="file-1",
        name="draft.md",
        path=path,
        mime_type="text/markdown",
        size_bytes=42,
        download_url="workspace://file-1#text/markdown",
    )


def test_write_workspace_file_reports_file_event() -> None:
    result = _write_response("/sessions/s1/draft.md")

    draft = WriteWorkspaceFileTool().activity_event(
        session=_make_session(), result=result, filename="draft.md"
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
        session=_make_session(), result=result, filename="draft.md", overwrite=True
    )

    assert draft is not None
    assert draft.event_type == "file.updated"


def test_error_response_reports_nothing() -> None:
    draft = WriteWorkspaceFileTool().activity_event(
        session=_make_session(),
        result=ErrorResponse(message="boom", session_id="test-session"),
        filename="draft.md",
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

    draft = tool.activity_event(session=session, result=with_provider)
    assert draft is not None
    assert draft.category == "INTEGRATION"
    assert draft.provider == "google"
    assert draft.title == "Send Email"

    assert tool.activity_event(session=session, result=without_provider) is None
    assert tool.activity_event(session=session, result=dry_run) is None


def test_schedule_followup_reports_schedule_event() -> None:
    result = ScheduleCreatedResponse(
        message="Follow-up scheduled",
        schedule_id="sched-1",
        next_run_time="2026-08-29T07:00:00+00:00",
        is_recurring=True,
        session_id="test-session",
    )

    draft = ScheduleFollowupTool().activity_event(
        session=_make_session(expert_id="maria"),
        result=result,
        message="Draft the next blog post",
        cron="0 7 */3 * *",
        name="persian.sh blog draft",
    )

    assert draft is not None
    assert draft.category == "SCHEDULE"
    assert draft.event_type == "schedule.created"
    assert draft.title == "persian.sh blog draft"
    assert draft.schedule_id == "sched-1"
    assert draft.data["cron"] == "0 7 */3 * *"
