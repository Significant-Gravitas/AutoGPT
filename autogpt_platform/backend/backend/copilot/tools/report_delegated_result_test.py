from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.experts.models import ExpertWorkArtifact, ExpertWorkItem

from .models import (
    DelegatedWorkReportedResponse,
    ErrorResponse,
    WorkspaceFileInfoData,
)
from .report_delegated_result import ReportDelegatedResultTool

NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


def _session(*, expert_id: str | None = "expert-1") -> MagicMock:
    session = MagicMock()
    session.session_id = "delegated-1"
    session.expert_id = expert_id
    session.metadata.delegated_by_session_id = "manager-1"
    return session


def _item(*, deliverable_mode: str = "message") -> ExpertWorkItem:
    return ExpertWorkItem(
        id="work-1",
        expert_id="expert-1",
        manager_session_id="manager-1",
        delegated_session_id="delegated-1",
        project_phase="Research",
        task_title="Find launch partners",
        expected_deliverable="A verified partner shortlist",
        deliverable_mode=deliverable_mode,
        success_criteria=[],
        dependencies=[],
        source_artifacts=[],
        constraints=[],
        approval_boundaries=[],
        estimate_minutes=30,
        progress=0,
        status="running",
        result=None,
        blocker=None,
        confidence="unknown",
        artifacts=[],
        created_at=NOW,
        updated_at=NOW,
        started_at=NOW,
        completed_at=None,
        link="/copilot?sessionId=delegated-1&workItemId=work-1",
    )


@pytest.fixture
def dependencies(monkeypatch):
    item = _item()
    enabled = AsyncMock(return_value=True)
    get_item = AsyncMock(return_value=item)
    report = AsyncMock(
        return_value=(item.model_copy(update={"status": "delivered"}), True)
    )
    should_wake = AsyncMock(return_value=True)
    queue = AsyncMock(return_value=("queued", MagicMock()))
    monkeypatch.setattr(
        "backend.copilot.tools.report_delegated_result.is_feature_enabled", enabled
    )
    monkeypatch.setattr(
        "backend.copilot.tools.report_delegated_result.work_items.get_work_item",
        get_item,
    )
    monkeypatch.setattr(
        "backend.copilot.tools.report_delegated_result.work_items.report_work_item",
        report,
    )
    monkeypatch.setattr(
        "backend.copilot.tools.report_delegated_result.work_items.should_enqueue_parent_wake",
        should_wake,
    )
    monkeypatch.setattr(
        "backend.copilot.tools.report_delegated_result.run_copilot_turn_via_queue",
        queue,
    )
    monkeypatch.setattr(
        "backend.copilot.tools.report_delegated_result.list_sub_workspace_files",
        AsyncMock(return_value=[]),
    )
    return item, enabled, get_item, report, should_wake, queue


@pytest.mark.asyncio
async def test_flag_off_rejects_execution(dependencies):
    _, enabled, get_item, *_ = dependencies
    enabled.return_value = False

    response = await ReportDelegatedResultTool()._execute(
        "user-1",
        _session(),
        work_item_id="work-1",
        status="delivered",
        summary="Done",
    )

    assert isinstance(response, ErrorResponse)
    get_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_plain_session_cannot_report(dependencies):
    response = await ReportDelegatedResultTool()._execute(
        "user-1",
        _session(expert_id=None),
        work_item_id="work-1",
        status="delivered",
        summary="Done",
    )

    assert isinstance(response, ErrorResponse)


@pytest.mark.asyncio
async def test_wrong_expert_work_item_is_rejected(dependencies):
    item, _, _, report, *_ = dependencies
    item.expert_id = "expert-2"

    response = await ReportDelegatedResultTool()._execute(
        "user-1",
        _session(),
        work_item_id="work-1",
        status="delivered",
        summary="Done",
    )

    assert isinstance(response, ErrorResponse)
    report.assert_not_awaited()


@pytest.mark.asyncio
async def test_required_files_without_workspace_artifact_become_partial(
    dependencies,
):
    item, _, get_item, report, *_ = dependencies
    file_item = _item(deliverable_mode="workspace_files")
    get_item.return_value = file_item
    report.return_value = (
        file_item.model_copy(update={"status": "partial"}),
        True,
    )

    response = await ReportDelegatedResultTool()._execute(
        "user-1",
        _session(),
        work_item_id="work-1",
        status="delivered",
        summary="Files are done",
    )

    assert isinstance(response, DelegatedWorkReportedResponse)
    assert report.await_args.kwargs["status"] == "partial"
    assert "not promoted" in report.await_args.kwargs["blocker"]


@pytest.mark.asyncio
async def test_promoted_files_are_persisted_as_workspace_uris(dependencies):
    item, _, get_item, report, _, queue = dependencies
    file_item = _item(deliverable_mode="workspace_files")
    artifact = ExpertWorkArtifact(
        name="launch-plan.md",
        uri="workspace://file-1#text/markdown",
        mime_type="text/markdown",
        size_bytes=1200,
    )
    get_item.return_value = file_item
    report.return_value = (
        file_item.model_copy(update={"status": "delivered", "artifacts": [artifact]}),
        True,
    )
    files = AsyncMock(
        return_value=[
            WorkspaceFileInfoData(
                file_id="file-1",
                name="launch-plan.md",
                path="/sessions/delegated-1/launch-plan.md",
                mime_type="text/markdown",
                size_bytes=1200,
            )
        ]
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "backend.copilot.tools.report_delegated_result.list_sub_workspace_files",
            files,
        )
        await ReportDelegatedResultTool()._execute(
            "user-1",
            _session(),
            work_item_id="work-1",
            status="delivered",
            summary="Files are ready",
        )

    assert report.await_args.kwargs["artifacts"] == [artifact]
    assert "workspace://file-1#text/markdown" in queue.await_args.kwargs["message"]


@pytest.mark.asyncio
async def test_repeated_terminal_report_wakes_parent_exactly_once(dependencies):
    item, _, _, report, should_wake, queue = dependencies
    delivered = item.model_copy(update={"status": "delivered"})
    report.side_effect = [(delivered, True), (delivered, False)]

    for _ in range(2):
        await ReportDelegatedResultTool()._execute(
            "user-1",
            _session(),
            work_item_id="work-1",
            status="delivered",
            summary="2 verified partners",
        )

    should_wake.assert_awaited_once_with("work-1", "user-1")
    queue.assert_awaited_once()
    assert queue.await_args.kwargs["session_id"] == "manager-1"
