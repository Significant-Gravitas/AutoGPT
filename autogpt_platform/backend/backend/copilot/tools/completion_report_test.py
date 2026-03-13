from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.session_types import ChatSessionStartType
from backend.copilot.tools.completion_report import CompletionReportTool
from backend.copilot.tools.models import CompletionReportSavedResponse, ResponseType


@pytest.mark.asyncio
async def test_completion_report_rejects_manual_sessions() -> None:
    tool = CompletionReportTool()
    session = ChatSession.new("user-1")

    response = await tool._execute(
        user_id="user-1",
        session=session,
        thoughts="Wrapped up the session.",
        should_notify_user=False,
        email_title=None,
        email_body=None,
        callback_session_message=None,
        approval_summary=None,
    )

    assert response.type == ResponseType.ERROR
    assert "non-manual sessions" in response.message


@pytest.mark.asyncio
async def test_completion_report_requires_approval_summary_when_pending(
    mocker,
) -> None:
    tool = CompletionReportTool()
    session = ChatSession.new(
        "user-1",
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
    )

    pending_reviews = Mock()
    pending_reviews.count = AsyncMock(return_value=2)
    mocker.patch(
        "backend.copilot.tools.completion_report.PendingHumanReview.prisma",
        return_value=pending_reviews,
    )

    response = await tool._execute(
        user_id="user-1",
        session=session,
        thoughts="Prepared a recommendation for the user.",
        should_notify_user=True,
        email_title="Your nightly update",
        email_body="I found something worth reviewing.",
        callback_session_message="Let's review the next step together.",
        approval_summary=None,
    )

    assert response.type == ResponseType.ERROR
    assert "approval_summary is required" in response.message


@pytest.mark.asyncio
async def test_completion_report_succeeds_without_pending_approvals(
    mocker,
) -> None:
    tool = CompletionReportTool()
    session = ChatSession.new(
        "user-1",
        start_type=ChatSessionStartType.AUTOPILOT_CALLBACK,
    )

    pending_reviews = Mock()
    pending_reviews.count = AsyncMock(return_value=0)
    mocker.patch(
        "backend.copilot.tools.completion_report.PendingHumanReview.prisma",
        return_value=pending_reviews,
    )

    response = await tool._execute(
        user_id="user-1",
        session=session,
        thoughts="Reviewed the account and prepared a useful follow-up.",
        should_notify_user=True,
        email_title="Autopilot found something useful",
        email_body="I put together a recommendation for you.",
        callback_session_message="Open this chat and I will walk you through it.",
        approval_summary=None,
    )

    assert response.type == ResponseType.COMPLETION_REPORT_SAVED
    response = cast(CompletionReportSavedResponse, response)
    assert response.has_pending_approvals is False
    assert response.pending_approval_count == 0
