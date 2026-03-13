"""Tool for finalizing non-manual Copilot sessions."""

from typing import Any

from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview

from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.copilot.model import ChatSession
from backend.copilot.session_types import CompletionReportInput

from .base import BaseTool
from .models import CompletionReportSavedResponse, ErrorResponse, ToolResponseBase


class CompletionReportTool(BaseTool):
    @property
    def name(self) -> str:
        return "completion_report"

    @property
    def description(self) -> str:
        return (
            "Finalize a non-manual session after you have finished the work. "
            "Use this exactly once at the end of the flow. "
            "Summarize what you did, state whether the user should be notified, "
            "and provide any email/callback content that should be used."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        schema = CompletionReportInput.model_json_schema()
        return {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": [
                "thoughts",
                "should_notify_user",
                "email_title",
                "email_body",
                "callback_session_message",
                "approval_summary",
            ],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        if session.is_manual:
            return ErrorResponse(
                message="completion_report is only available in non-manual sessions.",
                session_id=session.session_id,
            )

        try:
            report = CompletionReportInput.model_validate(kwargs)
        except Exception as exc:
            return ErrorResponse(
                message="completion_report arguments are invalid.",
                error=str(exc),
                session_id=session.session_id,
            )

        pending_approval_count = await PendingHumanReview.prisma().count(
            where={
                "graphExecId": f"{COPILOT_SESSION_PREFIX}{session.session_id}",
                "status": ReviewStatus.WAITING,
            }
        )

        if pending_approval_count > 0 and not report.approval_summary:
            return ErrorResponse(
                message=(
                    "approval_summary is required because this session has pending approvals."
                ),
                session_id=session.session_id,
            )

        return CompletionReportSavedResponse(
            message="Completion report recorded successfully.",
            session_id=session.session_id,
            has_pending_approvals=pending_approval_count > 0,
            pending_approval_count=pending_approval_count,
        )
