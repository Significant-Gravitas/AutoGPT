"""Tool for finalizing non-manual Copilot sessions."""

from typing import Any

from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.copilot.model import ChatSession
from backend.copilot.session_types import CompletionReportInput
from backend.data.db_accessors import review_db

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
            # Keep the tool contract strict so the model always returns a complete
            # structured payload. CompletionReportInput remains the semantic validator
            # for which fields are actually required based on should_notify_user.
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

        pending_approval_count = await review_db().count_pending_reviews_for_graph_exec(
            f"{COPILOT_SESSION_PREFIX}{session.session_id}",
            session.user_id,
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
