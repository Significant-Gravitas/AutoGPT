from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ChatSessionStartType(str, Enum):
    MANUAL = "MANUAL"
    AUTOPILOT_NIGHTLY = "AUTOPILOT_NIGHTLY"
    AUTOPILOT_CALLBACK = "AUTOPILOT_CALLBACK"
    AUTOPILOT_INVITE_CTA = "AUTOPILOT_INVITE_CTA"


class ChatSessionConfig(BaseModel):
    system_prompt_override: str | None = None
    initial_user_message: str | None = None
    initial_assistant_message: str | None = None
    extra_tools: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)

    def allows_tool(self, tool_name: str) -> bool:
        return tool_name in self.extra_tools

    def disables_tool(self, tool_name: str) -> bool:
        return tool_name in self.disabled_tools


class CompletionReportInput(BaseModel):
    thoughts: str
    should_notify_user: bool
    email_title: str | None = None
    email_body: str | None = None
    callback_session_message: str | None = None
    approval_summary: str | None = None

    @model_validator(mode="after")
    def validate_notification_fields(self) -> "CompletionReportInput":
        if self.should_notify_user:
            required_fields = {
                "email_title": self.email_title,
                "email_body": self.email_body,
                "callback_session_message": self.callback_session_message,
            }
            missing = [
                field_name for field_name, value in required_fields.items() if not value
            ]
            if missing:
                raise ValueError(
                    "Missing required notification fields: " + ", ".join(missing)
                )
        return self


class StoredCompletionReport(CompletionReportInput):
    has_pending_approvals: bool
    pending_approval_count: int
    pending_approval_graph_exec_id: str | None = None
    saved_at: datetime
