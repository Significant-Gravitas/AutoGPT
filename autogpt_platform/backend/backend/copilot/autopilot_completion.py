from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field, ValidationError

from backend.copilot.autopilot_dispatch import (
    _enqueue_session_turn,
    get_graph_exec_id_for_session,
)
from backend.copilot.autopilot_prompts import (
    MAX_COMPLETION_REPORT_REPAIRS,
    wrap_internal_message,
)
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    upsert_chat_session,
)
from backend.copilot.session_types import CompletionReportInput, StoredCompletionReport
from backend.data.db_accessors import review_db

logger = logging.getLogger(__name__)


# --------------- models --------------- #


class CompletionReportToolCallFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class CompletionReportToolCall(BaseModel):
    id: str
    function: CompletionReportToolCallFunction = Field(
        default_factory=CompletionReportToolCallFunction
    )


class ToolOutputEnvelope(BaseModel):
    type: str | None = None


# --------------- approval metadata --------------- #


async def _get_pending_approval_metadata(
    session: ChatSession,
) -> tuple[int, str | None]:
    graph_exec_id = get_graph_exec_id_for_session(session.session_id)
    pending_count = await review_db().count_pending_reviews_for_graph_exec(
        graph_exec_id,
        session.user_id,
    )
    return pending_count, graph_exec_id if pending_count > 0 else None


# --------------- extraction --------------- #


def _extract_completion_report_from_session(
    session: ChatSession,
    *,
    pending_approval_count: int,
) -> CompletionReportInput | None:
    tool_outputs = {
        message.tool_call_id: message.content
        for message in session.messages
        if message.role == "tool" and message.tool_call_id
    }

    latest_report: CompletionReportInput | None = None
    for message in session.messages:
        if message.role != "assistant" or not message.tool_calls:
            continue

        for tool_call in message.tool_calls:
            try:
                parsed_tool_call = CompletionReportToolCall.model_validate(tool_call)
            except ValidationError:
                continue

            if parsed_tool_call.function.name != "completion_report":
                continue

            output = tool_outputs.get(parsed_tool_call.id)
            if not output:
                continue

            try:
                output_payload = ToolOutputEnvelope.model_validate_json(output)
            except ValidationError:
                output_payload = None

            if output_payload is not None and output_payload.type == "error":
                continue

            try:
                raw_arguments = parsed_tool_call.function.arguments or "{}"
                report = CompletionReportInput.model_validate_json(raw_arguments)
            except ValidationError:
                continue

            if pending_approval_count > 0 and not report.approval_summary:
                continue

            latest_report = report

    return latest_report


# --------------- repair --------------- #


def _build_completion_report_repair_message(
    *,
    attempt: int,
    pending_approval_count: int,
) -> str:
    approval_instruction = ""
    if pending_approval_count > 0:
        approval_instruction = (
            f" There are currently {pending_approval_count} pending approval item(s). "
            "If they still exist, include approval_summary."
        )

    return wrap_internal_message(
        "The session completed without a valid completion_report tool call. "
        f"This is repair attempt {attempt}. Call completion_report now and do not do any additional user-facing work."
        + approval_instruction
    )


async def _queue_completion_report_repair(
    session: ChatSession,
    *,
    pending_approval_count: int,
) -> None:
    attempt = session.completion_report_repair_count + 1
    repair_message = _build_completion_report_repair_message(
        attempt=attempt,
        pending_approval_count=pending_approval_count,
    )
    session.messages.append(ChatMessage(role="user", content=repair_message))
    session.completion_report_repair_count = attempt
    session.completion_report_repair_queued_at = datetime.now(UTC)
    session.completed_at = None
    session.completion_report = None
    await upsert_chat_session(session)
    await _enqueue_session_turn(
        session,
        message=repair_message,
        tool_name="completion_report_repair",
    )


# --------------- handler --------------- #


async def handle_non_manual_session_completion(session_id: str) -> None:
    session = await get_chat_session(session_id)
    if session is None or session.is_manual:
        return

    pending_approval_count, graph_exec_id = await _get_pending_approval_metadata(
        session
    )
    report = _extract_completion_report_from_session(
        session,
        pending_approval_count=pending_approval_count,
    )

    if report is not None:
        session.completion_report = StoredCompletionReport(
            **report.model_dump(),
            has_pending_approvals=pending_approval_count > 0,
            pending_approval_count=pending_approval_count,
            pending_approval_graph_exec_id=graph_exec_id,
            saved_at=datetime.now(UTC),
        )
        session.completion_report_repair_queued_at = None
        session.completed_at = datetime.now(UTC)
        await upsert_chat_session(session)
        return

    if session.completion_report_repair_count >= MAX_COMPLETION_REPORT_REPAIRS:
        session.completion_report_repair_queued_at = None
        session.completed_at = datetime.now(UTC)
        await upsert_chat_session(session)
        return

    await _queue_completion_report_repair(
        session,
        pending_approval_count=pending_approval_count,
    )
