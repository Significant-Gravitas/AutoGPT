"""The three things a task's worker says about its own work: where it is, the
proof for what it claims, and the result when it is done. Everything else on
the record is written by the harness (``copilot.task_recorder``).

The user skims a task, so each of these is short and the caps are enforced
here: an over-long line is refused with the limit, not stored.
"""

from typing import Any

from backend.copilot import turn_end
from backend.copilot.model import ChatSession
from backend.data.db_accessors import task_record_db, task_result_db
from backend.data.task_record import EvidenceInput
from backend.data.task_text import (
    CLAIM_MAX_CHARS,
    PROGRESS_MAX_CHARS,
    SUMMARY_MAX_CHARS,
    clip,
    too_long,
)

from .base import BaseTool
from .external_scope import error_detail
from .models import ErrorResponse, ToolResponseBase
from .task_models import (
    TaskEvidenceAddedResponse,
    TaskFinishedResponse,
    TaskProgressResponse,
)

FINISHED = "Result saved on the task. The task is finished."
FINISHED_IN_CHAT = (
    "Result saved on the task. End now with one line of a few words ('Brief "
    "saved.'); the summary already says the rest, so do not repeat or add to it."
)
NO_TASK = (
    "This chat is not working on a task, so there is nothing to report on. "
    "Carry on with the work."
)


class ReportProgressTool(BaseTool):
    """Say where the task is and what comes next."""

    @property
    def name(self) -> str:
        return "report_progress"

    @property
    def description(self) -> str:
        return (
            "While working on a task: say what you are doing now, in a few "
            "plain words the user skims ('Comparing pricing pages'). No "
            "findings, numbers or tool names. Call it when you start a new "
            "part of the work."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status_text": {
                    "type": "string",
                    "description": (
                        f"What you are doing now. At most {PROGRESS_MAX_CHARS} "
                        "characters."
                    ),
                },
                "next_step": {
                    "type": "string",
                    "description": "What comes after, in a few words.",
                },
            },
            "required": ["status_text"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        status_text: str = "",
        next_step: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        assert user_id is not None  # guaranteed by requires_auth
        status_text = status_text or _said(kwargs, "message", "text", "status")
        if not status_text.strip():
            return _error("Say what you are doing now in status_text.", session)
        if refusal := too_long(status_text, PROGRESS_MAX_CHARS, "progress line"):
            return _error(refusal, session)
        try:
            task = await task_record_db().set_progress(
                session.session_id,
                user_id,
                status_text,
                clip(next_step, PROGRESS_MAX_CHARS),
                session.expert_id,
            )
        except Exception as e:
            return _error(error_detail(e, session), session)
        if task is None:
            return _error(NO_TASK, session)
        return TaskProgressResponse(
            message="Noted on the task. Carry on.", session_id=session.session_id
        )


class AddEvidenceTool(BaseTool):
    """Attach proof to a claim the task makes."""

    @property
    def name(self) -> str:
        return "add_evidence"

    @property
    def description(self) -> str:
        return (
            "While working on a task: attach proof that something happened "
            "('Reminder sent to Acme'). Name what shows it: the file you "
            "saved, the tool you ran or the page you read. It is checked "
            "against what this task really did."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": (
                        "What happened, as the user would say it. At most "
                        f"{CLAIM_MAX_CHARS} characters."
                    ),
                },
                "file": {
                    "type": "string",
                    "description": "Name of a file this task saved.",
                },
                "tool": {
                    "type": "string",
                    "description": "Name of one tool this task ran.",
                },
                "url": {"type": "string", "description": "A page this task read."},
                "quote": {
                    "type": "string",
                    "description": "A short quote of the proof. No secrets.",
                },
            },
            "required": ["claim"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        claim: str = "",
        tool: str | None = None,
        file: str | None = None,
        url: str | None = None,
        quote: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        assert user_id is not None  # guaranteed by requires_auth
        claim = claim or _said(kwargs, "description", "summary", "text")
        if not claim.strip():
            return _error("Say what happened in claim.", session)
        if refusal := too_long(claim, CLAIM_MAX_CHARS, "claim"):
            return _error(refusal, session)
        try:
            evidence = await task_record_db().add_evidence(
                session.session_id,
                user_id,
                EvidenceInput(claim=claim, tool=tool, file=file, url=url, quote=quote),
                session.expert_id,
            )
        except Exception as e:
            return _error(error_detail(e, session), session)
        if evidence is None:
            return _error(NO_TASK, session)
        return TaskEvidenceAddedResponse(
            message=(
                "Proof attached and verified."
                if evidence.verified
                else "Kept as unverified: nothing on this task's record matches. "
                "Name a file this task saved, a tool it ran or a page it read, "
                "or leave it unverified."
            ),
            evidence_id=evidence.id,
            verified=evidence.verified,
            session_id=session.session_id,
        )


class FinishTaskTool(BaseTool):
    """Leave the task's result: a short summary and the full document."""

    @property
    def name(self) -> str:
        return "finish_task"

    @property
    def description(self) -> str:
        return (
            "Finish the task you are working on: a short summary the user "
            "reads at a glance, and the saved file that holds the full "
            "result. Call it last: it ends a background task's turn."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "What was done and the one thing the user should "
                        "know: two or three short plain sentences, at most "
                        f"{SUMMARY_MAX_CHARS} characters."
                    ),
                },
                "file": {
                    "type": "string",
                    "description": (
                        "Name of the file you saved with the full result. "
                        "Omit when the task made no document."
                    ),
                },
            },
            "required": ["summary"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        summary: str = "",
        file: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        assert user_id is not None  # guaranteed by requires_auth
        summary = summary or _said(kwargs, "result", "message", "text")
        file = file or _said(kwargs, "filename", "document", "path") or None
        if not summary.strip():
            return _error("Say what was done in summary.", session)
        if refusal := too_long(summary, SUMMARY_MAX_CHARS, "summary"):
            return _error(f"{refusal} The detail belongs in the file.", session)
        try:
            finish = await task_result_db().finish_task(
                session.session_id, user_id, summary, file
            )
        except Exception as e:
            return _error(error_detail(e, session), session)
        if finish is None:
            return _error(NO_TASK, session)
        if finish.missing_file:
            return _error(
                f"This task saved no file named '{file}', so nothing was "
                "finished. Save the full result with write_workspace_file, then "
                "call finish_task with that file's name; leave file out when "
                "there is no document.",
                session,
            )
        if finish.in_background:
            # Nobody reads a worker's closing message: the turn ends here.
            turn_end.request_finish(session.session_id)
        return TaskFinishedResponse(
            message=FINISHED if finish.in_background else FINISHED_IN_CHAT,
            document=finish.document_title,
            session_id=session.session_id,
        )


def _said(kwargs: dict[str, Any], *names: str) -> str:
    """The value under another name a worker reaches for: these tools are
    called by id with no schema in view, so 'message' comes as often as
    'status_text'."""
    return next(
        (
            value
            for name in names
            if isinstance(value := kwargs.get(name), str) and value.strip()
        ),
        "",
    )


def _error(message: str, session: ChatSession) -> ErrorResponse:
    return ErrorResponse(message=message, session_id=session.session_id)
