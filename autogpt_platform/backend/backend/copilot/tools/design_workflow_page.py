"""DesignWorkflowPageTool - Otto composes a page for one of the user's workflows."""

import json
from typing import Any

from pydantic import ValidationError

from backend.api.features.library.model import LibraryAgent
from backend.api.features.workflow_pages.design import WorkflowPageDesign, check_design
from backend.api.features.workflow_pages.page import (
    PageLatestResult,
    ResolvedWorkflowPage,
)
from backend.api.features.workflow_pages.service import build_page
from backend.copilot.gate.subject import NO_OP, Subject
from backend.copilot.model import ChatSession
from backend.data.db_accessors import library_db, workflow_pages_db
from backend.util.exceptions import NotFoundError
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase
from .render_ui_models import compact_schema

DESIGN_WORKFLOW_PAGE_TOOL = "design_workflow_page"
# Tool data replays into context on later turns; the page shows the rest.
_MAX_OUTPUT_CHARS = 2_000
_MAX_REPORTED_PROBLEMS = 5


class WorkflowPageResponse(ToolResponseBase):
    """A workflow page for the chat to draw, previewed or saved."""

    type: ResponseType = ResponseType.WORKFLOW_PAGE
    page: ResolvedWorkflowPage
    saved: bool = False


class DesignWorkflowPageTool(BaseTool):
    has_gate_subject = True

    @property
    def name(self) -> str:
        return DESIGN_WORKFLOW_PAGE_TOOL

    @property
    def description(self) -> str:
        return (
            "Compose a page for one of the user's library workflows, which they "
            "open from the library to run it and read its latest result without "
            "chatting. Read the workflow's inputs and outputs first "
            "(find_library_agent). Show a preview (save false), and save only "
            "once the user applies it. Bind cards only to the workflow's own "
            "output names. Say 'workflow' in page text, never 'agent'."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "library_agent_id": {
                    "type": "string",
                    "description": "The workflow's library id.",
                },
                "page": {
                    **compact_schema(WorkflowPageDesign),
                    "description": "The page, top to bottom.",
                },
                "save": {
                    "type": "boolean",
                    "description": "Store the page. Only after the user applied it.",
                },
            },
            "required": ["library_agent_id", "page"],
        }

    async def gate_subject(
        self, user_id: str, session: ChatSession, args: dict[str, Any]
    ) -> Subject | None:
        """A preview stores nothing, so nothing asks; a save is a platform edit."""
        return None if args.get("save") is True else NO_OP

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id or not await is_feature_enabled(Flag.WORKFLOW_PAGES, user_id):
            return _error(
                "Workflow pages are not available.", "unavailable", session_id
            )
        try:
            design = WorkflowPageDesign.model_validate(kwargs.get("page"))
        except ValidationError as e:
            return _error(_describe(e), "invalid_page", session_id)
        library_agent_id = str(kwargs.get("library_agent_id") or "")
        try:
            agent: LibraryAgent = await library_db().get_library_agent(
                library_agent_id, user_id
            )
        except NotFoundError:
            return _error(
                "No workflow with that library id.", "workflow_not_found", session_id
            )
        problems = check_design(design, agent.input_schema, agent.output_schema)
        if problems:
            shown = "; ".join(
                f"section {p.section}: {p.message}"
                for p in problems[:_MAX_REPORTED_PROBLEMS]
            )
            return _error(f"Invalid page. {shown}", "invalid_page", session_id)

        save = kwargs.get("save") is True
        has_previous = False
        if save:
            record = await workflow_pages_db().save_workflow_page(
                user_id, library_agent_id, design.model_dump(exclude_none=True)
            )
            has_previous = record.previous_design is not None
        page = _clipped(
            await build_page(user_id, agent, design, has_previous=has_previous)
        )
        message = (
            "Saved. The page now opens with the workflow in the library."
            if save
            else "Preview shown to the user. Save it only once they apply it."
        )
        results = [s for s in page.sections if isinstance(s, PageLatestResult)]
        return WorkflowPageResponse(
            message=message, session_id=session_id, page=page, saved=save
        ).from_outside(*[(s.blocks, s.other_outputs) for s in results])


def _clipped(page: ResolvedWorkflowPage) -> ResolvedWorkflowPage:
    """*page* with each long output cut down for the transcript."""
    sections = [
        (
            s.model_copy(
                update={
                    "other_outputs": {
                        name: [_clip_value(v) for v in values]
                        for name, values in s.other_outputs.items()
                    }
                }
            )
            if isinstance(s, PageLatestResult)
            else s
        )
        for s in page.sections
    ]
    return page.model_copy(update={"sections": sections})


def _clip_value(value: Any) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= _MAX_OUTPUT_CHARS:
        return value
    return text[: _MAX_OUTPUT_CHARS - 1] + "…"


def _error(message: str, error: str, session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=message, error=error, session_id=session_id
    ).from_outside()


def _describe(error: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors(include_input=False, include_url=False)
    ]
    return f"Invalid page. {'; '.join(problems[:_MAX_REPORTED_PROBLEMS])}"


async def workflow_pages_disabled_tools(session: ChatSession) -> frozenset[str]:
    """Hide the tool where its preview card can't be seen, and without the flag."""
    hidden = frozenset({DESIGN_WORKFLOW_PAGE_TOOL})
    metadata = session.metadata
    if metadata.source_platform or metadata.text_only_result or not session.user_id:
        return hidden
    if metadata.delegated_by_session_id and not metadata.handed_off_from_expert_id:
        return hidden
    enabled = await is_feature_enabled(Flag.WORKFLOW_PAGES, session.user_id)
    return frozenset() if enabled else hidden
