"""DesignPageTool - Otto composes a page from the user's workflows and experts."""

import json
from typing import Any

from pydantic import ValidationError

from backend.api.features.pages.design import MAX_SECTIONS, PageDesign
from backend.api.features.pages.resolve import resolve_page
from backend.api.features.pages.resolved import (
    PageCollectionNumbers,
    PageCollectionTable,
    PageExpertLearned,
    PageExpertWork,
    PageLatestResult,
    PageRecentRuns,
    PageRunHistory,
    PageWaiting,
    ResolvedPage,
)
from backend.api.features.pages.service import (
    check,
    dated,
    resolve_record,
    stored_design,
)
from backend.copilot.gate.subject import NO_OP, Subject
from backend.copilot.model import ChatSession
from backend.data.db_accessors import pages_db
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase
from .render_ui_models import compact_schema

DESIGN_PAGE_TOOL = "design_page"
# Tool data replays into context on later turns; the page shows the rest.
_MAX_OUTPUT_CHARS = 2_000
_MAX_REPORTED_PROBLEMS = 5


class PageResponse(ToolResponseBase):
    """A page for the chat to draw, its stored design, or the section catalogue."""

    type: ResponseType = ResponseType.PAGE
    page: ResolvedPage | None = None
    design: dict[str, Any] | None = None
    catalogue: dict[str, Any] | None = None
    saved: bool = False


class DesignPageTool(BaseTool):
    has_gate_subject = True

    @property
    def name(self) -> str:
        return DESIGN_PAGE_TOOL

    @property
    def description(self) -> str:
        return (
            "Compose a page the user opens from Pages, built from any of their "
            "workflows and experts: forms to run workflows, latest results, run "
            "history, schedules, what an expert is doing and has learned, and "
            "approvals waiting on them. Call with no arguments first for the "
            "section catalogue; with page_id alone to read a stored page. Show a "
            "preview (save false) and save only once the user applies it, passing "
            "the revision you previewed as based_on. Say 'workflow' in page text, "
            "never 'agent'."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "A stored page to read or change; omit for a new page.",
                    "entity": None,
                },
                "page": {
                    "type": "object",
                    "description": "The whole page design, in the catalogue's shape.",
                },
                "pin_to_workflow": {
                    "type": "string",
                    "description": "A library workflow id whose library view opens on this page.",
                    "entity": "library_agent",
                },
                "save": {
                    "type": "boolean",
                    "description": "Store the page. Only after the user applied it.",
                },
                "based_on": {
                    "type": "integer",
                    "description": "The stored page's revision the preview showed.",
                },
            },
        }

    async def gate_subject(
        self, user_id: str, session: ChatSession, args: dict[str, Any]
    ) -> Subject | None:
        """A preview or a read stores nothing, so nothing asks; a save is a platform edit."""
        return None if args.get("save") is True else NO_OP

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id or not await is_feature_enabled(Flag.PAGES, user_id):
            return _error("Pages are not available.", "unavailable", session_id)
        page_id = str(kwargs.get("page_id") or "") or None
        raw_page = kwargs.get("page")
        if raw_page is None and page_id is None:
            return PageResponse(
                message="The section catalogue. Compose a page in this shape.",
                session_id=session_id,
                catalogue=catalogue(),
            ).from_outside()
        record = await pages_db().get_page(user_id, page_id) if page_id else None
        if page_id and record is None:
            return _error("No page with that id.", "page_not_found", session_id)
        if raw_page is None:
            assert record is not None
            return PageResponse(
                message=f"The stored design, at revision {record.revision}.",
                session_id=session_id,
                design=stored_design(record).model_dump(mode="json", exclude_none=True),
            ).from_outside()
        try:
            design = dated(PageDesign.model_validate(raw_page))
        except ValidationError as e:
            return _error(_describe(e), "invalid_page", session_id)
        pin = str(kwargs.get("pin_to_workflow") or "") or None
        sources, problems = await check(user_id, design, pin_to=pin)
        if problems:
            return _error(_problems(problems), "invalid_page", session_id)
        if kwargs.get("save") is True:
            return await _save(
                user_id, session, design, record, pin, kwargs.get("based_on")
            )
        page = await resolve_page(
            user_id,
            design,
            sources,
            page_id=page_id,
            revision=record.revision if record else None,
            chat_session_id=session_id,
        )
        based_on = f" with based_on {record.revision}" if record else ""
        return _page_response(
            f"Preview shown to the user. Save it{based_on} only once they apply it.",
            session_id,
            page,
            saved=False,
        )


async def _save(
    user_id: str,
    session: ChatSession,
    design: PageDesign,
    record: Any,
    pin: str | None,
    based_on: Any,
) -> ToolResponseBase:
    session_id = session.session_id
    stored = design.model_dump(mode="json", exclude_none=True)
    if record is None:
        made = [
            p
            for p in await pages_db().list_pages(user_id)
            if p.chat_session_id == session_id
        ]
        if made:
            # An older preview applied again would make a second page.
            return _error(
                f"This chat already made page {made[0].id}. To change it, pass that "
                "page_id and its revision as based_on; for another page, start a new chat.",
                "stale_page",
                session_id,
            )
        try:
            saved = await pages_db().create_page(
                user_id, design.title, stored, chat_session_id=session_id
            )
        except ValueError as e:
            return _error(str(e), "page_limit", session_id)
    elif not isinstance(based_on, int):
        return _error(
            "Pass based_on: the revision your preview showed.",
            "invalid_page",
            session_id,
        )
    else:
        try:
            saved = await pages_db().save_page_design(
                user_id, record.id, design.title, stored, based_on=based_on
            )
        except ValueError as e:
            return _error(
                f"{e}. Read the page again and preview it.", "stale_page", session_id
            )
    if pin:
        saved = await pages_db().set_page_pin(user_id, saved.id, pin)
    page = await resolve_record(user_id, saved)
    where = " It also opens with the workflow in the library." if pin else ""
    return _page_response(
        f"Saved at revision {saved.revision}. The page is in Pages.{where}",
        session_id,
        page,
        saved=True,
    )


def catalogue() -> dict[str, Any]:
    return {
        "limits": {
            "sections": MAX_SECTIONS,
            "workflows_per_page": 6,
            "experts_per_page": 4,
        },
        "notes": [
            "A section's 'workflow' is a library workflow id; its 'expert' an expert id.",
            "Bind fields and cards only to that workflow's own input and output names.",
            "A row button or a from_output field names a workflow that has its own "
            "section on the page.",
            "'cards' holds fixed content you write; everything else is read live.",
            "collection_table and collection_numbers show one of the user's "
            "collections (the collections tool lists them): source "
            "{type: 'collection', id}, naming its fields exactly.",
        ],
        "page": compact_schema(PageDesign),
    }


def _page_response(
    message: str, session_id: str, page: ResolvedPage, *, saved: bool
) -> ToolResponseBase:
    page = _clipped(page)
    return PageResponse(
        message=message, session_id=session_id, page=page, saved=saved
    ).from_outside(*_outside_parts(page))


def _outside_parts(page: ResolvedPage) -> list[Any]:
    """Every value on *page* read from a run, a review or memory, not written by Otto."""
    parts: list[Any] = []
    for s in page.sections:
        if isinstance(s, PageLatestResult):
            parts += [s.blocks, s.other_outputs, [r.values for r in s.row_runs]]
        elif isinstance(s, PageRunHistory):
            parts.append([row.cells for row in s.rows])
        elif isinstance(s, PageRecentRuns):
            parts.append([run.summary for run in s.runs])
        elif isinstance(s, PageExpertWork):
            parts.append([item.summary for item in s.items])
        elif isinstance(s, PageExpertLearned):
            parts.append([fact.fact for fact in s.facts])
        elif isinstance(s, PageWaiting):
            parts.append([review.payload for review in s.reviews])
        elif isinstance(s, PageCollectionTable):
            parts.append([[row.values, row.pending] for row in s.rows])
        elif isinstance(s, PageCollectionNumbers):
            parts.append(s.block)
    return parts


def _clipped(page: ResolvedPage) -> ResolvedPage:
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


def _problems(problems: list[Any]) -> str:
    shown = "; ".join(
        f"section {p.section}: {p.message}" if p.section >= 0 else p.message
        for p in problems[:_MAX_REPORTED_PROBLEMS]
    )
    return f"Invalid page. {shown}"


def _describe(error: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors(include_input=False, include_url=False)
    ]
    return f"Invalid page. {'; '.join(problems[:_MAX_REPORTED_PROBLEMS])}"


async def pages_disabled_tools(session: ChatSession) -> frozenset[str]:
    """Hide the tool where its preview card can't be seen, and without the flag."""
    hidden = frozenset({DESIGN_PAGE_TOOL})
    metadata = session.metadata
    if metadata.source_platform or metadata.text_only_result or not session.user_id:
        return hidden
    if metadata.delegated_by_session_id and not metadata.handed_off_from_expert_id:
        return hidden
    enabled = await is_feature_enabled(Flag.PAGES, session.user_id)
    return frozenset() if enabled else hidden
