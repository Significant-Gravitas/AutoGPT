"""Reading, previewing, saving and editing pages, for the routes and Otto's tool.

Data is read through ``db_accessors``, so the same code runs in the API, where
the database is local, and in the copilot executor, where it is an RPC.
"""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ValidationError

from backend.data.db_accessors import execution_db, pages_db
from backend.data.execution import GraphExecution
from backend.util.exceptions import NotFoundError

from .db import PageRecord
from .design import CardsSection, DesignProblem, PageDesign, check_design
from .resolve import resolve_page
from .resolved import ResolvedPage
from .sources import PageSources, load_sources

_MAX_FOLLOWED = 6


class PageListItem(BaseModel):
    id: str
    title: str
    revision: int
    workflow_count: int
    expert_count: int
    pinned_library_agent_id: str | None = None
    chat_session_id: str | None = None
    updated_at: datetime


async def list_pages(user_id: str) -> list[PageListItem]:
    records = await pages_db().list_pages(user_id)
    return [_list_item(record) for record in records]


async def read_page(
    user_id: str, page_id: str, *, run_ids: Sequence[str] = ()
) -> ResolvedPage:
    record = await pages_db().get_page(user_id, page_id)
    if record is None:
        raise NotFoundError(f"Page #{page_id} not found")
    return await resolve_record(user_id, record, run_ids=run_ids)


async def read_pinned_page(
    user_id: str, library_agent_id: str, *, run_ids: Sequence[str] = ()
) -> ResolvedPage:
    record = await pages_db().get_pinned_page(user_id, library_agent_id)
    if record is None:
        raise NotFoundError(f"No page is pinned to library agent #{library_agent_id}")
    return await resolve_record(user_id, record, run_ids=run_ids)


async def resolve_record(
    user_id: str, record: PageRecord, *, run_ids: Sequence[str] = ()
) -> ResolvedPage:
    design = stored_design(record)
    sources = await load_sources(user_id, design)
    return await resolve_page(
        user_id,
        design,
        sources,
        followed=await _followed_runs(user_id, run_ids, sources),
        page_id=record.id,
        revision=record.revision,
        pinned_library_agent_id=record.pinned_library_agent_id,
        chat_session_id=record.chat_session_id,
        has_previous=record.previous_design is not None,
    )


def stored_design(record: PageRecord) -> PageDesign:
    try:
        return PageDesign.model_validate(record.design)
    except ValidationError as e:
        raise NotFoundError("This page's design can no longer be read") from e


async def check(
    user_id: str, design: PageDesign, *, pin_to: str | None = None
) -> tuple[PageSources, list[DesignProblem]]:
    """The design's sources for *user_id*, and every way it fails them; a
    source that could not be read is a problem here, since a save trusts it."""
    sources = await load_sources(
        user_id, design, extra_workflows=[pin_to] if pin_to else []
    )
    problems = check_design(design, sources)
    if (
        pin_to
        and pin_to not in sources.failed
        and sources.workflows.get(pin_to) is None
    ):
        problems.append(
            DesignProblem(
                section=-1, message="pin_to_workflow is not one of your workflows"
            )
        )
    if sources.failed:
        problems.append(
            DesignProblem(
                section=-1, message="one of its sources could not be read just now"
            )
        )
    return sources, problems


def dated(design: PageDesign) -> PageDesign:
    """*design* with each new cards section marked with today's date."""
    today = datetime.now(UTC).date()
    sections = [
        (
            s.model_copy(update={"written_on": today})
            if isinstance(s, CardsSection) and s.written_on is None
            else s
        )
        for s in design.sections
    ]
    return design.model_copy(update={"sections": sections})


async def remove_section(user_id: str, page_id: str, index: int) -> ResolvedPage:
    record, design = await _stored(user_id, page_id)
    if not 0 <= index < len(design.sections):
        raise NotFoundError(f"This page has no section {index}")
    if len(design.sections) == 1:
        raise ValueError("A page needs one section; delete the page instead")
    kept = [s for i, s in enumerate(design.sections) if i != index]
    return await _save_edit(
        user_id, record, design.model_copy(update={"sections": kept})
    )


async def edit_page(
    user_id: str,
    page_id: str,
    *,
    title: str | None = None,
    order: list[int] | None = None,
) -> ResolvedPage:
    record, design = await _stored(user_id, page_id)
    if order is not None:
        if sorted(order) != list(range(len(design.sections))):
            raise ValueError("The new order must list every section once")
        design = design.model_copy(
            update={"sections": [design.sections[i] for i in order]}
        )
    if title is not None:
        design = PageDesign.model_validate({**design.model_dump(), "title": title})
    return await _save_edit(user_id, record, design)


async def _stored(user_id: str, page_id: str) -> tuple[PageRecord, PageDesign]:
    record = await pages_db().get_page(user_id, page_id)
    if record is None:
        raise NotFoundError(f"Page #{page_id} not found")
    return record, stored_design(record)


async def _save_edit(
    user_id: str, record: PageRecord, design: PageDesign
) -> ResolvedPage:
    saved = await pages_db().save_page_design(
        user_id,
        record.id,
        design.title,
        design.model_dump(mode="json", exclude_none=True),
        based_on=record.revision,
    )
    return await resolve_record(user_id, saved)


async def _followed_runs(
    user_id: str, run_ids: Sequence[str], sources: PageSources
) -> list[GraphExecution]:
    """The runs a page follows, kept only where they belong to one of its workflows."""
    graph_ids = {agent.graph_id for agent in sources.workflows.values() if agent}
    runs = await asyncio.gather(
        *(
            execution_db().get_graph_execution(user_id=user_id, execution_id=run_id)
            for run_id in list(dict.fromkeys(run_ids))[:_MAX_FOLLOWED]
        )
    )
    return [run for run in runs if run is not None and run.graph_id in graph_ids]


def _list_item(record: PageRecord) -> PageListItem:
    try:
        design = PageDesign.model_validate(record.design)
        workflows, experts = len(design.workflow_ids()), len(design.expert_ids())
    except ValidationError:
        workflows = experts = 0
    return PageListItem(
        id=record.id,
        title=record.title,
        revision=record.revision,
        workflow_count=workflows,
        expert_count=experts,
        pinned_library_agent_id=record.pinned_library_agent_id,
        chat_session_id=record.chat_session_id,
        updated_at=record.updated_at,
    )
