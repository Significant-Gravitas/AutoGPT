"""Resolve a page design for its owner: every section reads its own source,
concurrently, and one that fails comes back as a notice, never as empty."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import ValidationError

from backend.api.features.experts.models import Expert
from backend.api.features.home.agents import compose_agent_statuses
from backend.api.features.library.model import LibraryAgent
from backend.api.features.memory.routes import read_live_facts
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.tools.render_ui_models import (
    ChartBlock,
    ChartSeries,
    GalleryBlock,
    GalleryImage,
    StatItem,
    StatsBlock,
    TableBlock,
    TimelineBlock,
    TimelineItem,
)
from backend.data.db_accessors import execution_db, experts_db, library_db, review_db
from backend.data.execution import ExecutionStatus, GraphExecution, GraphExecutionMeta

from .collection_design import (
    CollectionNumbersSection,
    CollectionTableSection,
    section_collection,
)
from .collection_resolve import resolve_numbers, resolve_table
from .design import (
    UNAVAILABLE,
    Card,
    CardsSection,
    ChartCard,
    ExpertActivitySection,
    ExpertLearnedSection,
    ExpertStatusSection,
    ExpertWorkSection,
    GalleryCard,
    LatestResultSection,
    PageDesign,
    QuickRunsSection,
    RecentRunsSection,
    RunFormSection,
    RunHistorySection,
    RunNumbersSection,
    RunTrendSection,
    ScheduleSection,
    StatsCard,
    TableCard,
    TextSection,
    TimelineCard,
    WaitingSection,
    check_design,
    section_expert,
    section_workflow,
)
from .resolved import (
    HistoryRow,
    LearnedFact,
    PageCards,
    PageExpertActivity,
    PageExpertLearned,
    PageExpertStatus,
    PageExpertWork,
    PageField,
    PageLatestResult,
    PageNotice,
    PageQuickRuns,
    PageRecentRuns,
    PageRun,
    PageRunForm,
    PageRunHistory,
    PageRunNumbers,
    PageRunTrend,
    PageSchedule,
    PageSection,
    PageTemplate,
    PageText,
    PageWaiting,
    ResolvedPage,
    RowRun,
    WorkItem,
)
from .row_sources import CollectionRowSource
from .sources import PageSources

logger = logging.getLogger(__name__)

DEFAULT_BUTTON_LABEL = "Run"
OTHER_INPUTS_GROUP = "More options"
COULD_NOT_LOAD = "This couldn't be loaded just now."
STALE = "This section no longer fits its {source}. Ask Otto to fix it."
GONE = "The {source} this section shows is no longer available."
_RECENT_FETCHED = 30
_ACTIVITY_DAYS = 30
_MAX_TABLE_ROWS = 20
_MAX_CHART_POINTS = 50
_MAX_ROW_VALUE = 500
_PERIODS = {"today": None, "7d": 7, "30d": 30}


async def resolve_page(
    user_id: str,
    design: PageDesign,
    sources: PageSources,
    *,
    followed: Sequence[GraphExecution] = (),
    page_id: str | None = None,
    revision: int | None = None,
    pinned_library_agent_id: str | None = None,
    chat_session_id: str | None = None,
    has_previous: bool = False,
) -> ResolvedPage:
    stale = {p.section: p.message for p in reversed(check_design(design, sources))}
    reader = _Reader(user_id, followed)
    sections = await asyncio.gather(
        *(
            _guarded(index, section, sources, stale, reader)
            for index, section in enumerate(design.sections)
        )
    )
    return ResolvedPage(
        id=page_id,
        title=design.title,
        revision=revision,
        pinned_library_agent_id=pinned_library_agent_id,
        chat_session_id=chat_session_id,
        has_previous=has_previous,
        sections=list(sections),
    )


async def _guarded(
    index: int,
    section: Any,
    sources: PageSources,
    stale: dict[int, str],
    reader: "_Reader",
) -> PageSection:
    workflow, expert = section_workflow(section), section_expert(section)
    collection = section_collection(section)
    if any(ref and ref in sources.failed for ref in (workflow, expert, collection)):
        return _notice(index, section, "failed", COULD_NOT_LOAD)
    if index in stale:
        source = (
            "collection"
            if collection
            else "expert" if expert and not workflow else "workflow"
        )
        message = GONE if UNAVAILABLE in stale[index] else STALE
        return _notice(index, section, "stale", message.format(source=source))
    try:
        return await _resolve(index, section, sources, reader)
    except Exception:
        logger.warning(
            f"Page section {index} ({section.kind}) could not be read", exc_info=True
        )
        return _notice(index, section, "failed", COULD_NOT_LOAD)


async def _resolve(
    index: int, section: Any, sources: PageSources, reader: "_Reader"
) -> PageSection:
    if isinstance(section, TextSection):
        return PageText(
            kind="text",
            index=index,
            heading=section.heading,
            body=section.body,
            tone=section.tone,
        )
    if isinstance(section, CardsSection):
        return PageCards(
            kind="cards",
            index=index,
            blocks=section.blocks,
            written_on=section.written_on,
        )
    if isinstance(section, WaitingSection):
        return await _waiting(index, section, sources, reader)
    if collection := section_collection(section):
        info = sources.collections[collection]
        assert info is not None
        if isinstance(section, CollectionTableSection):
            return await resolve_table(
                index, section, CollectionRowSource(info), reader.user_id
            )
        assert isinstance(section, CollectionNumbersSection)
        return await resolve_numbers(
            index, section, CollectionRowSource(info), reader.user_id
        )
    workflow, expert = section_workflow(section), section_expert(section)
    if workflow:
        agent = sources.workflows[workflow]
        assert agent is not None
        return await _WORKFLOW_KINDS[type(section)](
            index, section, agent, sources, reader
        )
    assert expert
    found = sources.experts[expert]
    assert found is not None
    return await _EXPERT_KINDS[type(section)](index, section, found, reader)


class _Reader:
    """Reads shared by a page's sections, each fetched once per page read."""

    def __init__(self, user_id: str, followed: Sequence[GraphExecution]):
        self.user_id = user_id
        self._followed = {run.graph_id: run for run in followed}
        self._tasks: dict[tuple[str, str], asyncio.Future[Any]] = {}

    def recent(self, graph_id: str) -> Awaitable[list[GraphExecutionMeta]]:
        return self._once(("recent", graph_id), lambda: self._recent(graph_id))

    def latest(self, graph_id: str) -> Awaitable[GraphExecution | None]:
        if graph_id in self._followed:
            return _done(self._followed[graph_id])
        return self._once(("latest", graph_id), lambda: self._latest(graph_id))

    def full(self, run_id: str) -> Awaitable[GraphExecution | None]:
        return self._once(
            ("full", run_id),
            lambda: execution_db().get_graph_execution(
                user_id=self.user_id, execution_id=run_id
            ),
        )

    def _once(self, key: tuple[str, str], factory: Callable[[], Awaitable[Any]]):
        if key not in self._tasks:
            self._tasks[key] = asyncio.ensure_future(factory())
        return self._tasks[key]

    async def _recent(self, graph_id: str) -> list[GraphExecutionMeta]:
        runs = await execution_db().get_graph_executions(
            graph_id=graph_id, user_id=self.user_id, limit=_RECENT_FETCHED
        )
        return [run for run in runs if not run.is_dry_run]

    async def _latest(self, graph_id: str) -> GraphExecution | None:
        finished = await execution_db().get_graph_executions(
            graph_id=graph_id,
            user_id=self.user_id,
            statuses=[ExecutionStatus.COMPLETED],
            limit=5,
        )
        run_id = next((run.id for run in finished if not run.is_dry_run), None)
        return await self.full(run_id) if run_id else None


async def _run_form(
    index: int,
    section: RunFormSection,
    agent: LibraryAgent,
    sources: PageSources,
    reader: _Reader,
) -> PageRunForm:
    inputs: dict[str, Any] = agent.input_schema.get("properties") or {}
    required = set(agent.input_schema.get("required") or [])
    recent = await reader.recent(agent.graph_id)
    last: dict[str, Any] = dict(recent[0].inputs or {}) if recent else {}
    fields = []
    for f in section.fields:
        bound = (
            await _bound_value(f.from_output, sources, reader)
            if f.from_output
            else None
        )
        fields.append(
            PageField(
                input=f.input,
                label=f.label or inputs[f.input].get("title") or f.input,
                help=f.help,
                default=_first(
                    bound, last.get(f.input), f.default, inputs[f.input].get("default")
                ),
                group=f.group,
                input_schema=inputs[f.input],
                required=f.input in required,
            )
        )
    named = {f.input for f in section.fields}
    # An optional input the design leaves out still has to be reachable.
    fields += [
        PageField(
            input=name,
            label=schema.get("title") or name,
            default=_first(last.get(name), schema.get("default")),
            group=OTHER_INPUTS_GROUP,
            input_schema=schema,
        )
        for name, schema in inputs.items()
        if name not in named
    ]
    return PageRunForm(
        kind="run_form",
        index=index,
        library_agent_id=agent.id,
        workflow_name=agent.name,
        fields=fields,
        button_label=section.button_label or DEFAULT_BUTTON_LABEL,
    )


async def _quick_runs(
    index: int,
    section: QuickRunsSection,
    agent: LibraryAgent,
    _s: PageSources,
    reader: _Reader,
) -> PageQuickRuns:
    listed = await library_db().list_presets(
        reader.user_id, page=1, page_size=20, graph_id=agent.graph_id
    )
    templates = [
        PageTemplate(id=p.id, name=_clip(p.name, 60))
        for p in listed.presets
        if p.is_active and not p.webhook_id
    ][:4]
    return PageQuickRuns(
        kind="quick_runs",
        index=index,
        library_agent_id=agent.id,
        workflow_name=agent.name,
        templates=templates,
    )


async def _latest_result(
    index: int,
    section: LatestResultSection,
    agent: LibraryAgent,
    sources: PageSources,
    reader: _Reader,
) -> PageLatestResult:
    run = await reader.latest(agent.graph_id)
    return resolve_result(index, section, agent, run, sources)


def resolve_result(
    index: int,
    section: LatestResultSection,
    agent: LibraryAgent,
    run: GraphExecution | None,
    sources: PageSources,
) -> PageLatestResult:
    """Each card fed from *run*'s outputs; a card whose value does not fit is
    dropped, so its output shows with the rest rather than not at all."""
    base = {
        "kind": "latest_result",
        "index": index,
        "library_agent_id": agent.id,
        "workflow_name": agent.name,
    }
    if run is None:
        return PageLatestResult(**base)
    latest = {name: values[-1] for name, values in run.outputs.items() if values}
    blocks: list[Any] = []
    row_runs: list[RowRun] = []
    shown: set[str] = set()
    for card in section.cards:
        built = _card_block(card, latest)
        if built is None:
            continue
        block, rows = built
        if isinstance(card, TableCard) and card.row_action:
            target = sources.workflows.get(card.row_action.workflow)
            if target is not None:
                row_runs.append(_row_run(len(blocks), card, target, rows))
        blocks.append(block)
        shown |= _bound_outputs(card)
    return PageLatestResult(
        **base,
        run_id=run.id,
        run_status=run.status,
        finished_at=run.ended_at,
        blocks=blocks,
        row_runs=row_runs,
        other_outputs={k: v for k, v in run.outputs.items() if k not in shown},
    )


async def _recent_runs(
    index: int,
    section: RecentRunsSection,
    agent: LibraryAgent,
    _s: PageSources,
    reader: _Reader,
) -> PageRecentRuns:
    runs = (await reader.recent(agent.graph_id))[: section.count]
    return PageRecentRuns(
        kind="recent_runs",
        index=index,
        library_agent_id=agent.id,
        workflow_name=agent.name,
        runs=[_page_run(run) for run in runs],
    )


async def _run_history(
    index: int,
    section: RunHistorySection,
    agent: LibraryAgent,
    _s: PageSources,
    reader: _Reader,
) -> PageRunHistory:
    runs = (await reader.recent(agent.graph_id))[: section.count]
    fulls = await asyncio.gather(*(reader.full(run.id) for run in runs))
    rows = [
        HistoryRow(
            run_id=run.id,
            status=run.status,
            started_at=run.started_at,
            cells=[_history_cell(full, column.output) for column in section.columns],
        )
        for run, full in zip(runs, fulls)
    ]
    return PageRunHistory(
        kind="run_history",
        index=index,
        library_agent_id=agent.id,
        workflow_name=agent.name,
        columns=[column.label for column in section.columns],
        rows=rows,
    )


async def _run_numbers(
    index: int,
    section: RunNumbersSection,
    agent: LibraryAgent,
    _s: PageSources,
    reader: _Reader,
) -> PageRunNumbers:
    now = datetime.now(UTC)
    days = _PERIODS[section.period]
    since = (
        now - timedelta(days=days)
        if days
        else now.replace(hour=0, minute=0, second=0, microsecond=0)
    )
    runs = await execution_db().get_graph_executions(
        graph_id=agent.graph_id,
        user_id=reader.user_id,
        created_time_gte=since,
        limit=1000,
    )
    runs = [run for run in runs if not run.is_dry_run]
    finished = [r for r in runs if r.status in _FINISHED]
    succeeded = sum(r.status == ExecutionStatus.COMPLETED for r in finished)
    spent = sum(r.stats.cost for r in runs if r.stats)
    rate = f"{round(100 * succeeded / len(finished))}%" if finished else "–"
    block = StatsBlock(
        type="stats",
        items=[
            StatItem(label="Runs", value=f"{len(runs):,}"),
            StatItem(label="Succeeded", value=rate),
            StatItem(label="Spent", value=f"${spent / 100:,.2f}"),
        ],
    )
    return PageRunNumbers(
        kind="run_numbers",
        index=index,
        library_agent_id=agent.id,
        workflow_name=agent.name,
        period=section.period,
        block=block,
    )


async def _schedule(
    index: int,
    _section: ScheduleSection,
    agent: LibraryAgent,
    _s: PageSources,
    _r: _Reader,
) -> PageSchedule:
    return PageSchedule(
        kind="schedule",
        index=index,
        library_agent_id=agent.id,
        workflow_name=agent.name,
        is_scheduled=agent.is_scheduled,
        next_run=agent.next_scheduled_run,
        unavailable=agent.schedule_info_unavailable,
    )


async def _run_trend(
    index: int,
    section: RunTrendSection,
    agent: LibraryAgent,
    _s: PageSources,
    reader: _Reader,
) -> PageRunTrend:
    runs = list(reversed((await reader.recent(agent.graph_id))[: section.runs]))
    if section.value == "output" and section.output:
        fulls = await asyncio.gather(*(reader.full(run.id) for run in runs))
        points = [
            (run, value)
            for run, full in zip(runs, fulls)
            if (value := _numeric_output(full, section.output)) is not None
        ]
        label, unit = section.output, None
    else:
        points = [(run, _run_measure(run, section.value)) for run in runs if run.stats]
        label = "Cost ($)" if section.value == "cost" else "Duration (s)"
        unit = None
    block = None
    if len(points) >= 2:
        block = ChartBlock(
            type="chart",
            kind="line",
            x_labels=[_when(run.started_at or run.ended_at) for run, _ in points],
            series=[ChartSeries(name=_clip(label, 60), values=[v for _, v in points])],
            unit=unit,
        )
    return PageRunTrend(
        kind="run_trend",
        index=index,
        library_agent_id=agent.id,
        workflow_name=agent.name,
        block=block,
    )


_WORKFLOW_KINDS: dict[type, Callable[..., Awaitable[PageSection]]] = {
    RunFormSection: _run_form,
    QuickRunsSection: _quick_runs,
    LatestResultSection: _latest_result,
    RecentRunsSection: _recent_runs,
    RunHistorySection: _run_history,
    RunNumbersSection: _run_numbers,
    ScheduleSection: _schedule,
    RunTrendSection: _run_trend,
}


async def _expert_status(
    index: int, _section: ExpertStatusSection, expert: Expert, reader: _Reader
) -> PageExpertStatus:
    running, week = await asyncio.gather(
        execution_db().get_graph_executions(
            user_id=reader.user_id,
            expert_id=expert.id,
            statuses=[ExecutionStatus.RUNNING, ExecutionStatus.QUEUED],
            limit=1,
        ),
        execution_db().get_graph_executions(
            user_id=reader.user_id,
            expert_id=expert.id,
            created_time_gte=datetime.now(UTC) - timedelta(days=7),
            limit=500,
        ),
    )
    (status,) = compose_agent_statuses(
        experts=[expert],
        running_expert_ids={expert.id} if running else set(),
        next_run_by_expert={},
        spend_by_expert={expert.id: sum(r.stats.cost for r in week if r.stats)},
    )
    return PageExpertStatus(
        kind="expert_status",
        index=index,
        expert_id=expert.id,
        expert_name=expert.name,
        status=status.status,
        detail=status.detail,
        spend_cents=status.spend_cents,
    )


async def _expert_work(
    index: int, section: ExpertWorkSection, expert: Expert, reader: _Reader
) -> PageExpertWork:
    runs = await experts_db().list_expert_runs(reader.user_id, expert.id, section.count)
    # An empty id list would leave the query unfiltered.
    metas = (
        await execution_db().get_graph_executions(
            execution_ids=[run.execution_id for run in runs], user_id=reader.user_id
        )
        if runs
        else []
    )
    summaries = {m.id: m.stats.activity_status for m in metas if m.stats}
    items = [
        WorkItem(
            execution_id=run.execution_id,
            workflow_name=run.agent_name,
            status=run.status,
            started_at=run.started_at,
            summary=summaries.get(run.execution_id),
            link=run.link,
        )
        for run in runs
    ]
    return PageExpertWork(
        kind="expert_work",
        index=index,
        expert_id=expert.id,
        expert_name=expert.name,
        items=items,
    )


async def _expert_learned(
    index: int, section: ExpertLearnedSection, expert: Expert, reader: _Reader
) -> PageExpertLearned:
    facts = await read_live_facts(
        MemoryScope.for_expert(reader.user_id, expert.id), section.count
    )
    return PageExpertLearned(
        kind="expert_learned",
        index=index,
        expert_id=expert.id,
        expert_name=expert.name,
        facts=[
            LearnedFact(fact=f.fact, created_at=f.created_at) for f in facts if f.fact
        ],
    )


async def _expert_activity(
    index: int, _section: ExpertActivitySection, expert: Expert, reader: _Reader
) -> PageExpertActivity:
    activity = await experts_db().get_expert_activity(reader.user_id, expert.id)
    days = activity.days[-_ACTIVITY_DAYS:]
    block = (
        ChartBlock(
            type="chart",
            kind="bar",
            x_labels=[day.day.strftime("%-d %b") for day in days],
            series=[
                ChartSeries(name="Chats", values=[float(day.sessions) for day in days]),
                ChartSeries(name="Runs", values=[float(day.runs) for day in days]),
            ],
        )
        if days
        else None
    )
    return PageExpertActivity(
        kind="expert_activity",
        index=index,
        expert_id=expert.id,
        expert_name=expert.name,
        block=block,
    )


_EXPERT_KINDS: dict[type, Callable[..., Awaitable[PageSection]]] = {
    ExpertStatusSection: _expert_status,
    ExpertWorkSection: _expert_work,
    ExpertLearnedSection: _expert_learned,
    ExpertActivitySection: _expert_activity,
}


async def _waiting(
    index: int, section: WaitingSection, sources: PageSources, reader: _Reader
) -> PageWaiting:
    reviews = await review_db().get_pending_reviews_for_user(reader.user_id, 1, 50)
    scope_name = None
    if section.workflow:
        agent = sources.workflows[section.workflow]
        assert agent is not None
        reviews = [r for r in reviews if r.graph_id == agent.graph_id]
        scope_name = agent.name
    elif section.expert:
        expert = sources.experts[section.expert]
        assert expert is not None
        paused = await execution_db().get_graph_executions(
            user_id=reader.user_id,
            expert_id=expert.id,
            statuses=[ExecutionStatus.REVIEW],
            limit=50,
        )
        ids = {run.id for run in paused}
        reviews = [r for r in reviews if r.graph_exec_id in ids]
        scope_name = expert.name
    return PageWaiting(
        kind="waiting", index=index, scope_name=scope_name, reviews=reviews
    )


_FINISHED = {
    ExecutionStatus.COMPLETED,
    ExecutionStatus.FAILED,
    ExecutionStatus.TERMINATED,
}


def _notice(
    index: int, section: Any, reason: Literal["stale", "failed"], message: str
) -> PageNotice:
    return PageNotice(
        kind="notice",
        index=index,
        section_kind=section.kind,
        reason=reason,
        message=message,
    )


async def _bound_value(ref: Any, sources: PageSources, reader: _Reader) -> str | None:
    target = sources.workflows.get(ref.workflow)
    if target is None:
        return None
    run = await reader.latest(target.graph_id)
    values = run.outputs.get(ref.output) if run else None
    if not values:
        return None
    value = values[-1]
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _row_run(
    block_index: int, card: TableCard, target: LibraryAgent, rows: list[Any]
) -> RowRun:
    action = card.row_action
    assert action is not None
    return RowRun(
        block_index=block_index,
        label=action.label,
        library_agent_id=target.id,
        graph_id=target.graph_id,
        graph_version=target.graph_version,
        input=action.input,
        values=[
            _clip(_format(_cell(row, action.column)), _MAX_ROW_VALUE) for row in rows
        ],
    )


def _page_run(run: GraphExecutionMeta) -> PageRun:
    return PageRun(
        id=run.id,
        status=run.status,
        started_at=run.started_at,
        ended_at=run.ended_at,
        cost_cents=run.stats.cost if run.stats else 0,
        summary=run.stats.activity_status if run.stats else None,
    )


def _history_cell(run: GraphExecution | None, output: str) -> str | None:
    values = run.outputs.get(output) if run else None
    if not values:
        return None
    value = values[-1]
    try:
        # A record reads as its fields, not as JSON.
        if isinstance(value, dict) and value:
            return _clip(
                " · ".join(f"{k}: {_format(v)}" for k, v in value.items()), 200
            )
        return _clip(_format(value), 200)
    except ValueError:
        return None


def _numeric_output(run: GraphExecution | None, output: str) -> float | None:
    values = run.outputs.get(output) if run else None
    if not values:
        return None
    try:
        return _number(values[-1])
    except (TypeError, ValueError):
        return None


def _run_measure(run: GraphExecutionMeta, value: str) -> float:
    assert run.stats is not None
    return (
        round(run.stats.cost / 100, 4)
        if value == "cost"
        else round(run.stats.duration, 1)
    )


def _bound_outputs(card: Card) -> set[str]:
    if isinstance(card, StatsCard):
        return {item.output for item in card.items}
    return {card.output}


def _card_block(card: Card, latest: dict[str, Any]) -> tuple[Any, list[Any]] | None:
    try:
        if isinstance(card, StatsCard):
            return _stats(card, latest), []
        if isinstance(card, TableCard):
            return _table(card, latest)
        if isinstance(card, ChartCard):
            return _chart(card, latest), []
        if isinstance(card, GalleryCard):
            return _gallery(card, latest), []
        return _timeline(card, latest), []
    except (KeyError, TypeError, ValueError, ValidationError):
        return None


def _stats(card: StatsCard, latest: dict[str, Any]) -> StatsBlock:
    items = []
    for item in card.items:
        value = latest[item.output]
        if isinstance(value, (dict, list)) or value is None:
            raise TypeError(f"{item.output} is not one value")
        items.append(StatItem(label=item.label, value=_clip(_format(value), 60)))
    return StatsBlock(type="stats", items=items)


def _table(card: TableCard, latest: dict[str, Any]) -> tuple[TableBlock, list[Any]]:
    rows = _as_list(latest[card.output])
    shown = rows[:_MAX_TABLE_ROWS]
    cells = [
        [_clip(_format(_cell(row, column.key)), 200) for column in card.columns]
        for row in shown
    ]
    block = TableBlock(
        type="table",
        title=_with_count(card.title, len(shown), len(rows), "rows"),
        columns=[column.label for column in card.columns],
        rows=cells,
    )
    return block, shown


def _chart(card: ChartCard, latest: dict[str, Any]) -> ChartBlock:
    points = _as_list(latest[card.output])
    shown = points[-_MAX_CHART_POINTS:]
    return ChartBlock(
        type="chart",
        title=_with_count(card.title, len(shown), len(points), "points"),
        kind=card.kind,
        x_labels=[_clip(_format(point[card.x_key]), 60) for point in shown],
        series=[
            ChartSeries(name=key, values=[_number(point[key]) for point in shown])
            for key in card.y_keys
        ],
        unit=card.unit,
    )


def _gallery(card: GalleryCard, latest: dict[str, Any]) -> GalleryBlock:
    images = []
    for position, item in enumerate(_as_list(latest[card.output])[:6], start=1):
        src = item.get("src") if isinstance(item, dict) else item
        caption = item.get("caption") if isinstance(item, dict) else None
        images.append(
            GalleryImage(
                src=str(src), caption=_clip(str(caption or f"Image {position}"), 120)
            )
        )
    return GalleryBlock(type="gallery", title=card.title, images=images)


def _timeline(card: TimelineCard, latest: dict[str, Any]) -> TimelineBlock:
    items = [
        TimelineItem(
            date=_clip(_format(row[card.date_key]), 60),
            title=_clip(_format(row[card.title_key]), 120),
            detail=(
                _clip(_format(row[card.detail_key]), 200)
                if card.detail_key and row.get(card.detail_key)
                else None
            ),
        )
        for row in _as_list(latest[card.output])[:10]
    ]
    return TimelineBlock(type="timeline", title=card.title, items=items)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise TypeError("not a list")
    return value


def _cell(row: Any, key: str | None) -> Any:
    if key is None:
        if isinstance(row, (dict, list)):
            raise TypeError("a plain-value column needs plain values")
        return row
    return row[key]


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("not a number")
    return float(value)


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    if not text:
        raise ValueError("empty value")
    return text


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _with_count(title: str | None, shown: int, total: int, noun: str) -> str | None:
    if shown == total:
        return title
    note = f"{shown} of {total} {noun}"
    return _clip(f"{title} · {note}" if title else note, 120)


def _when(moment: datetime | None) -> str:
    return moment.strftime("%-d %b %H:%M UTC") if moment else "–"


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None and v != ""), None)


def _done(value: Any) -> "asyncio.Future[Any]":
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future
