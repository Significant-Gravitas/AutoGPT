"""A workflow page as the frontend draws it: the design resolved against the
workflow's inputs, its latest run, its recent runs and its schedule."""

import json
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from backend.api.features.library.model import LibraryAgent
from backend.copilot.tools.render_ui_models import (
    ChartBlock,
    ChartSeries,
    StatItem,
    StatsBlock,
    TableBlock,
)
from backend.data.execution import ExecutionStatus, GraphExecution, GraphExecutionMeta

from .design import (
    Card,
    ChartCard,
    LatestResultSection,
    RecentRunsSection,
    RunFormSection,
    ScheduleSection,
    StatsCard,
    TableCard,
    TextSection,
    WorkflowPageDesign,
    check_design,
)

DEFAULT_BUTTON_LABEL = "Run"
OTHER_INPUTS_GROUP = "More options"
_MAX_TABLE_ROWS = 20
_MAX_CHART_POINTS = 50

ResultBlock = Annotated[
    StatsBlock | TableBlock | ChartBlock, Field(discriminator="type")
]


class PageField(BaseModel):
    input: str
    label: str
    help: str | None = None
    default: Any = None
    group: str | None = None
    input_schema: dict[str, Any]
    required: bool = False


class PageText(BaseModel):
    kind: Literal["text"]
    index: int
    heading: str | None = None
    body: str


class PageRunForm(BaseModel):
    kind: Literal["run_form"]
    index: int
    fields: list[PageField]
    button_label: str


class PageLatestResult(BaseModel):
    kind: Literal["latest_result"]
    index: int
    run_id: str | None = None
    run_status: ExecutionStatus | None = None
    finished_at: datetime | None = None
    blocks: list[ResultBlock] = Field(default_factory=list)
    other_outputs: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Outputs no card shows, drawn as the run view does.",
    )


class PageRun(BaseModel):
    id: str
    status: ExecutionStatus
    started_at: datetime | None = None
    ended_at: datetime | None = None
    cost_cents: int = 0


class PageRecentRuns(BaseModel):
    kind: Literal["recent_runs"]
    index: int
    runs: list[PageRun]


class PageSchedule(BaseModel):
    kind: Literal["schedule"]
    index: int
    is_scheduled: bool
    next_run: str | None = None
    unavailable: bool = False


PageSection = Annotated[
    PageText | PageRunForm | PageLatestResult | PageRecentRuns | PageSchedule,
    Field(discriminator="kind"),
]


class LeftOutSection(BaseModel):
    index: int
    kind: str
    problem: str


class ResolvedWorkflowPage(BaseModel):
    library_agent_id: str
    graph_id: str
    title: str
    sections: list[PageSection]
    left_out: list[LeftOutSection] = Field(
        default_factory=list, description="Sections that no longer fit the workflow."
    )
    has_previous: bool = False


def resolve_page(
    design: WorkflowPageDesign,
    agent: LibraryAgent,
    result_run: GraphExecution | None,
    recent_runs: list[GraphExecutionMeta],
    *,
    has_previous: bool = False,
) -> ResolvedWorkflowPage:
    """The page to draw; a section the workflow no longer fits is left out
    and named, never drawn half right."""
    problems = check_design(design, agent.input_schema, agent.output_schema)
    first_problem = {p.section: p.message for p in reversed(problems)}
    sections: list[PageSection] = []
    left_out: list[LeftOutSection] = []
    for index, section in enumerate(design.sections):
        if index in first_problem:
            left_out.append(
                LeftOutSection(
                    index=index, kind=section.kind, problem=first_problem[index]
                )
            )
            continue
        if isinstance(section, TextSection):
            sections.append(
                PageText(
                    kind="text", index=index, heading=section.heading, body=section.body
                )
            )
        elif isinstance(section, RunFormSection):
            sections.append(_run_form(index, section, agent.input_schema))
        elif isinstance(section, LatestResultSection):
            sections.append(resolve_result(index, section, result_run))
        elif isinstance(section, RecentRunsSection):
            runs = [_run(run) for run in recent_runs[: section.count]]
            sections.append(PageRecentRuns(kind="recent_runs", index=index, runs=runs))
        elif isinstance(section, ScheduleSection):
            sections.append(_schedule(index, agent))
    return ResolvedWorkflowPage(
        library_agent_id=agent.id,
        graph_id=agent.graph_id,
        title=design.title,
        sections=sections,
        left_out=left_out,
        has_previous=has_previous,
    )


def resolve_result(
    index: int,
    section: LatestResultSection,
    run: GraphExecution | None,
) -> PageLatestResult:
    """Each card fed from *run*'s outputs; a card whose value does not fit is
    dropped, so its output shows with the rest rather than not at all."""
    if run is None:
        return PageLatestResult(kind="latest_result", index=index)
    outputs = run.outputs
    latest = {name: values[-1] for name, values in outputs.items() if values}
    blocks: list[StatsBlock | TableBlock | ChartBlock] = []
    shown: set[str] = set()
    for card in section.cards:
        block = _card_block(card, latest)
        if block is not None:
            blocks.append(block)
            shown |= _bound_outputs(card)
    return PageLatestResult(
        kind="latest_result",
        index=index,
        run_id=run.id,
        run_status=run.status,
        finished_at=run.ended_at,
        blocks=blocks,
        other_outputs={k: v for k, v in outputs.items() if k not in shown},
    )


def _run_form(
    index: int, section: RunFormSection, input_schema: dict[str, Any]
) -> PageRunForm:
    inputs: dict[str, Any] = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])
    fields = [
        PageField(
            input=f.input,
            label=f.label or inputs[f.input].get("title") or f.input,
            help=f.help,
            default=(
                f.default if f.default is not None else inputs[f.input].get("default")
            ),
            group=f.group,
            input_schema=inputs[f.input],
            required=f.input in required,
        )
        for f in section.fields
    ]
    # An optional input the design leaves out still has to be reachable.
    named = {f.input for f in section.fields}
    fields += [
        PageField(
            input=name,
            label=schema.get("title") or name,
            default=schema.get("default"),
            group=OTHER_INPUTS_GROUP,
            input_schema=schema,
        )
        for name, schema in inputs.items()
        if name not in named
    ]
    return PageRunForm(
        kind="run_form",
        index=index,
        fields=fields,
        button_label=section.button_label or DEFAULT_BUTTON_LABEL,
    )


def _run(run: GraphExecutionMeta) -> PageRun:
    return PageRun(
        id=run.id,
        status=run.status,
        started_at=run.started_at,
        ended_at=run.ended_at,
        cost_cents=run.stats.cost if run.stats else 0,
    )


def _schedule(index: int, agent: LibraryAgent) -> PageSchedule:
    return PageSchedule(
        kind="schedule",
        index=index,
        is_scheduled=agent.is_scheduled,
        next_run=agent.next_scheduled_run,
        unavailable=agent.schedule_info_unavailable,
    )


def _bound_outputs(card: Card) -> set[str]:
    if isinstance(card, StatsCard):
        return {item.output for item in card.items}
    return {card.output}


def _card_block(
    card: Card, latest: dict[str, Any]
) -> StatsBlock | TableBlock | ChartBlock | None:
    try:
        if isinstance(card, StatsCard):
            return _stats(card, latest)
        if isinstance(card, TableCard):
            return _table(card, latest)
        return _chart(card, latest)
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


def _table(card: TableCard, latest: dict[str, Any]) -> TableBlock:
    rows = _as_list(latest[card.output])
    shown = rows[:_MAX_TABLE_ROWS]
    cells = [
        [_clip(_format(_cell(row, column.key)), 200) for column in card.columns]
        for row in shown
    ]
    return TableBlock(
        type="table",
        title=_with_count(card.title, len(shown), len(rows), "rows"),
        columns=[column.label for column in card.columns],
        rows=cells,
    )


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
