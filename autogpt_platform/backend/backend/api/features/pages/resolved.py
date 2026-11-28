"""A page as the frontend draws it: each section resolved for its owner."""

from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.copilot.tools.render_ui_models import (
    ChartBlock,
    GalleryBlock,
    StatsBlock,
    TableBlock,
    TimelineBlock,
)
from backend.data.collections.model import (
    CollectionRowRecord,
    Condition,
    FieldType,
    Sort,
)
from backend.data.execution import ExecutionStatus

from .design import StoredCard

ResultBlock = Annotated[
    StatsBlock | TableBlock | ChartBlock | GalleryBlock | TimelineBlock,
    Field(discriminator="type"),
]


class PageField(BaseModel):
    input: str
    label: str
    help: str | None = None
    default: Any = None
    group: str | None = None
    input_schema: dict[str, Any]
    required: bool = False


class RowRun(BaseModel):
    """A table's per-row button: run another workflow with each row's value."""

    block_index: int
    label: str
    library_agent_id: str
    graph_id: str
    graph_version: int
    input: str
    values: list[str]


class PageRun(BaseModel):
    id: str
    status: ExecutionStatus
    started_at: datetime | None = None
    ended_at: datetime | None = None
    cost_cents: int = 0
    summary: str | None = None


class PageTemplate(BaseModel):
    id: str
    name: str


class HistoryRow(BaseModel):
    run_id: str
    status: ExecutionStatus
    started_at: datetime | None = None
    cells: list[str | None]


class WorkItem(BaseModel):
    execution_id: str
    workflow_name: str
    status: str
    started_at: datetime | None = None
    summary: str | None = None
    link: str | None = None


class LearnedFact(BaseModel):
    fact: str
    created_at: str | None = None


class _Sourced(BaseModel):
    index: int
    library_agent_id: str
    workflow_name: str


class _ExpertSourced(BaseModel):
    index: int
    expert_id: str
    expert_name: str


class PageText(BaseModel):
    kind: Literal["text"]
    index: int
    heading: str | None = None
    body: str
    tone: Literal["takeaway", "warning"] | None = None


class PageCards(BaseModel):
    kind: Literal["cards"]
    index: int
    blocks: list[StoredCard]
    written_on: date | None = None


class PageRunForm(_Sourced):
    kind: Literal["run_form"]
    fields: list[PageField]
    button_label: str


class PageQuickRuns(_Sourced):
    kind: Literal["quick_runs"]
    templates: list[PageTemplate]


class PageLatestResult(_Sourced):
    kind: Literal["latest_result"]
    run_id: str | None = None
    run_status: ExecutionStatus | None = None
    finished_at: datetime | None = None
    blocks: list[ResultBlock] = Field(default_factory=list)
    row_runs: list[RowRun] = Field(default_factory=list)
    other_outputs: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Outputs no card shows, drawn as the run view does.",
    )


class PageRecentRuns(_Sourced):
    kind: Literal["recent_runs"]
    runs: list[PageRun]


class PageRunHistory(_Sourced):
    kind: Literal["run_history"]
    columns: list[str]
    rows: list[HistoryRow]


class PageRunNumbers(_Sourced):
    kind: Literal["run_numbers"]
    period: Literal["today", "7d", "30d"]
    block: StatsBlock


class PageSchedule(_Sourced):
    kind: Literal["schedule"]
    is_scheduled: bool
    next_run: str | None = None
    unavailable: bool = False


class PageRunTrend(_Sourced):
    kind: Literal["run_trend"]
    block: ChartBlock | None = Field(None, description="None until there are two runs.")


class PageExpertStatus(_ExpertSourced):
    kind: Literal["expert_status"]
    status: Literal["ready", "working", "needs_setup", "paused", "failed"]
    detail: str
    spend_cents: int = 0


class PageExpertWork(_ExpertSourced):
    kind: Literal["expert_work"]
    items: list[WorkItem]


class PageExpertLearned(_ExpertSourced):
    kind: Literal["expert_learned"]
    facts: list[LearnedFact]


class PageExpertActivity(_ExpertSourced):
    kind: Literal["expert_activity"]
    block: ChartBlock | None = None


class PageWaiting(BaseModel):
    kind: Literal["waiting"]
    index: int
    scope_name: str | None = Field(None, description="The workflow or expert, if one.")
    reviews: list[PendingHumanReviewModel]


class _CollectionSourced(BaseModel):
    index: int
    collection_id: str
    collection_name: str


class PageTableColumn(BaseModel):
    field: str
    type: FieldType
    options: list[str] = Field(default_factory=list)
    unit: str | None = None


class PageCollectionTable(_CollectionSourced):
    kind: Literal["collection_table"]
    title: str | None = None
    columns: list[PageTableColumn]
    dropped_columns: list[str] = Field(
        default_factory=list, description="Columns the collection no longer has."
    )
    rows: list[CollectionRowRecord]
    total: int
    filter: list[Condition] = Field(default_factory=list)
    sort: Sort | None = None
    read_only: bool
    key_field: str | None = None


class PageCollectionNumbers(_CollectionSourced):
    kind: Literal["collection_numbers"]
    block: StatsBlock


class PageNotice(BaseModel):
    """A section left out: its source no longer fits, or it could not be read."""

    kind: Literal["notice"]
    index: int
    section_kind: str
    reason: Literal["stale", "failed"]
    message: str


PageSection = Annotated[
    PageText
    | PageCards
    | PageRunForm
    | PageQuickRuns
    | PageLatestResult
    | PageRecentRuns
    | PageRunHistory
    | PageRunNumbers
    | PageSchedule
    | PageRunTrend
    | PageExpertStatus
    | PageExpertWork
    | PageExpertLearned
    | PageExpertActivity
    | PageWaiting
    | PageCollectionTable
    | PageCollectionNumbers
    | PageNotice,
    Field(discriminator="kind"),
]


class ResolvedPage(BaseModel):
    id: str | None = Field(None, description="None for a preview not yet saved.")
    title: str
    revision: int | None = None
    pinned_library_agent_id: str | None = None
    chat_session_id: str | None = None
    has_previous: bool = False
    sections: list[PageSection]
    running: bool = Field(
        False,
        description="A run the page shows or follows is still going; re-read it soon.",
    )
