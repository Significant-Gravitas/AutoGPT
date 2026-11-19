from datetime import datetime

from pydantic import BaseModel, Field


class BriefingRunItem(BaseModel):
    expert_id: str | None
    expert_name: str | None
    expert_avatar_url: str | None
    agent_name: str
    graph_id: str
    execution_id: str
    library_agent_id: str | None
    status: str
    # Raw `stats.activity_status`, kept verbatim for the markdown renderer.
    summary: str | None
    link: str | None
    # Card fields the home dashboard reads off a stored briefing. All default,
    # so a stored row that predates them still validates — the home mapper
    # falls back to the agent-name headline when `title` is empty.
    expert_role: str | None = None
    title: str = ""
    detail: str = ""
    occurred_at: datetime | None = None
    duration_seconds: float = 0
    cost_cents: int = 0
    # What started the run when it was not a person: the scheduler job or
    # the webhook. Both None for a run started by hand and for rows stored
    # before the fields existed.
    schedule_id: str | None = None
    webhook_id: str | None = None
    # The execution's own `trigger_source`, set even when a schedule created
    # before its job carried a `schedule_id` fired the run.
    trigger_source: str | None = None


class BriefingDecisionItem(BaseModel):
    node_exec_id: str
    graph_exec_id: str | None
    title: str
    expert_id: str | None
    expert_name: str | None
    expert_avatar_url: str | None
    link: str


class BriefingTaskItem(BaseModel):
    task_id: str
    title: str
    # The task's status when the brief was written: DONE, FAILED, NEEDS_INPUT.
    status: str
    owner_expert_id: str | None = None
    # A copy of the owner's display fields, for an expert since archived.
    owner_name: str | None = None
    owner_avatar_url: str | None = None
    # One line: the result, the reason it failed, or what it is waiting on.
    line: str = ""
    occurred_at: datetime | None = None


class BriefingTasks(BaseModel):
    """The user's tasks as the brief saw them. Lists are capped for reading;
    the totals are counted before the cap."""

    finished: list[BriefingTaskItem] = Field(default_factory=list)
    failed: list[BriefingTaskItem] = Field(default_factory=list)
    waiting: list[BriefingTaskItem] = Field(default_factory=list)
    finished_total: int = Field(default=0, ge=0)
    failed_total: int = Field(default=0, ge=0)
    waiting_total: int = Field(default=0, ge=0)
    in_progress_total: int = Field(default=0, ge=0)

    @property
    def is_empty(self) -> bool:
        return not (self.finished_total or self.failed_total or self.waiting_total)


class BriefingContent(BaseModel):
    generated_at: datetime
    timezone: str
    zero_expert_fallback: bool
    run_items: list[BriefingRunItem]
    decision_items: list[BriefingDecisionItem]
    # Pre-cap totals. Each list above is truncated for rendering, so these are
    # what a consumer counts from; they default to 0 both so rows stored before
    # the field existed still validate and so "<= len(<list>)" reads as
    # "nothing was truncated". `ge=0` because a stored row is treated as
    # canonical — a negative count is a corrupt row, and failing validation
    # sends both readers down their existing recompute path rather than
    # letting it reach the card.
    decision_total: int = Field(default=0, ge=0)
    # Split by outcome because home reports completed and failed separately.
    completed_total: int = Field(default=0, ge=0)
    failed_total: int = Field(default=0, ge=0)
    # The AI-voice lede (see `narrative.py`), written once by the 9am job.
    # Optional because it is best-effort: rows stored before the field existed,
    # and any briefing whose narrative call failed, carry None and render as
    # template-only.
    narrative: str | None = None
    # None on rows stored before the brief covered tasks; home then joins
    # the tasks in itself.
    tasks: BriefingTasks | None = None
