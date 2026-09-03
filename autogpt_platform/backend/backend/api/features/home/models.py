from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.api.features.executions.review.model import PendingHumanReviewModel


class HomeExpert(BaseModel):
    id: str
    name: str
    role: str
    avatar_url: str | None


class HomeAction(BaseModel):
    label: str
    href: str


class HomeAttentionItem(BaseModel):
    id: str
    kind: Literal["approval", "setup", "paused", "credits", "question"]
    priority: Literal["high", "normal"]
    title: str
    description: str
    why_it_matters: str
    expert: HomeExpert | None = None
    agent_name: str | None = None
    created_at: datetime | None = None
    preview: str | None = None
    review: PendingHumanReviewModel | None = None
    primary_action: HomeAction
    secondary_action: HomeAction | None = None


class HomeBriefingOutcome(BaseModel):
    id: str
    status: Literal["completed", "failed"]
    title: str
    summary: str
    expert: HomeExpert | None = None
    agent_name: str
    occurred_at: datetime | None = None
    duration_seconds: float = 0
    cost_cents: int = 0
    link: str | None = None


class HomeBriefing(BaseModel):
    generated_at: datetime
    window_started_at: datetime
    completed_count: int
    failed_count: int
    routine_count: int
    outcomes: list[HomeBriefingOutcome]
    # The AI-voice opening the copilot thread was posted with, read off the
    # stored briefing. None on the live path (nothing was generated) and
    # whenever the AI-summary flag is off.
    narrative: str | None = None
    # "persisted": anchored on the morning `UserBriefing` the copilot thread was
    # posted from, plus any run that finished after it. "live": no usable row
    # for today, so the rolling 24h window was recomputed instead.
    source: Literal["persisted", "live"] = "live"


class HomeActiveTask(BaseModel):
    id: str
    title: str
    status: Literal["running", "queued"]
    expert: HomeExpert | None = None
    started_at: datetime | None = None
    link: str | None = None


class HomeUpcomingTask(BaseModel):
    id: str
    title: str
    kind: Literal["agent", "followup"]
    expert: HomeExpert | None = None
    next_run_time: datetime


class HomeAgentStatus(BaseModel):
    expert: HomeExpert
    status: Literal["ready", "working", "needs_setup", "paused", "failed"]
    detail: str
    next_run_time: datetime | None = None
    # Expert-attributed spend over the same 7-day window as `HomeWeekSummary`.
    spend_cents: int = 0


class HomeTeamSummary(BaseModel):
    total: int
    ready: int
    working: int
    needs_attention: int
    # Sum of `spend_cents` across the listed agents, so the header total and
    # the rows under it always reconcile. Spend stamped to an archived expert
    # is therefore excluded, as is unattributed spend.
    spend_cents: int = 0


class HomeDailyActivity(BaseModel):
    date: date
    completed_count: int
    review_count: int
    failed_count: int


class HomeWeekSummary(BaseModel):
    run_count: int
    completed_count: int
    review_count: int
    failed_count: int
    total_runtime_seconds: float
    # Runs that actually contributed to `total_runtime_seconds`; dividing by
    # `run_count` would bias an average down with in-flight/stats-less runs.
    timed_run_count: int
    total_cost_cents: int
    credits_balance: int | None
    daily: list[HomeDailyActivity]


class HomeWorkActor(BaseModel):
    # "expert" = a hired expert did it, "autopilot" = the default copilot
    # assistant, "agent" = a graph run with no expert attribution.
    kind: Literal["expert", "autopilot", "agent"]
    name: str
    expert: HomeExpert | None = None


class HomeRecentWorkItem(BaseModel):
    id: str
    category: Literal["file", "integration", "schedule"]
    event_type: str
    title: str
    occurred_at: datetime
    provider: str | None = None
    file_id: str | None = None
    mime_type: str | None = None


class HomeRecentWorkGroup(BaseModel):
    actor: HomeWorkActor
    session_id: str | None = None
    session_title: str | None = None
    link: str | None = None
    latest_at: datetime
    items: list[HomeRecentWorkItem]
    more_count: int = 0


class HomeRecentWork(BaseModel):
    groups: list[HomeRecentWorkGroup] = Field(default_factory=list)
    total_count: int = 0


class HomeDashboardResponse(BaseModel):
    generated_at: datetime
    timezone: str
    attention: list[HomeAttentionItem]
    briefing: HomeBriefing
    active_tasks: list[HomeActiveTask]
    upcoming_tasks: list[HomeUpcomingTask]
    team: HomeTeamSummary
    agents: list[HomeAgentStatus]
    week: HomeWeekSummary
    # Optional with a default so pre-existing clients (and their fixtures)
    # keep validating; the backend always populates it.
    recent_work: HomeRecentWork = Field(default_factory=HomeRecentWork)
