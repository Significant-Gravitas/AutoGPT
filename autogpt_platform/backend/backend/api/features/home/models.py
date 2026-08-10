from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

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
    kind: Literal["approval", "setup", "paused", "credits"]
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


class HomeTeamSummary(BaseModel):
    total: int
    ready: int
    working: int
    needs_attention: int


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


class HomeDashboardResponse(BaseModel):
    generated_at: datetime
    timezone: str
    is_demo: bool
    attention: list[HomeAttentionItem]
    briefing: HomeBriefing
    active_tasks: list[HomeActiveTask]
    upcoming_tasks: list[HomeUpcomingTask]
    team: HomeTeamSummary
    agents: list[HomeAgentStatus]
    week: HomeWeekSummary
