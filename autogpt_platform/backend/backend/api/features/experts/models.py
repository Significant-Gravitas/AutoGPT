from datetime import datetime

from pydantic import BaseModel


class ExpertWorkflowRef(BaseModel):
    id: str
    store_listing_version_id: str | None
    library_agent_id: str | None
    graph_id: str | None
    name: str | None
    description: str | None
    # Roster cadence + the schedule created from it at install time.
    # A cron with a null schedule_id means the schedule could not be
    # created yet (e.g. missing credentials) — the workflow needs setup.
    schedule_cron: str | None = None
    schedule_id: str | None = None


class Expert(BaseModel):
    id: str
    name: str
    avatar_url: str | None
    role: str
    tagline: str | None
    bio: str | None
    skills: list[str]
    identity: str
    is_template: bool
    source_template_id: str | None
    is_archived: bool
    workflows: list[ExpertWorkflowRef]
    # Latest expert-attributed execution, for the /team card's status line.
    last_run_at: datetime | None = None
    last_run_status: str | None = None
    # Weekly credit guardrail: effective budget (expert's own or the
    # platform default; None = guardrail disabled), current-week spend,
    # and the pause flag set on budget breach or archive.
    weekly_budget: int | None = None
    weekly_spend: int = 0
    schedules_paused_at: datetime | None = None


class ExpertDetachPreview(BaseModel):
    """What archiving the expert would pause — drives the confirm dialog."""

    schedule_names: list[str]
    trigger_names: list[str]


class HireResult(BaseModel):
    expert: Expert
    failed_preloads: list[str]
