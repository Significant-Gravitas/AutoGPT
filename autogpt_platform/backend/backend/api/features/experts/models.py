from datetime import datetime

from pydantic import BaseModel, Field, field_validator

AI_DISCLOSURE_RULE = "The expert discloses that it is AI when acting externally."
EXTERNAL_ACTION_APPROVAL_RULE = "External actions require approval."
PROTECTED_SOUL_RULES = (AI_DISCLOSURE_RULE, EXTERNAL_ACTION_APPROVAL_RULE)


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
    voice_preferences: str
    boundaries: str
    protected_soul_rules: list[str]
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


class ExpertRun(BaseModel):
    """One expert-attributed execution, for the /team Work surface."""

    execution_id: str
    graph_id: str
    agent_name: str
    library_agent_id: str | None
    status: str
    # "table" | "doc" | "image" | "unknown" — drives the typed viewer.
    output_type: str
    # Which output pin was classified, so the viewer opens exactly that value.
    output_key: str | None
    needs_review: bool
    started_at: datetime | None
    ended_at: datetime | None
    link: str | None


class ExpertDetachPreview(BaseModel):
    """What archiving the expert would pause — drives the confirm dialog."""

    schedule_names: list[str]
    trigger_names: list[str]


class HireResult(BaseModel):
    expert: Expert
    failed_preloads: list[str]


class ExpertSoulUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    identity: str = Field(min_length=1, max_length=10_000)
    voice_preferences: str = Field(max_length=4_000)
    boundaries: str = Field(max_length=4_000)

    @field_validator("name", "identity")
    @classmethod
    def strip_required_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field must not be blank")
        return stripped

    @field_validator("voice_preferences", "boundaries")
    @classmethod
    def strip_optional_fields(cls, value: str) -> str:
        # Whitespace-only input must collapse to "" so prompt rendering falls
        # back to "Not specified." instead of emitting a blank section.
        return value.strip()
