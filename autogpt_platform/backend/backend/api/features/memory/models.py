"""Response models for the user-facing memory API (settings page)."""

from datetime import datetime

from pydantic import BaseModel


class MemoryScopeOverview(BaseModel):
    """Counts for one memory scope, shown on the settings page and in the
    erase-confirmation dialog."""

    expert_id: str | None = None
    facts: int
    entities: int
    episodes: int


class MemoryFact(BaseModel):
    """One live fact edge, as shown in the recent-memories list."""

    uuid: str
    fact: str | None = None
    name: str | None = None
    source: str
    target: str
    created_at: str | None = None


class MemoryFactListResponse(BaseModel):
    expert_id: str | None = None
    items: list[MemoryFact]


class ForgetFactResponse(BaseModel):
    """A forgotten fact. ``derived_forgotten`` counts the facts the dream had
    derived from it, retracted with it (``graphiti/recall_cascade.py``).
    ``resumed`` is True when the fact had already been erased by an earlier
    hard forget, and this one went on retracting and erasing what the dream
    derived from it (``graphiti/recall_forget.py``)."""

    uuid: str
    forgotten: bool
    derived_forgotten: int = 0
    resumed: bool = False


class EraseMemoryResponse(BaseModel):
    """Result of a scope wipe. ``deleted_nodes`` counts every node removed —
    entities, episodes (raw text included), and communities."""

    expert_id: str | None = None
    deleted_nodes: int
    erased: bool


class MemoryUnderstanding(BaseModel):
    """What Otto keeps about the caller's work (``data/understanding.py``), from
    onboarding and chat; ``updated_at`` is None when nothing is kept."""

    updated_at: datetime | None = None
    user_name: str | None = None
    job_title: str | None = None
    business_name: str | None = None
    industry: str | None = None
    business_size: str | None = None
    user_role: str | None = None
    key_workflows: list[str] = []
    daily_activities: list[str] = []
    pain_points: list[str] = []
    bottlenecks: list[str] = []
    manual_tasks: list[str] = []
    automation_goals: list[str] = []
    current_software: list[str] = []
    existing_automation: list[str] = []
    additional_notes: str | None = None


class ForgetUnderstandingResponse(BaseModel):
    forgotten: bool
