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
    # Card fields the home dashboard reads off a stored briefing. All default
    # so rows written before the composer was unified still validate — the
    # home mapper falls back to the agent-name headline when `title` is empty.
    expert_role: str | None = None
    title: str = ""
    detail: str = ""
    occurred_at: datetime | None = None
    duration_seconds: float = 0
    cost_cents: int = 0


class BriefingDecisionItem(BaseModel):
    node_exec_id: str
    graph_exec_id: str
    title: str
    expert_id: str | None
    expert_name: str | None
    expert_avatar_url: str | None
    link: str


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
