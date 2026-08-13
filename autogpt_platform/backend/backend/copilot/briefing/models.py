from datetime import datetime

from pydantic import BaseModel


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
    # How many decisions were pending in total, before the list above was
    # capped. Defaults to 0 so briefings stored before this field existed
    # still validate; the renderer treats "<= len(decision_items)" as
    # "nothing was truncated".
    decision_total: int = 0
