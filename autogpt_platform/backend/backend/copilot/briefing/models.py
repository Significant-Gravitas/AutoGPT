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
    summary: str | None
    link: str | None


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
