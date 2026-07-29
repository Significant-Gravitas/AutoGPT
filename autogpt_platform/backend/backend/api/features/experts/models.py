from pydantic import BaseModel


class ExpertWorkflowRef(BaseModel):
    id: str
    store_listing_version_id: str | None
    library_agent_id: str | None
    graph_id: str | None
    name: str | None
    description: str | None


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


class HireResult(BaseModel):
    expert: Expert
    failed_preloads: list[str]
