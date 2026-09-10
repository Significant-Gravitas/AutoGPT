"""Response models for the user-facing memory API (settings page)."""

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
    uuid: str
    forgotten: bool


class EraseMemoryResponse(BaseModel):
    """Result of a scope wipe. ``deleted_nodes`` counts every node removed —
    entities, episodes (raw text included), and communities."""

    expert_id: str | None = None
    deleted_nodes: int
    erased: bool
