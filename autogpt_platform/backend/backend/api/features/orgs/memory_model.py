"""Pydantic models for org shared-memory governance (held-memory review)."""

from pydantic import BaseModel


class HeldMemory(BaseModel):
    """A single tentative ('held') memory edge awaiting org-admin review."""

    id: str  # the RELATES_TO edge uuid
    tier: str  # "org" | "team"
    team_id: str | None = None  # set only for team-tier edges
    team_name: str | None = None
    name: str | None = None  # extracted relation name (e.g. "works_on")
    fact: str | None = None  # the fact text
    created_at: str | None = None
    # Provenance the write stamped. The edge carries no per-user attribution;
    # source_kind + provenance (session id) are the available origin signals.
    source_kind: str | None = None
    provenance: str | None = None


class HeldMemoryListResponse(BaseModel):
    org_id: str
    items: list[HeldMemory]


class MemoryActionResult(BaseModel):
    """Result of approving or rejecting a held memory."""

    id: str
    action: str  # "approve" | "reject"
    applied: bool  # False when the edge was already resolved (idempotent no-op)
    tier: str
    team_id: str | None = None
