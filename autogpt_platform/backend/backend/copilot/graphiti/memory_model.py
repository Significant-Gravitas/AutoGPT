"""Generic memory metadata model for Graphiti episodes.

Domain-agnostic envelope that works across business, fiction, research,
personal life, and arbitrary knowledge domains.  Designed so retrieval
can distinguish user-asserted facts from assistant-derived findings
and filter by scope.
"""

from enum import Enum

from pydantic import BaseModel, Field


class SourceKind(str, Enum):
    user_asserted = "user_asserted"
    assistant_derived = "assistant_derived"
    tool_observed = "tool_observed"


class MemoryKind(str, Enum):
    fact = "fact"
    preference = "preference"
    rule = "rule"
    finding = "finding"
    plan = "plan"
    event = "event"
    procedure = "procedure"


class MemoryStatus(str, Enum):
    active = "active"
    tentative = "tentative"
    superseded = "superseded"
    contradicted = "contradicted"


class MemoryEnvelope(BaseModel):
    """Structured wrapper for explicit memory storage.

    Serialized as JSON and ingested via ``EpisodeType.json`` so that
    Graphiti extracts entities from the ``content`` field while the
    metadata fields survive as episode-level context.
    """

    content: str = Field(description="The memory content — the actual fact, rule, or finding")
    source_kind: SourceKind = Field(default=SourceKind.user_asserted)
    scope: str = Field(
        default="real:global",
        description="Namespace: 'real:global', 'project:<name>', 'book:<title>', 'session:<id>'",
    )
    memory_kind: MemoryKind = Field(default=MemoryKind.fact)
    status: MemoryStatus = Field(default=MemoryStatus.active)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: str | None = Field(
        default=None,
        description="Origin reference — session_id, tool_call_id, or URL",
    )
