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


class RuleMemory(BaseModel):
    """Structured representation of a standing instruction or rule.

    Preserves the exact user intent rather than relying on LLM
    extraction to reconstruct it from prose.
    """

    instruction: str = Field(
        description="The actionable instruction (e.g. 'CC Sarah on client communications')"
    )
    actor: str | None = Field(
        default=None, description="Who performs or is subject to the rule"
    )
    trigger: str | None = Field(
        default=None,
        description="When the rule applies (e.g. 'client-related communications')",
    )
    negation: str | None = Field(
        default=None,
        description="What NOT to do, if applicable (e.g. 'do not use SMTP')",
    )


class ProcedureStep(BaseModel):
    """A single step in a multi-step procedure."""

    order: int = Field(description="Step number (1-based)")
    action: str = Field(description="What to do in this step")
    tool: str | None = Field(default=None, description="Tool or service to use")
    condition: str | None = Field(default=None, description="When/if this step applies")
    negation: str | None = Field(
        default=None, description="What NOT to do in this step"
    )


class ProcedureMemory(BaseModel):
    """Structured representation of a multi-step workflow.

    Steps with ordering, tools, conditions, and negations that don't
    decompose cleanly into fact triples.
    """

    description: str = Field(description="What this procedure accomplishes")
    steps: list[ProcedureStep] = Field(default_factory=list)


class MemoryEnvelope(BaseModel):
    """Structured wrapper for explicit memory storage.

    Serialized as JSON and ingested via ``EpisodeType.json`` so that
    Graphiti extracts entities from the ``content`` field while the
    metadata fields survive as episode-level context.

    For ``memory_kind=rule``, populate the ``rule`` field with a
    ``RuleMemory`` to preserve the exact instruction.  For
    ``memory_kind=procedure``, populate ``procedure`` with a
    ``ProcedureMemory`` for structured steps.
    """

    content: str = Field(
        description="The memory content — the actual fact, rule, or finding"
    )
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
    rule: RuleMemory | None = Field(
        default=None,
        description="Structured rule data — populate when memory_kind=rule",
    )
    procedure: ProcedureMemory | None = Field(
        default=None,
        description="Structured procedure data — populate when memory_kind=procedure",
    )
