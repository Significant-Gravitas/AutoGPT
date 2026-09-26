"""Generic memory metadata model for Graphiti episodes.

Domain-agnostic envelope that works across business, fiction, research,
personal life, and arbitrary knowledge domains.  Designed so retrieval
can distinguish user-asserted facts from assistant-derived findings
and filter by scope.

Also holds the fact-status and forget vocabulary shared by the recall
layer (``recall.py``) and the chat tools that report on it.
"""

from enum import Enum

from pydantic import BaseModel, Field, ValidationError


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
    # Only ``active`` and ``tentative`` facts are recalled (``recall.py``);
    # a fact in any other state stays on the graph for audit.
    active = "active"
    tentative = "tentative"
    superseded = "superseded"
    contradicted = "contradicted"
    retracted = "retracted"
    """Set by an explicit user or API forget (``recall_forget.retract``). A
    retracted fact is never recalled; the edge is kept for audit."""


class MemoryForgetFailureCode(str, Enum):
    """Stable, machine-switchable reason a forget delete failed.

    The frontend/model can branch on this code (retry vs. give up) without
    parsing the free-text ``reason``. New codes may be added over time, so
    consumers must tolerate unknown values.
    """

    NO_MATCH = "no_match"
    QUERY_ERROR = "query_error"
    CLEANUP_ERROR = "cleanup_error"
    """The fact was forgotten, but the clean-up after it (redacting or
    deleting what it came from) failed. Recall keeps the text hidden anyway,
    and forgetting it again is safe."""


# Reason given when a forget matched no edge: the UUID is stale, already
# deleted, or not a forgettable edge type.
FORGET_NO_MATCH_REASON = (
    "No matching edge found — it may already be deleted, or the UUID is not a "
    "forgettable edge (RELATES_TO, MENTIONS, HAS_MEMBER)."
)


class MemoryForgetFailure(BaseModel):
    """One edge that could not be deleted, with an actionable reason.

    Surfaced so the assistant (and user) can tell *why* a delete failed —
    e.g. the edge was not found vs. the query itself errored — instead of a
    bare "N failed" count that gives the model nothing to act on.
    """

    uuid: str
    code: MemoryForgetFailureCode
    reason: str

    @classmethod
    def no_match(cls, uuid: str) -> "MemoryForgetFailure":
        return cls(
            uuid=uuid,
            code=MemoryForgetFailureCode.NO_MATCH,
            reason=FORGET_NO_MATCH_REASON,
        )

    @classmethod
    def query_error(cls, uuid: str, exc: Exception) -> "MemoryForgetFailure":
        """The reason names the exception type and its first argument (e.g.
        FalkorDB's ``Unknown function 'datetime'``), so the model can tell a
        real query error from a plain no-match; never the full ``repr``,
        which can carry connection details. Log the exception where caught.
        """
        return cls(
            uuid=uuid,
            code=MemoryForgetFailureCode.QUERY_ERROR,
            reason=f"Deletion query failed: {_describe(exc)}",
        )

    @classmethod
    def cleanup_error(cls, uuid: str, exc: Exception) -> "MemoryForgetFailure":
        """The fact is forgotten but its clean-up failed; the reason says so,
        so the model neither reports a plain success nor a lost forget."""
        return cls(
            uuid=uuid,
            code=MemoryForgetFailureCode.CLEANUP_ERROR,
            reason=(
                "Forgotten and no longer recalled, but the clean-up after it "
                f"failed: {_describe(exc)}. Forgetting it again is safe."
            ),
        )


def _describe(exc: Exception) -> str:
    detail = exc.args[0] if exc.args else type(exc).__name__
    return f"{type(exc).__name__}: {detail}"


class ForgetResult(BaseModel):
    """What one ``recall_forget.retract`` call did.

    ``deleted`` lists the edges retracted (soft) or removed (hard);
    ``failures`` holds one entry per requested uuid that was not, in the
    shape ``memory_forget_confirm`` reports, plus a ``cleanup_error`` for an
    edge whose clean-up failed, which is in ``deleted`` too when its own write
    landed. The episode and entity lists record the clean-up done: a hard
    forget empties an episode nothing else cites into a tombstone rather than
    deleting it, so the chat session it came from stays known.
    """

    deleted: list[str] = Field(default_factory=list)
    failures: list[MemoryForgetFailure] = Field(default_factory=list)
    redacted_episodes: list[str] = Field(default_factory=list)
    tombstoned_episodes: list[str] = Field(default_factory=list)
    deleted_entities: list[str] = Field(default_factory=list)


def envelope_provenance(content: str | None) -> str | None:
    """The ``provenance`` an episode body records when it is a
    ``MemoryEnvelope`` (``session:<id>#msg:<n>`` for a stored memory)."""
    try:
        return _EnvelopeProvenance.model_validate_json(content or "").provenance
    except ValidationError:
        return None


class _EnvelopeProvenance(BaseModel):
    provenance: str | None = None


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

    user: str | None = Field(
        default=None,
        description="Display name of the person the memory is about",
    )
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
