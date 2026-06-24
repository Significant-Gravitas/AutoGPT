"""Structured outputs for the three-phase dream pipeline.

Each phase returns a typed Pydantic model so the orchestrator can pass
phase 1's output into phase 2's prompt without freeform JSON parsing.
The phase 3 sanitizer's output (``DreamOperations``) is what
``apply.py`` consumes when writing back to Graphiti + Postgres.

Per ``dream/p0-spec.md`` §2 "Schemas (sketch)".
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from backend.copilot.graphiti.memory_model import MemoryKind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 1 — Consolidation
# ---------------------------------------------------------------------------


class ConsolidatedFact(BaseModel):
    """A cluster of related facts merged into a single canonical statement.

    Phase 1 output. Provenance always points back to the source episodes
    so apply.py can record where the consolidation came from.
    """

    content: str = Field(description="Canonical statement of the consolidated fact.")
    scope: str = Field(
        default="real:global",
        description="Memory scope namespace, e.g. 'real:global' or 'project:foo'.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Phase 1 model's confidence in the consolidated statement.",
    )
    source_episode_uuids: list[str] = Field(
        default_factory=list,
        description="UUIDs of the :Episodic nodes the fact was consolidated from.",
    )


class ConsolidationOutput(BaseModel):
    facts: list[ConsolidatedFact] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 2 — Recombination
# ---------------------------------------------------------------------------


class ProposedFinding(BaseModel):
    """A novel connection or weak-link discovery from phase 2.

    All proposals land as ``status=tentative`` in apply.py and ride the
    P-0.4 ratification loop. The ``rationale`` is recorded as part of
    the episode body so a reviewer (or future audit) can see *why* the
    dream pass proposed the finding.
    """

    content: str = Field(description="The proposed finding.")
    scope: str = Field(
        default="real:global",
        description="Memory scope.",
    )
    memory_kind: MemoryKind = Field(
        default=MemoryKind.finding,
        description="Envelope kind — finding | rule | preference | plan.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Phase 2 model's self-rated confidence.",
    )
    rationale: str = Field(
        description="Why the dream pass proposes this — recorded for audit.",
    )
    source_episode_uuids: list[str] = Field(default_factory=list)
    source_fact_uuids: list[str] = Field(default_factory=list)


class RecombinationOutput(BaseModel):
    proposals: list[ProposedFinding] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _drop_proposals_with_invalid_memory_kind(cls, data: Any) -> Any:
        """LLMs occasionally invent kinds outside the MemoryKind enum
        (observed: "inference", "meta"). Drop those proposals rather
        than failing the whole phase — the surviving valid proposals
        still get written as tentative findings."""
        if not isinstance(data, dict):
            return data
        raw_proposals = data.get("proposals")
        if not isinstance(raw_proposals, list):
            return data
        valid_kinds = {k.value for k in MemoryKind}
        kept: list[Any] = []
        for p in raw_proposals:
            if isinstance(p, dict):
                kind = p.get("memory_kind")
                if kind is not None and kind not in valid_kinds:
                    logger.info(
                        "dream phase_2: dropping proposal with unknown memory_kind=%r",
                        kind,
                    )
                    continue
            kept.append(p)
        data["proposals"] = kept
        return data


# ---------------------------------------------------------------------------
# Phase 3 — Sanitize + Gate (the final DreamOperations)
# ---------------------------------------------------------------------------


class DreamDemotion(BaseModel):
    """Demote an existing :RELATES_TO edge.

    Sets ``expired_at`` and ``status``; ``invalid_at`` is reserved for
    contradiction-detector world-changes per Snodgrass bi-temporal
    semantics (see ``dream/p0-spec.md`` §4 and the audit at §6.13).
    """

    edge_uuid: str = Field(description="UUID of the :RELATES_TO edge to demote.")
    reason: str = Field(
        description=(
            "Short reason recorded on the edge. Examples: 'stale_fact', "
            "'contradicted_by:{uuid}', 'entity_invalidated:{uuid}', "
            "'user_signal', 'unratified', 'web_contradicted:{url}'."
        ),
    )
    new_status: Literal["superseded", "contradicted"] = "superseded"


class EntityInvalidation(BaseModel):
    """Demote every :RELATES_TO edge directly attached to an entity.

    Single-hop only — apply.py calls ``invalidate_entity_direct_neighbors``
    which clamps to ``[r:RELATES_TO]-(other)``; never expands to
    neighbors-of-neighbors.
    """

    entity_uuid: str
    reason: str


class DreamOperations(BaseModel):
    """Phase 3 output — what apply.py writes back to the world.

    Guardrails enforced by phase 3's prompt and double-checked in
    apply.py:
      * ≤ ``max_demotions_per_pass`` demotions per pass (runaway-demotion
        mitigation per spec §3 / TODO P0.3b).
      * Scope match enforced — proposals cannot cross scopes.
      * Empty ``writes`` and ``proposals`` is fine; a pass can be no-op.
    """

    writes: list[ConsolidatedFact] = Field(
        default_factory=list,
        description="Phase 1 consolidated facts that survived the sanitizer.",
    )
    proposals: list[ProposedFinding] = Field(
        default_factory=list,
        description="Phase 2 proposals that survived the sanitizer.",
    )
    demotions: list[DreamDemotion] = Field(default_factory=list)
    entity_invalidations: list[EntityInvalidation] = Field(default_factory=list)
    summary_for_user: str = Field(
        default="",
        description="Short narrative for the dream-kind ChatSession body.",
    )


# ---------------------------------------------------------------------------
# DreamPass — what the orchestrator returns to its caller
# ---------------------------------------------------------------------------


class WriteSummary(BaseModel):
    """One line per envelope written by apply.py.

    Carries the durable Graphiti edge uuid so the eval / admin UI / the
    future inline chat-stream `dream.operations` event (P9 daydreaming)
    can render or audit individual operations. ``edge_uuid`` is None
    when the underlying ``enqueue_episode`` call did not return one
    (FalkorDB unreachable, etc.) — we still record the attempted write.
    """

    edge_uuid: str | None = None
    content: str
    scope: str = "real:global"
    confidence: float | None = None
    status: Literal["active", "tentative"] = "active"
    source_episode_uuids: list[str] = Field(default_factory=list)
    source_fact_uuids: list[str] = Field(default_factory=list)


class DemotionSummary(BaseModel):
    """One line per demoted RELATES_TO edge."""

    edge_uuid: str
    reason: str
    new_status: Literal["superseded", "contradicted"]
    applied: bool = True
    """False when the underlying Cypher reported zero rows touched —
    typically because the edge uuid was stale by the time apply.py ran."""


class EntityInvalidationSummary(BaseModel):
    """Per-entity invalidation rollup."""

    entity_uuid: str
    reason: str
    edges_touched: list[str] = Field(default_factory=list)


class DreamOperationsSnapshot(BaseModel):
    """Detailed per-operation rollup of a single dream pass.

    Shape used by three downstream consumers:
      1. The admin memory-visualizer UI (renders the per-edge detail).
      2. AgentProbe eval scorers (read via ``rawExchangeKey``).
      3. The future chat-stream ``dream.operations`` SSE event (P6 + P9
         daydreaming, see ``dream/TODO.md``).

    Kept additive on ``DreamPassResult`` so adding fields here doesn't
    bump the count-only top-level columns existing clients rely on.
    """

    writes: list[WriteSummary] = Field(default_factory=list)
    proposals: list[WriteSummary] = Field(default_factory=list)
    demotions: list[DemotionSummary] = Field(default_factory=list)
    entity_invalidations: list[EntityInvalidationSummary] = Field(default_factory=list)


class PhaseUsage(BaseModel):
    """Per-phase token + cost telemetry."""

    phase: Literal["consolidate", "recombine", "sanitize"]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None
    """``None`` when the provider didn't return a cost and the model
    isn't in ``model_pricing.py`` — caller treats as unknown rather
    than zero. ``0.0`` is legitimate (zero tokens, edge case)."""


class DreamPassUsage(BaseModel):
    """Aggregate usage across all phases of one dream pass.

    Surfaced on ``DreamPassResult.usage`` so eval / admin UI / billing
    can read tokens + cost without recomputing. Written to
    ``PlatformCostLog`` from apply.py with ``provider='dream_pass'``;
    charged against the user's rate-limit window via
    ``persist_and_record_usage`` so dream-pass spend rolls into the
    same daily/weekly subscription tier budget as chat-execution
    overage.
    """

    phases: list[PhaseUsage] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cost_usd: float | None = None
    """Sum of phase cost_usd values; ``None`` if any phase was unknown."""
    discount_applied: float = 0.0
    """Execution-path discount factor in [0, 1] — 0.0 for sync_baseline,
    0.5 for anthropic_batch / openai_batch. The ``total_cost_usd``
    above already has this factored in; recorded separately so the
    cost ledger can audit the savings."""


class DreamPassResult(BaseModel):
    """Return value of ``execute_dream_pass`` and the admin API.

    Mirrors ``RebuildResponse`` from the community-rebuild endpoint so
    the admin frontend can render both with the same toast logic.
    """

    user_id: str
    pass_id: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    elapsed_seconds: float | None = None
    execution_path: Literal["sync_baseline", "anthropic_batch", "openai_batch"] = (
        "sync_baseline"
    )

    # Per-phase telemetry — null when the phase did not run.
    consolidated_count: int = 0
    proposal_count: int = 0
    demotion_count: int = 0
    entity_invalidation_count: int = 0

    summary_for_user: str = ""
    dream_session_id: str | None = None

    # Detailed per-operation rollup. ``None`` when the pass was skipped
    # or errored before phase 3 produced operations; ``DreamOperationsSnapshot()``
    # with empty lists when the pass ran but produced no operations.
    operations: DreamOperationsSnapshot | None = None

    # Token + cost telemetry across all phases that actually ran.
    # ``None`` for skipped passes (lock_held / no_input / insufficient_credits).
    # Populated even on partial failures — we still paid for the phases
    # that ran before the failure.
    usage: DreamPassUsage | None = None

    # Failure / skip signalling — mirrors RebuildResponse.
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
