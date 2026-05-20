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
    execution_path: Literal["batch", "sync_baseline"] = "sync_baseline"

    # Per-phase telemetry — null when the phase did not run.
    consolidated_count: int = 0
    proposal_count: int = 0
    demotion_count: int = 0
    entity_invalidation_count: int = 0

    summary_for_user: str = ""
    dream_session_id: str | None = None

    # Failure / skip signalling — mirrors RebuildResponse.
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
