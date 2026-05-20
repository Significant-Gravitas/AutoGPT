"""Apply a sanitized ``DreamOperations`` payload to the world.

Three side-effects, in order:
  1. Writes (consolidated facts) → ``status='active'`` MemoryEnvelope
     episodes via ``enqueue_episode``.
  2. Proposals (novel findings) → ``status='tentative'`` envelopes.
     Ratification (P-0.4) will flip these to active or supersede them.
  3. Demotions / entity invalidations → ``mark_edges_superseded`` /
     ``invalidate_entity_direct_neighbors`` against the FalkorDB driver.

Finally writes a ``ChatSession`` with ``metadata.kind='dream'`` plus
``metadata.dream_pass_id`` and a single assistant message containing
``summary_for_user`` so the dream is reviewable in the admin UI.
"""

from __future__ import annotations

import logging
import uuid as uuidlib
from datetime import datetime, timezone

from backend.copilot.graphiti.client import derive_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.copilot.graphiti.ingest import enqueue_episode
from backend.copilot.graphiti.memory_model import (
    MemoryEnvelope,
    MemoryKind,
    MemoryStatus,
    SourceKind,
)
from backend.copilot.tools.graphiti_forget import (
    invalidate_entity_direct_neighbors,
    mark_edges_superseded,
)

from .schemas import (
    ConsolidatedFact,
    DreamDemotion,
    DreamOperations,
    EntityInvalidation,
    ProposedFinding,
)

logger = logging.getLogger(__name__)


def _provenance(pass_id: str, phase: str) -> str:
    """Provenance string written into the MemoryEnvelope.

    Format matches Graphiti audit §6.12 / TODO P-1.5 grain — encodes
    the dream pass id and the phase so ratification can find originating
    dream-write episodes by prefix-match.
    """
    return f"dream:{pass_id}:{phase}:{datetime.now(timezone.utc).isoformat()}"


def _episode_name(pass_id: str, phase: str, counter: int) -> str:
    """Stable, auditable episode name for dream-derived writes."""
    return f"dream_{pass_id}_{phase}_{counter:03d}"


async def _write_consolidated_fact(
    user_id: str,
    pass_id: str,
    counter: int,
    fact: ConsolidatedFact,
    session_id: str,
) -> bool:
    envelope = MemoryEnvelope(
        content=fact.content,
        source_kind=SourceKind.assistant_derived,
        memory_kind=MemoryKind.fact,
        status=MemoryStatus.active,
        confidence=fact.confidence,
        scope=fact.scope,
        provenance=_provenance(pass_id, "consolidate"),
    )
    return await enqueue_episode(
        user_id=user_id,
        session_id=session_id,
        name=_episode_name(pass_id, "consolidate", counter),
        episode_body=envelope.model_dump_json(),
        source_description=(
            f"dream-pass consolidation; src_episodes="
            f"{','.join(fact.source_episode_uuids[:5])}"
        ),
        is_json=True,
    )


async def _write_proposed_finding(
    user_id: str,
    pass_id: str,
    counter: int,
    finding: ProposedFinding,
    session_id: str,
) -> bool:
    envelope = MemoryEnvelope(
        content=finding.content,
        source_kind=SourceKind.assistant_derived,
        memory_kind=finding.memory_kind,
        status=MemoryStatus.tentative,
        confidence=finding.confidence,
        scope=finding.scope,
        provenance=_provenance(pass_id, "recombine"),
    )
    description_parts: list[str] = ["dream-pass proposal"]
    if finding.rationale:
        description_parts.append(f"rationale={finding.rationale[:240]}")
    if finding.source_fact_uuids:
        description_parts.append(f"src_facts={','.join(finding.source_fact_uuids[:5])}")
    return await enqueue_episode(
        user_id=user_id,
        session_id=session_id,
        name=_episode_name(pass_id, "recombine", counter),
        episode_body=envelope.model_dump_json(),
        source_description="; ".join(description_parts),
        is_json=True,
    )


async def _apply_demotions(
    user_id: str,
    group_id: str,
    demotions: list[DreamDemotion],
) -> tuple[int, int]:
    """Run mark_edges_superseded once per (reason, new_status) bucket.

    Returns (succeeded, failed).
    """
    if not demotions:
        return 0, 0

    # Group by (new_status, reason) so we minimize round-trips.
    buckets: dict[tuple[str, str], list[str]] = {}
    for d in demotions:
        buckets.setdefault((d.new_status, d.reason), []).append(d.edge_uuid)

    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
    )
    succeeded = 0
    failed = 0
    try:
        for (new_status, reason), uuids in buckets.items():
            ok, bad = await mark_edges_superseded(
                driver,
                uuids,
                reason=reason,
                new_status=new_status,  # type: ignore[arg-type]
                user_id=user_id,
            )
            succeeded += len(ok)
            failed += len(bad)
    finally:
        await driver.close()

    return succeeded, failed


async def _apply_entity_invalidations(
    group_id: str,
    invalidations: list[EntityInvalidation],
) -> int:
    """Single-hop demotion of every :RELATES_TO around each invalidated entity.

    Returns total count of edges that were touched.
    """
    if not invalidations:
        return 0
    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
    )
    total = 0
    try:
        for inv in invalidations:
            uuids = await invalidate_entity_direct_neighbors(
                driver,
                group_id=group_id,
                entity_uuid=inv.entity_uuid,
                reason=inv.reason,
            )
            total += len(uuids)
    finally:
        await driver.close()
    return total


async def _write_dream_session(
    user_id: str,
    pass_id: str,
    summary_for_user: str,
) -> str:
    """Create a dream-kind ChatSession + an assistant message holding the summary.

    Returns the session id. We use a fresh uuid here rather than the
    pass_id so that re-runs of the same pass (admin retries on
    failure) can each produce their own session row.
    """
    # Lazy import — avoids a circular dependency between dream/ and copilot/db
    # at module-import time.
    from backend.copilot.db import add_chat_message, create_chat_session
    from backend.copilot.model import ChatSessionMetadata

    session_id = str(uuidlib.uuid4())
    await create_chat_session(
        session_id=session_id,
        user_id=user_id,
        metadata=ChatSessionMetadata(kind="dream", dream_pass_id=pass_id),
    )
    body = summary_for_user.strip() or "Dream pass completed with no narrative output."
    await add_chat_message(
        session_id=session_id,
        role="assistant",
        sequence=0,
        content=body,
        metadata={"dream_pass_id": pass_id},
    )
    return session_id


async def apply_operations(
    user_id: str,
    pass_id: str,
    ops: DreamOperations,
) -> dict[str, int | str]:
    """Apply a sanitized DreamOperations to Graphiti + Postgres.

    Returns a small stats dict the orchestrator can fold into
    ``DreamPassResult``.
    """
    group_id = derive_group_id(user_id)

    # Phase A — write a placeholder session up front so the
    # MemoryEnvelope provenance can reference it. We could also use
    # the session_id as part of provenance, but for slice 1 the
    # pass_id is sufficient.
    session_id = await _write_dream_session(
        user_id=user_id, pass_id=pass_id, summary_for_user=ops.summary_for_user
    )

    written = 0
    for i, fact in enumerate(ops.writes):
        if await _write_consolidated_fact(
            user_id, pass_id, i, fact, session_id=session_id
        ):
            written += 1

    proposed = 0
    for i, prop in enumerate(ops.proposals):
        if await _write_proposed_finding(
            user_id, pass_id, i, prop, session_id=session_id
        ):
            proposed += 1

    demoted_ok, demoted_fail = await _apply_demotions(user_id, group_id, ops.demotions)
    entity_edges_demoted = await _apply_entity_invalidations(
        group_id, ops.entity_invalidations
    )

    logger.info(
        "Dream pass %s applied for user %s: "
        "writes=%d proposals=%d demoted=%d (failed=%d) entity_edges=%d",
        pass_id,
        user_id[:12],
        written,
        proposed,
        demoted_ok,
        demoted_fail,
        entity_edges_demoted,
    )

    return {
        "session_id": session_id,
        "consolidated_count": written,
        "proposal_count": proposed,
        "demotion_count": demoted_ok,
        "demotion_failed_count": demoted_fail,
        "entity_invalidation_count": entity_edges_demoted,
    }
