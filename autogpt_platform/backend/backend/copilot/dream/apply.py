"""Apply a sanitized ``DreamOperations`` payload to the world.

Three side-effects, in order:
  1. Writes (consolidated facts) → ``status='active'`` MemoryEnvelope
     episodes via ``enqueue_episode``.
  2. Proposals (novel findings) → ``status='tentative'`` envelopes.
     Ratification (P-0.4) will flip these to active or supersede them.
  3. Demotions / entity invalidations → ``mark_edges_superseded`` /
     ``invalidate_entity_direct_neighbors`` against the FalkorDB driver.

A ``ChatSession`` shell (``metadata.kind='dream'`` +
``metadata.dream_pass_id``) is created up front so the MemoryEnvelope
provenance can reference its id; the assistant message holding
``summary_for_user`` is appended LAST, after the ops above, so a partway
failure leaves an empty dream rather than a narrative with no memory.
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
from backend.util.feature_flag import Flag, is_feature_enabled

from .schemas import (
    ConsolidatedFact,
    DemotionSummary,
    DreamDemotion,
    DreamOperations,
    DreamOperationsSnapshot,
    EntityInvalidation,
    EntityInvalidationSummary,
    ProposedFinding,
    WriteSummary,
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
) -> tuple[int, int, list[DemotionSummary]]:
    """Run mark_edges_superseded once per (reason, new_status) bucket.

    Returns ``(succeeded_count, failed_count, summaries)`` where each
    summary records the original DreamDemotion plus whether the
    underlying Cypher actually touched a row (``applied`` flag).
    """
    if not demotions:
        return 0, 0, []

    # Group by (new_status, reason) so we minimize round-trips.
    buckets: dict[tuple[str, str], list[str]] = {}
    for d in demotions:
        buckets.setdefault((d.new_status, d.reason), []).append(d.edge_uuid)

    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        # Indices live with the chat-write client; skip the per-driver
        # indexing race ("Buffer is closed" spam).
        build_indices=False,
    )
    succeeded = 0
    failed = 0
    succeeded_uuids: set[str] = set()
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
            succeeded_uuids.update(ok)
    finally:
        await driver.close()

    summaries = [
        DemotionSummary(
            edge_uuid=d.edge_uuid,
            reason=d.reason,
            new_status=d.new_status,
            applied=d.edge_uuid in succeeded_uuids,
        )
        for d in demotions
    ]
    return succeeded, failed, summaries


async def _apply_entity_invalidations(
    group_id: str,
    invalidations: list[EntityInvalidation],
) -> tuple[int, list[EntityInvalidationSummary]]:
    """Single-hop demotion of every :RELATES_TO around each invalidated entity.

    Returns ``(total_edges_touched, summaries)`` — summaries enumerate
    the per-entity edge uuids so callers can render or audit which
    edges fell off when an entity was invalidated.
    """
    if not invalidations:
        return 0, []
    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        # Indices live with the chat-write client; skip the per-driver
        # indexing race ("Buffer is closed" spam).
        build_indices=False,
    )
    total = 0
    summaries: list[EntityInvalidationSummary] = []
    try:
        for inv in invalidations:
            uuids = await invalidate_entity_direct_neighbors(
                driver,
                group_id=group_id,
                entity_uuid=inv.entity_uuid,
                reason=inv.reason,
            )
            total += len(uuids)
            summaries.append(
                EntityInvalidationSummary(
                    entity_uuid=inv.entity_uuid,
                    reason=inv.reason,
                    edges_touched=list(uuids),
                )
            )
    finally:
        await driver.close()
    return total, summaries


async def _create_dream_session(user_id: str, pass_id: str) -> str:
    """Create the dream-kind ChatSession shell and return its id.

    Written up front (before the memory ops) because the fact/proposal
    ``MemoryEnvelope`` provenance references this ``session_id``. The
    user-facing narrative is written separately, AFTER the ops land
    (``_write_dream_summary_message``), so a partway failure leaves an
    empty dream rather than a 'completed' narrative with no memory.

    We use a fresh uuid rather than the pass_id so re-runs of the same
    pass (admin retries on failure) each produce their own session row.
    """
    # Lazy import — avoids circular dependency at module-import time
    # AND keeps the dream-pass / chat-model coupling explicit. Routing
    # through ``chat_db()`` means the dream pass (running in the
    # Scheduler subprocess) auto-uses the DatabaseManager RPC client;
    # the DatabaseManager process itself uses the direct module.
    from backend.copilot.model import ChatSessionMetadata
    from backend.data.db_accessors import chat_db

    session_id = str(uuidlib.uuid4())
    await chat_db().create_chat_session(
        session_id=session_id,
        user_id=user_id,
        metadata=ChatSessionMetadata(kind="dream", dream_pass_id=pass_id),
    )
    return session_id


async def _write_dream_summary_message(
    session_id: str, pass_id: str, summary_for_user: str
) -> None:
    """Append the assistant narrative to an already-created dream session.

    Called at the END of ``apply_operations`` so the user-visible summary
    only appears once the memory ops above have been attempted.
    """
    from backend.data.db_accessors import chat_db

    body = summary_for_user.strip() or "Dream pass completed with no narrative output."
    await chat_db().add_chat_message(
        session_id=session_id,
        role="assistant",
        sequence=0,
        content=body,
        metadata={"dream_pass_id": pass_id},
    )


async def apply_operations(
    user_id: str,
    pass_id: str,
    ops: DreamOperations,
) -> dict[str, int | str | DreamOperationsSnapshot]:
    """Apply a sanitized DreamOperations to Graphiti + Postgres.

    Returns a small stats dict the orchestrator can fold into
    ``DreamPassResult``. Includes a ``snapshot`` key carrying the
    detailed ``DreamOperationsSnapshot`` payload for consumers that
    need per-operation rollups (eval, admin UI, future P9 SSE event).

    Postgres writes route through ``chat_db()`` / equivalent
    accessors. The dream pass runs in the Scheduler subprocess where
    Prisma is intentionally NOT locally connected — those accessors
    auto-route to the DatabaseManager RPC client. We deliberately do
    NOT call ``platform_db.connect()`` here: setting ``is_connected``
    True before the local Prisma engine is reachable causes a race
    with concurrent ``platform_cost_db()`` callers from
    ``token_tracking._safe_log`` (they'd see ``is_connected=True``,
    try direct Prisma, hit "All connection attempts failed" while
    the engine is still booting).
    """
    group_id = derive_group_id(user_id)

    # Phase A — create the session shell up front so the MemoryEnvelope
    # provenance can reference its id. The user-facing narrative summary
    # is written AFTER the ops (see below), so a partway failure leaves an
    # empty dream rather than a 'completed' narrative with no memory.
    session_id = await _create_dream_session(user_id=user_id, pass_id=pass_id)

    written = 0
    write_summaries: list[WriteSummary] = []
    for i, fact in enumerate(ops.writes):
        if await _write_consolidated_fact(
            user_id, pass_id, i, fact, session_id=session_id
        ):
            written += 1
            write_summaries.append(
                WriteSummary(
                    content=fact.content,
                    scope=fact.scope,
                    confidence=fact.confidence,
                    status="active",
                    source_episode_uuids=list(fact.source_episode_uuids),
                )
            )

    proposed = 0
    proposal_summaries: list[WriteSummary] = []
    for i, prop in enumerate(ops.proposals):
        if await _write_proposed_finding(
            user_id, pass_id, i, prop, session_id=session_id
        ):
            proposed += 1
            proposal_summaries.append(
                WriteSummary(
                    content=prop.content,
                    scope=prop.scope,
                    confidence=prop.confidence,
                    status="tentative",
                    source_episode_uuids=list(prop.source_episode_uuids),
                    source_fact_uuids=list(prop.source_fact_uuids),
                )
            )

    demoted_ok, demoted_fail, demotion_summaries = await _apply_demotions(
        user_id, group_id, ops.demotions
    )
    # Entity invalidation single-hop demotes every edge around the
    # entity — the most destructive op in the pass — so it stays behind
    # its own LD flag for staged rollout, independent of the dream pass
    # being enabled. Truthiness check short-circuits the flag eval when
    # the model proposed nothing to invalidate.
    if ops.entity_invalidations and await is_feature_enabled(
        Flag.DREAM_PASS_INVALIDATE_ENTITY, user_id
    ):
        entity_edges_demoted, entity_summaries = await _apply_entity_invalidations(
            group_id, ops.entity_invalidations
        )
    else:
        entity_edges_demoted, entity_summaries = 0, []

    # Narrative summary last — only surface the user-facing dream story
    # once the memory ops above have been attempted.
    await _write_dream_summary_message(session_id, pass_id, ops.summary_for_user)

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

    snapshot = DreamOperationsSnapshot(
        writes=write_summaries,
        proposals=proposal_summaries,
        demotions=demotion_summaries,
        entity_invalidations=entity_summaries,
    )

    return {
        "session_id": session_id,
        "consolidated_count": written,
        "proposal_count": proposed,
        "demotion_count": demoted_ok,
        "demotion_failed_count": demoted_fail,
        "entity_invalidation_count": entity_edges_demoted,
        "snapshot": snapshot,
    }
