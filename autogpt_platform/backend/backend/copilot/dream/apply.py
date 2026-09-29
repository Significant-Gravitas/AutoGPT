"""Apply a sanitized ``DreamOperations`` payload to the world.

Three side-effects, in order:
  1. Writes (consolidated facts) → ``status='active'`` MemoryEnvelope
     episodes via ``enqueue_episode``.
  2. Proposals (novel findings) → ``status='tentative'`` envelopes.
     Ratification (P-0.4) will flip these to active or supersede them.
  3. Demotions / entity invalidations, written by ``demotions.py`` with the
     recall guard in each statement.

Every write and proposal must cite what the pass read in its own scope;
one that cites nothing of it is dropped first and counted (``citations.py``).

A ``ChatSession`` shell (``metadata.kind='dream'`` +
``metadata.dream_pass_id``) is created up front so the MemoryEnvelope
provenance can reference its id; the assistant message holding
``summary_for_user`` is appended LAST, after the ops above, so a partway
failure leaves an empty dream rather than a narrative with no memory.
A pass with no operations left at all creates neither — see the empty-pass
guard at the top of ``apply_operations``.
"""

from __future__ import annotations

import logging
import uuid as uuidlib
from collections.abc import Collection, Mapping
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from backend.copilot.graphiti.ingest import (
    IngestionCompletion,
    enqueue_episode,
    wait_for_ingestion,
)
from backend.copilot.graphiti.memory_model import (
    MemoryEnvelope,
    MemoryKind,
    MemoryStatus,
    SourceKind,
)
from backend.copilot.graphiti.recall_citations import Citations
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.transports import resolve_default_chat_route

from .citations import source_description, validated_citations
from .demotions import apply_demotions
from .fetch import DREAM_EPISODE_NAME_PREFIX
from .locks import DreamLockHandle, DreamLockLostError
from .schemas import (
    ConsolidatedFact,
    DreamOperations,
    DreamOperationsSnapshot,
    IngestionDrainStatus,
    ProposedFinding,
    WriteSummary,
)

if TYPE_CHECKING:
    from .lease import ApplyLease

logger = logging.getLogger(__name__)

# Upper bound on waiting for the per-user ingestion queue to drain before
# apply_operations returns. Dev numbers put a single add_episode at 7-131s,
# so a full pass (up to 30 writes + 20 proposals) can take far longer than
# any defensible in-lock wait. The caller holds the dream lock until apply
# returns, and renews it right before the drain to a budget that covers this
# cap (``LOCK_DRAIN_RENEWAL_SECONDS``). Past the cap the enqueued episodes
# keep processing fire-and-forget in this process: apply warns and reports
# ``timed_out`` rather than failing the pass.
INGESTION_DRAIN_TIMEOUT_SECONDS = 300

# The dream-lock TTL set right before the ingestion drain on the sync path
# (see ``apply_operations``' ``lock_handle``), sized to cover the drain cap
# plus the demotions / entity invalidations / summary write that follow it
# inside apply. The pass has renewed its lock before each phase and again
# before its first write (``lease.py``); this renewal proves it still holds
# the lock before the longest and most destructive tail, and can shorten
# what the earlier ones granted. Like every renewal it is a timer, not a
# storage fence: a tail that outruns this TTL writes on after the lock
# lapsed, and a newer pass may have taken the scope by then.
LOCK_DRAIN_RENEWAL_SECONDS = INGESTION_DRAIN_TIMEOUT_SECONDS + 180

# Drain bound for the Anthropic batch path. apply runs there inside
# ``handle_dream_batch_result``, which ``BatchExecutor.walk_once`` awaits
# SERIALLY in its single poll loop (one pending entry at a time). A 300s
# in-line drain would stall the poll/dispatch of every OTHER user's pending
# batch for the whole window — and per the drain math above a full pass
# almost never drains in 300s anyway, so the cost is paid for near-zero
# benefit while risking MAX_BATCH_LIFETIME_SECONDS expiry on the batches
# stuck behind it. The batch path therefore SKIPS the drain (0 = no wait):
# the enqueued episodes still process fire-and-forget in the executor
# process, and the pass honestly reports ``IngestionDrainStatus.skipped`` so
# the non-drained state is visible downstream instead of masked as success.
BATCH_INGESTION_DRAIN_TIMEOUT_SECONDS = 0


def drain_status_from_stats(
    stats: Mapping[str, object],
) -> IngestionDrainStatus:
    """Coerce the apply-stats drain flag into the typed tri-state enum.

    Shared by both the sync orchestrator and the batch callback so the
    read lives in exactly one place. Fail-closed: a missing or malformed
    key reads as ``timed_out`` (writes potentially at risk), never as a
    confirmed drain — missing observability must not present as success.
    The batch path always writes ``skipped`` explicitly, so the default
    only bites when apply produced no stats at all (e.g. an upstream bug).
    """
    raw = stats.get("ingestion_drain_status")
    if isinstance(raw, IngestionDrainStatus):
        return raw
    if isinstance(raw, str):
        try:
            return IngestionDrainStatus(raw)
        except ValueError:
            return IngestionDrainStatus.timed_out
    return IngestionDrainStatus.timed_out


def _provenance(pass_id: str, phase: str) -> str:
    """Provenance string written into the MemoryEnvelope.

    Format matches Graphiti audit §6.12 / TODO P-1.5 grain — encodes
    the dream pass id and the phase so ratification can find originating
    dream-write episodes by prefix-match.
    """
    return f"dream:{pass_id}:{phase}:{datetime.now(timezone.utc).isoformat()}"


def _episode_name(pass_id: str, phase: str, counter: int) -> str:
    """Stable, auditable episode name for dream-derived writes.

    Shares ``DREAM_EPISODE_NAME_PREFIX`` with the novelty check
    (:func:`fetch.is_dream_authored_episode`) so dream-authored episodes
    stay recognizable and never re-trigger a paid pass on themselves.
    """
    return f"{DREAM_EPISODE_NAME_PREFIX}{pass_id}_{phase}_{counter:03d}"


def _edge_metadata(envelope: MemoryEnvelope) -> dict:
    """Cypher-serializable MemoryFact attributes from a dream envelope.

    Stamped onto the edges the envelope's episode newly creates (see
    ``ingest._stamp_edge_metadata``) so dream provenance/status/source_kind
    land deterministically on the edge — graphiti's text-based attribute
    extraction can't recover them from the episode body. Enums are reduced
    to their string values; ``confidence``/``provenance`` may be None.
    """
    return {
        "status": envelope.status.value,
        "source_kind": envelope.source_kind.value,
        "scope": envelope.scope,
        "confidence": envelope.confidence,
        "provenance": envelope.provenance,
    }


class _Cited(BaseModel):
    """A pass's writes and proposals, each with the citations that name
    something the pass read in its scope (``citations.validated_citations``),
    by its index in the pass; ``uncited`` counts those left citing nothing,
    ``cross_scope`` the citations dropped for naming another scope."""

    writes: list[tuple[int, ConsolidatedFact, Citations]] = Field(default_factory=list)
    proposals: list[tuple[int, ProposedFinding, Citations]] = Field(
        default_factory=list
    )
    uncited: int = 0
    cross_scope: int = 0


def _cite(
    ops: DreamOperations,
    known_facts: Collection[str],
    known_episodes: Collection[str],
    source_scopes: Mapping[str, str],
) -> _Cited:
    """``ops``' writes and proposals checked against what the pass read."""
    check = partial(
        validated_citations,
        known_facts=known_facts,
        known_episodes=known_episodes,
        source_scopes=source_scopes,
    )
    items: list[ConsolidatedFact | ProposedFinding] = [*ops.writes, *ops.proposals]
    checked = [
        check(item.source_fact_uuids, item.source_episode_uuids, scope=item.scope)
        for item in items
    ]
    found = checked[: len(ops.writes)], checked[len(ops.writes) :]
    writes = [
        (i, fact, kept.citations)
        for i, (fact, kept) in enumerate(zip(ops.writes, found[0]))
        if kept.citations is not None
    ]
    proposals = [
        (i, finding, kept.citations)
        for i, (finding, kept) in enumerate(zip(ops.proposals, found[1]))
        if kept.citations is not None
    ]
    return _Cited(
        writes=writes,
        proposals=proposals,
        uncited=len(items) - len(writes) - len(proposals),
        cross_scope=sum(kept.cross_scope for kept in checked),
    )


async def _write_consolidated_fact(
    scope: MemoryScope,
    pass_id: str,
    counter: int,
    fact: ConsolidatedFact,
    session_id: str,
    completion: IngestionCompletion,
    *,
    citations: Citations,
) -> bool:
    """Queue one consolidated fact with what it cites: the worker checks the
    citations against forgets, then records them (``citations.py``)."""
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
        scope,
        session_id=session_id,
        name=_episode_name(pass_id, "consolidate", counter),
        episode_body=envelope.model_dump_json(),
        source_description=source_description("consolidation", citations),
        is_json=True,
        edge_metadata=_edge_metadata(envelope),
        completion=completion,
        citations=citations,
    )


async def _write_proposed_finding(
    scope: MemoryScope,
    pass_id: str,
    counter: int,
    finding: ProposedFinding,
    session_id: str,
    completion: IngestionCompletion,
    *,
    citations: Citations,
) -> bool:
    """Queue one proposal with what it cites, like a consolidated fact."""
    envelope = MemoryEnvelope(
        content=finding.content,
        source_kind=SourceKind.assistant_derived,
        memory_kind=finding.memory_kind,
        status=MemoryStatus.tentative,
        confidence=finding.confidence,
        scope=finding.scope,
        provenance=_provenance(pass_id, "recombine"),
    )
    return await enqueue_episode(
        scope,
        session_id=session_id,
        name=_episode_name(pass_id, "recombine", counter),
        episode_body=envelope.model_dump_json(),
        source_description=source_description(
            "proposal", citations, rationale=finding.rationale
        ),
        is_json=True,
        edge_metadata=_edge_metadata(envelope),
        completion=completion,
        citations=citations,
    )


def _summary(
    item: ConsolidatedFact | ProposedFinding,
    citations: Citations,
    status: Literal["active", "tentative"],
) -> WriteSummary:
    """The snapshot line for a queued write: what it said and what it cites."""
    return WriteSummary(
        content=item.content,
        scope=item.scope,
        confidence=item.confidence,
        status=status,
        source_episode_uuids=list(citations.episode_uuids),
        source_fact_uuids=list(citations.fact_uuids),
    )


async def _queue_cited(
    scope: MemoryScope,
    pass_id: str,
    cited: _Cited,
    session_id: str,
    completion: IngestionCompletion,
) -> tuple[list[WriteSummary], list[WriteSummary]]:
    """Queue the cited writes, then the cited proposals, registering each
    queued episode on ``completion``; the snapshot lines of those queued."""
    writes: list[WriteSummary] = []
    for i, fact, citations in cited.writes:
        if await _write_consolidated_fact(
            scope,
            pass_id,
            i,
            fact,
            session_id=session_id,
            completion=completion,
            citations=citations,
        ):
            completion.register()
            writes.append(_summary(fact, citations, "active"))
    proposals: list[WriteSummary] = []
    for i, finding, citations in cited.proposals:
        if await _write_proposed_finding(
            scope,
            pass_id,
            i,
            finding,
            session_id=session_id,
            completion=completion,
            citations=citations,
        ):
            completion.register()
            proposals.append(_summary(finding, citations, "tentative"))
    return writes, proposals


async def _create_dream_session(scope: MemoryScope, pass_id: str) -> str:
    """Create the dream-kind ChatSession shell and return its id.

    Written up front (before the memory ops) because the fact/proposal
    ``MemoryEnvelope`` provenance references this ``session_id``. The
    user-facing narrative is written separately, AFTER the ops are
    attempted (``_write_dream_summary_message``), so a partway failure leaves an
    empty dream rather than a 'completed' narrative with no memory.

    We use a fresh uuid rather than the pass_id so re-runs of the same
    pass (admin retries on failure) each produce their own session row.
    """
    # Lazy import — avoids circular dependency at module-import time
    # AND keeps the dream-pass / chat-model coupling explicit. Routing
    # through ``chat_db()`` means the dream pass (running in the
    # Scheduler subprocess) auto-uses the DatabaseManager RPC client;
    # the DatabaseManager process itself uses the direct module.
    from backend.api.features.orgs.db import get_user_default_team
    from backend.copilot.model import ChatSessionMetadata
    from backend.data.db_accessors import chat_db

    user_id = scope.owner_user_id
    # Dream passes run per-user with no request context; the user's
    # default (personal) org is the correct tenant for their dreams.
    try:
        org_id, team_id = await get_user_default_team(user_id)
    except Exception:
        logger.warning(
            f"Could not resolve default team for dream session (user {user_id}); "
            "creating tenant-less session"
        )
        org_id, team_id = None, None

    session_id = str(uuidlib.uuid4())
    # A dream is unattended, but the user reads and replies to it, and on a
    # self-hosted install the platform route may not exist at all — so it
    # starts on the same connection the user chose for everything else.
    llm_auth_provider, llm_credential_id = await resolve_default_chat_route(user_id)
    metadata = ChatSessionMetadata(
        kind="dream",
        dream_pass_id=pass_id,
        llm_auth_provider=llm_auth_provider,
        llm_credential_id=llm_credential_id,
    )
    await chat_db().create_chat_session(
        session_id=session_id,
        user_id=user_id,
        organization_id=org_id,
        team_id=team_id,
        metadata=metadata,
        expert_id=scope.expert_id,
    )
    # ``create_chat_session`` takes no title; set it via the dedicated
    # accessor so the session doesn't render as "(untitled)" in the chat
    # list. Best-effort: a cosmetic title failure must never abort apply —
    # on the batch path the at-most-once apply gate is already claimed by
    # the time we run, so an exception here would permanently lose the
    # dream (a retry hits the "duplicate" branch and skips apply).
    title = f"Dream summary — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    try:
        await chat_db().update_chat_session_title(
            session_id=session_id, user_id=user_id, title=title
        )
    except Exception:
        logger.warning(
            f"Failed to title dream session {session_id[:12]} "
            f"for user {user_id[:12]}",
            exc_info=True,
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


async def _drain_ingestion(
    pass_id: str, completion: IngestionCompletion, timeout_seconds: float
) -> IngestionDrainStatus:
    """Wait for the worker to finish the dream's OWN enqueued episodes:
    each written, dropped for resting on a forget, or failed.

    ``enqueue_episode`` returning True only proves the episode reached the
    in-process asyncio queue; the real write (LLM extraction + embedding in
    ``_ingestion_worker``) happens later. The caller of ``apply_operations``
    holds the dream lock until apply returns, so draining here lands the
    writes before the pass gives its lock back (while the lock the drain
    renewal proved lasts: ``LOCK_DRAIN_RENEWAL_SECONDS``) — without it, a
    scheduler pod restart silently discards queued writes while the pass
    stays recorded successful.

    Scoped to ``completion`` — only the episodes THIS pass enqueued. The
    per-user queue is shared with live-chat ingestion, so a whole-queue
    barrier would let a user's concurrent chat activity extend the in-lock
    hold up to the full timeout (and items enqueued after the drain starts
    would never let it resolve). Tracking the pass's own episodes makes the
    drain resolve the instant they are finished, regardless of other queue
    traffic.

    Returns:
      * ``drained`` — nothing was enqueued (vacuous) or the worker finished
        all of the pass's episodes within the timeout: written, dropped for
        resting on a forget, or failed with a logged error.
      * ``skipped`` — ``timeout_seconds <= 0``; the batch path uses this to
        avoid stalling the shared, serial ``BatchExecutor.walk_once`` loop
        (see ``BATCH_INGESTION_DRAIN_TIMEOUT_SECONDS``). The episodes still
        process fire-and-forget in the executor process.
      * ``timed_out`` — the worker did not finish all the episodes within
        the timeout.
        They keep processing fire-and-forget; apply warns rather than
        failing the pass — partial visibility beats a failed pass.
    """
    if not completion.registered:
        return IngestionDrainStatus.drained
    if timeout_seconds <= 0:
        logger.info(
            "Dream pass %s: ingestion drain skipped (no wait) — %d episode(s) "
            "processing fire-and-forget; reporting drain status=skipped",
            pass_id,
            completion.registered,
        )
        return IngestionDrainStatus.skipped
    drained = await wait_for_ingestion(completion, timeout_seconds)
    if drained:
        return IngestionDrainStatus.drained
    logger.warning(
        "Dream pass %s: own ingestion episodes did not drain within %.0fs — "
        "reported write/proposal counts include episodes still queued "
        "in-process (lost if this pod restarts)",
        pass_id,
        timeout_seconds,
    )
    return IngestionDrainStatus.timed_out


async def apply_operations(
    scope: MemoryScope,
    pass_id: str,
    ops: DreamOperations,
    *,
    known_fact_uuids: set[str] | None = None,
    known_episode_uuids: set[str] | None = None,
    source_scopes: Mapping[str, str] | None = None,
    ingestion_drain_timeout: float = INGESTION_DRAIN_TIMEOUT_SECONDS,
    lock_handle: DreamLockHandle | None = None,
    lease: ApplyLease | None = None,
) -> dict[str, int | str | IngestionDrainStatus | DreamOperationsSnapshot]:
    """Apply a sanitized DreamOperations to Graphiti + Postgres.

    Returns a small stats dict the orchestrator can fold into
    ``DreamPassResult``. Includes a ``snapshot`` key carrying the
    detailed ``DreamOperationsSnapshot`` payload for consumers that
    need per-operation rollups (eval, admin UI, future P9 SSE event),
    and an ``ingestion_drain_status`` (``IngestionDrainStatus``) —
    ``timed_out`` means the write/proposal counts were reported while
    episodes were still queued in-process, ``skipped`` is the by-design
    batch skip, ``drained`` is a pass whose writes the worker all finished
    (see ``_drain_ingestion``). Read it back via ``drain_status_from_stats``.

    An empty pass — no writes, proposals, demotions, or entity
    invalidations, once the writes citing nothing the pass read are
    dropped — returns zero counts (``uncited_writes_dropped`` aside) and an
    empty snapshot WITHOUT creating the dream session or writing any
    message; the ``session_id`` key is absent so
    ``apply_stats.get("session_id")`` reads as ``None`` for both the
    orchestrator and the batch callback. A pass WITH operations but an
    empty ``summary_for_user`` still creates the session and writes the
    fallback narrative (the ops were attempted; only the narrative is
    missing).

    ``known_fact_uuids`` is the set of edge uuids the dream pass
    actually fetched (``DreamInput.known_fact_uuids``); demotions
    targeting anything outside it are dropped before any Cypher runs
    (``demotions.py``). ``None`` means "look up the persisted input
    bundle by pass_id" for the demotions.

    Each write and proposal keeps the citations that name a fact in
    ``known_fact_uuids`` or an episode in ``known_episode_uuids`` whose scope
    (``source_scopes``, from ``citations.source_scopes``) is its own
    (``citations.py``); the others are counted in
    ``cross_scope_citations_dropped``, and one left citing nothing is dropped
    before it is queued and counted in ``uncited_writes_dropped``. Both
    routes pass their input bundle's sets and scopes; ``None`` reads as
    nothing read, so every write is dropped, and a source with no scope
    given is ``citations.UNSCOPED``. The ingestion worker drops, unwritten, any whose citations a
    forget reached after the pass read the graph
    (``graphiti/recall_citations.py``), and records the citations of every
    write it makes (``graphiti/recall_derivation.py``).
    ``dropped_forgotten`` counts those dropped before apply returned: all of
    them on a drained pass, possibly fewer when the drain was skipped or
    timed out.

    ``ingestion_drain_timeout`` bounds the in-line wait for the worker to
    finish the enqueued episodes (see ``_drain_ingestion``). The sync path keeps the
    full ``INGESTION_DRAIN_TIMEOUT_SECONDS``; the batch path passes
    ``BATCH_INGESTION_DRAIN_TIMEOUT_SECONDS`` (0) so it never stalls the
    shared, serial ``BatchExecutor.walk_once`` loop.

    ``lock_handle`` is the sync path's held dream lock. It is renewed to
    ``LOCK_DRAIN_RENEWAL_SECONDS`` right before the drain, failing closed,
    so a pass that lost its lock stops before the drain and the destructive
    writes after it. The renewal does not keep the lock through those
    writes: a drain and tail that outrun the TTL go on writing after it
    lapsed (``lease.py``). ``None`` on the batch path — it already disowned
    the lock to its callback with a 24h TTL.

    ``lease`` is the pass's lease, on either path: renewed once more right
    after the session shell is created and before the first graph write,
    failing closed (``DreamLockLostError``), so a lock that lapsed while the
    session was created, and maybe went to a newer pass, stops the writes.
    It narrows the window, it does not fence each write: a lease can still
    lapse between this renewal and the writes after it (``lease.py``).

    The demotions and entity invalidations (``demotions.py``) each carry the
    recall guard in their own statement: a live fact the user recalled within
    the protection window is left alone unless the write overrides it, and
    ``protected_demotions`` counts the distinct facts it kept live, as one
    read after every acknowledged write finds them. A write that raised (it
    may have committed or still be queued) is counted in
    ``indeterminate_demotion_writes``, not as a failure, and leaves
    ``demotion_accounting_complete`` False, as does a failed read.

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
    user_id = scope.owner_user_id
    cited = _cite(
        ops,
        known_fact_uuids or set(),
        known_episode_uuids or set(),
        source_scopes or {},
    )
    if cited.uncited or cited.cross_scope:
        logger.warning(
            f"Dream pass {pass_id} for user {user_id[:12]}: dropped "
            f"{cited.cross_scope} citation(s) of a source in another scope, "
            f"and {cited.uncited} write(s) and proposal(s) left citing nothing "
            "the pass read in their scope"
        )
    if not (
        cited.writes or cited.proposals or ops.demotions or ops.entity_invalidations
    ):
        # Empty pass — nothing landed in memory, so don't manufacture a
        # user-visible artifact for it. Creating the session shell +
        # placeholder narrative here is what produced one untitled empty
        # chat per user per night for users with old facts but no new
        # activity. ``session_id`` is deliberately absent from the stats:
        # consumers read it via ``.get("session_id")`` and both the
        # orchestrator and batch_callbacks treat the missing key as None.
        logger.info(
            f"Dream pass {pass_id} for user {user_id[:12]} produced "
            f"no operations — skipping dream session creation"
        )
        return {
            "consolidated_count": 0,
            "proposal_count": 0,
            "demotion_count": 0,
            "demotion_failed_count": 0,
            "entity_invalidation_count": 0,
            "dropped_forgotten": 0,
            "uncited_writes_dropped": cited.uncited,
            "cross_scope_citations_dropped": cited.cross_scope,
            "protected_demotions": 0,
            "indeterminate_demotion_writes": 0,
            "demotion_accounting_complete": True,
            # Vacuously drained — the pass enqueued nothing.
            "ingestion_drain_status": IngestionDrainStatus.drained,
            "snapshot": DreamOperationsSnapshot(),
        }

    # Phase A — create the session shell up front so the MemoryEnvelope
    # provenance can reference its id. The user-facing narrative summary
    # is written AFTER the ops (see below), so a partway failure leaves an
    # empty dream rather than a 'completed' narrative with no memory.
    session_id = await _create_dream_session(scope, pass_id)
    # Creating the session can take long enough for the lock to lapse: a pass
    # that cannot prove it still holds it writes nothing (the shell stays, as
    # after any partway failure).
    if lease is not None and not await lease.renew():
        raise DreamLockLostError(user_id)

    # Tracks completion of only the episodes THIS pass enqueues, so the
    # drain below waits on the dream's own writes and not on unrelated
    # live-chat ingestion sharing the same per-user queue. Registered once
    # per successful enqueue; the worker signals each as it finishes it.
    completion = IngestionCompletion()
    write_summaries, proposal_summaries = await _queue_cited(
        scope, pass_id, cited, session_id, completion
    )
    written, proposed = len(write_summaries), len(proposal_summaries)

    # One episode was registered per successful enqueue, so the tracker's
    # count is exactly the writes + proposals we report — the drain waits on
    # precisely those and nothing else. A mismatch does not endanger the
    # writes (the drain would just resolve early or wait out its cap), so
    # log loudly instead of failing a pass that has already written.
    if completion.registered != written + proposed:
        logger.error(
            "Dream pass %s: ingestion tracker registered %d episode(s) but "
            "reported %d write(s) + %d proposal(s) — drain barrier is not "
            "scoped to exactly the reported writes",
            pass_id,
            completion.registered,
            written,
            proposed,
        )

    # Renew the dream lock right before the longest non-LLM tail (the
    # ingestion drain plus the demotions / summary write that follow), to a
    # budget sized for that tail. The renewal proves ownership at the instant
    # it runs; it is a timer, not a fence, so a tail that outruns the TTL
    # writes on after the lock lapsed (``lease.py``). Gated on there being
    # any mutating work left — enqueued episodes to drain OR demotions /
    # entity invalidations to apply, the most destructive ops in the pass.
    # The batch path passes no handle (it disowned the lock to its callback).
    # A failed renewal means the lock already expired — a newer pass may own
    # the graph — so abort before the drain and the destructive writes below.
    # The episodes already enqueued above keep processing fire-and-forget
    # (they cannot be recalled), but the pass is reported errored instead of
    # pretending exclusive ownership.
    if lock_handle is not None and (
        completion.registered or ops.demotions or ops.entity_invalidations
    ):
        if not await lock_handle.extend(LOCK_DRAIN_RENEWAL_SECONDS):
            raise DreamLockLostError(user_id)

    # Drain the in-process ingestion queue before anything downstream
    # treats the writes as finished (and before we return and the caller
    # releases the dream lock). See ``_drain_ingestion``.
    ingestion_drain_status = await _drain_ingestion(
        pass_id, completion, ingestion_drain_timeout
    )

    destroyed = await apply_demotions(scope, pass_id, ops, known_fact_uuids)

    # Narrative summary last — only surface the user-facing dream story
    # once the memory ops above have been attempted.
    await _write_dream_summary_message(session_id, pass_id, ops.summary_for_user)

    logger.info(
        "Dream pass %s applied for user %s: "
        "writes=%d proposals=%d uncited=%d dropped_forgotten=%d demoted=%d "
        "(failed=%d) protected=%d entity_edges=%d indeterminate=%d "
        "accounting_complete=%s ingestion_drain_status=%s",
        pass_id,
        user_id[:12],
        written,
        proposed,
        cited.uncited,
        completion.dropped_forgotten,
        destroyed.demoted,
        destroyed.failed,
        destroyed.protected,
        destroyed.entity_edges,
        destroyed.indeterminate,
        destroyed.accounting_complete,
        ingestion_drain_status.value,
    )

    snapshot = DreamOperationsSnapshot(
        writes=write_summaries,
        proposals=proposal_summaries,
        demotions=destroyed.demotions,
        entity_invalidations=destroyed.entity_invalidations,
    )

    return {
        "session_id": session_id,
        "consolidated_count": written,
        "proposal_count": proposed,
        "demotion_count": destroyed.demoted,
        "demotion_failed_count": destroyed.failed,
        "entity_invalidation_count": destroyed.entity_edges,
        # Writes and proposals dropped unwritten: a forget reached what they
        # rest on after the pass read the graph.
        "dropped_forgotten": completion.dropped_forgotten,
        # Writes and proposals dropped before they were queued: they cited
        # nothing the pass read in their scope.
        "uncited_writes_dropped": cited.uncited,
        # Citations dropped for naming a source in another scope.
        "cross_scope_citations_dropped": cited.cross_scope,
        # Distinct facts an acknowledged write spared and the accounting read
        # found live; provisional when the accounting is incomplete.
        "protected_demotions": destroyed.protected,
        # Destructive writes that raised: each may have committed or still land.
        "indeterminate_demotion_writes": destroyed.indeterminate,
        # False when the final read failed or any write's outcome is unknown.
        "demotion_accounting_complete": destroyed.accounting_complete,
        "ingestion_drain_status": ingestion_drain_status,
        "snapshot": snapshot,
    }
