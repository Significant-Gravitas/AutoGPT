"""Apply tests — mock the Graphiti/Postgres boundary, verify the right
calls are produced for each operation type.

These tests do NOT touch FalkorDB or Prisma. apply.py is a pure
fan-out: it builds MemoryEnvelopes for writes/proposals and delegates
to ``enqueue_episode`` / ``create_chat_session`` / ``add_chat_message``
and, through its destructive stage (``demotions.py``), to
``supersede_unless_recalled`` / ``invalidate_entity_direct_neighbors``.
Each of those is mocked here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.guarded_writes import NeighbourWrites, WriteOutcome
from backend.copilot.graphiti.ingest import IngestionCompletion
from backend.copilot.graphiti.recall_citations import Citations
from backend.copilot.graphiti.scope import MemoryScope

from . import apply as apply_mod
from . import demotions as demotions_mod
from .demotions import DemotionResults
from .fetch import DreamInput
from .locks import DreamLockLostError
from .schemas import (
    ConsolidatedFact,
    DreamDemotion,
    DreamOperations,
    DreamOperationsSnapshot,
    EntityInvalidation,
    IngestionDrainStatus,
    ProposedFinding,
)

# What these passes read. Every write and proposal below cites some of it:
# one citing nothing the pass read is dropped (``citations.py``).
_READ_FACT = "f1"
_READ_EPISODE = "ep-1"
_READ = {"known_fact_uuids": {_READ_FACT}, "known_episode_uuids": {_READ_EPISODE}}
_WRITE = ConsolidatedFact(
    content="A likes B", confidence=0.8, source_episode_uuids=[_READ_EPISODE]
)


def _bundle_with_known_facts(*uuids: str) -> DreamInput:
    return DreamInput(
        user_id="u-bundle",
        group_id="u-bundle",
        window_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 14, tzinfo=timezone.utc),
        known_fact_uuids=set(uuids),
    )


@pytest.fixture(autouse=True)
def _stub_boundaries(mocker):
    """Wire up the apply.py side-effects with AsyncMocks once per test."""
    mocker.patch.object(apply_mod, "enqueue_episode", AsyncMock(return_value=True))
    # enqueue_episode is stubbed True above, so every write/proposal registers
    # on the pass's IngestionCompletion — but no worker runs in tests, so an
    # unstubbed drain would block the full INGESTION_DRAIN_TIMEOUT_SECONDS
    # (300s) per test and time out the CI job. Default the drain to an
    # instant success; drain-behavior tests re-patch this explicitly.
    mocker.patch.object(apply_mod, "wait_for_ingestion", AsyncMock(return_value=True))
    # The driver factory + close — the destructive stage (demotions.py) opens
    # a FalkorDB driver for demotions and entity invalidations. Patch where
    # it's used.
    driver = mocker.MagicMock()
    driver.close = AsyncMock(return_value=None)
    mocker.patch.object(
        demotions_mod, "open_driver", mocker.MagicMock(return_value=driver)
    )
    # The guarded writers: every demotion lands, every invalidation touches
    # two edges, nothing is spared unless a test says otherwise.
    mocker.patch.object(
        demotions_mod,
        "supersede_unless_recalled",
        AsyncMock(
            side_effect=lambda driver, uuids, **kw: [WriteOutcome.CHANGED] * len(uuids)
        ),
    )
    mocker.patch.object(
        demotions_mod,
        "invalidate_entity_direct_neighbors",
        AsyncMock(return_value=NeighbourWrites(changed=["e1", "e2"])),
    )
    # The read that settles the protected count: every spared fact is still
    # live unless a test says otherwise.
    mocker.patch.object(
        demotions_mod,
        "live_fact_uuids",
        AsyncMock(side_effect=lambda driver, group_id, uuids: set(uuids)),
    )
    # ChatSession + ChatMessage writes — apply.py imports them lazily inside
    # ``_create_dream_session`` / ``_write_dream_summary_message`` to avoid a
    # circular import. Patch where the symbol is looked up (copilot.db).
    database = mocker.MagicMock()
    database.create_chat_session = AsyncMock(
        return_value=mocker.MagicMock(session_id="s1")
    )
    database.update_chat_session_title = AsyncMock(return_value=True)
    database.add_chat_message = AsyncMock(return_value=None)
    mocker.patch("backend.data.db_accessors.chat_db", return_value=database)
    mocker.patch("backend.copilot.db.create_chat_session", database.create_chat_session)
    mocker.patch(
        "backend.copilot.db.update_chat_session_title",
        database.update_chat_session_title,
    )
    mocker.patch("backend.copilot.db.add_chat_message", database.add_chat_message)
    # _create_dream_session's tenant lookup. Unmocked it runs REAL Prisma
    # queries on this test's function-scoped event loop whenever an earlier
    # test already connected Prisma (its except swallows the failure when
    # not connected, so the leak is invisible locally). Those connections
    # stay in the shared Prisma httpx pool bound to a dead loop and the
    # next session-loop test touching Prisma dies with "Event loop is
    # closed" (test_chatsession_redis_storage in CI).
    mocker.patch(
        "backend.api.features.orgs.db.get_user_default_team",
        AsyncMock(return_value=(None, None)),
    )
    # Entity invalidation is gated on DREAM_PASS_INVALIDATE_ENTITY. Default
    # the flag ON so the existing entity tests exercise the apply path; the
    # flag-off behavior has its own dedicated test below.
    mocker.patch.object(
        demotions_mod, "is_feature_enabled", AsyncMock(return_value=True)
    )
    # No persisted input bundle by default — the demotion pre-flight filter
    # fails open (keeps all demotions) so tests that don't care about uuid
    # validation behave as before. Filter tests re-patch with a bundle.
    mocker.patch.object(
        demotions_mod, "read_input_bundle", AsyncMock(return_value=None)
    )
    # derive_group_id is deterministic; let it run.


@pytest.mark.asyncio
async def test_writes_become_active_envelopes():
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(
                content="A likes B",
                confidence=0.8,
                scope="real:global",
                source_episode_uuids=[_READ_EPISODE],
            )
        ],
        summary_for_user="ok",
    )
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-1234567890ab"), pass_id="p-abc", ops=ops, **_READ
    )

    assert stats["consolidated_count"] == 1
    assert stats["proposal_count"] == 0
    apply_mod.enqueue_episode.assert_awaited()
    call_kwargs = apply_mod.enqueue_episode.await_args.kwargs
    assert call_kwargs["is_json"] is True
    assert call_kwargs["name"].startswith("dream_p-abc_consolidate_")
    # The envelope body should be JSON with status=active source_kind=assistant_derived
    body = call_kwargs["episode_body"]
    assert '"status":"active"' in body
    assert '"source_kind":"assistant_derived"' in body
    # #13389: edge_metadata is threaded so the write's envelope fields land
    # ON the edge (not just in the episode body). Active consolidated fact.
    em = call_kwargs["edge_metadata"]
    assert em["status"] == "active"
    assert em["source_kind"] == "assistant_derived"
    assert em["provenance"].startswith("dream:p-abc:consolidate")


@pytest.mark.asyncio
async def test_proposals_become_tentative_envelopes():
    ops = DreamOperations(
        proposals=[
            ProposedFinding(
                content="A trusts B",
                confidence=0.6,
                rationale="implied",
                source_fact_uuids=["f1"],
            )
        ],
        summary_for_user="ok",
    )
    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-x"), pass_id="p-2", ops=ops, **_READ
    )
    # First call was the consolidate write (there were no writes, so first call IS the proposal)
    call_kwargs = apply_mod.enqueue_episode.await_args.kwargs
    assert call_kwargs["name"].startswith("dream_p-2_recombine_")
    body = call_kwargs["episode_body"]
    assert '"status":"tentative"' in body
    # #13389: a proposal rides edge_metadata as tentative so ratification
    # can find it on the edge.
    em = call_kwargs["edge_metadata"]
    assert em["status"] == "tentative"
    assert em["source_kind"] == "assistant_derived"
    assert em["provenance"].startswith("dream:p-2:recombine")


@pytest.mark.asyncio
async def test_demotions_group_by_status_and_reason():
    """Bucketed guarded writes — one per (status, reason) pair."""
    ops = DreamOperations(
        demotions=[
            DreamDemotion(edge_uuid="a", reason="stale", new_status="superseded"),
            DreamDemotion(edge_uuid="b", reason="stale", new_status="superseded"),
            DreamDemotion(
                edge_uuid="c", reason="contradicted_by:x", new_status="contradicted"
            ),
        ],
    )
    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-y"), pass_id="p-3", ops=ops
    )

    # Three demotions but only TWO buckets: (superseded, stale) and
    # (contradicted, contradicted_by:x)
    assert demotions_mod.supersede_unless_recalled.await_count == 2
    bucket_args = [
        call.args[1] if len(call.args) > 1 else call.kwargs.get("uuids")
        for call in demotions_mod.supersede_unless_recalled.await_args_list
    ]
    # One bucket has 2 uuids, the other has 1
    assert sorted(len(b) for b in bucket_args) == [1, 2]


@pytest.mark.asyncio
async def test_demotions_pass_group_id_to_the_guarded_write():
    """The Cypher group_id predicate (defense-in-depth against a
    wrong-driver caller) only works if the stage threads the derived
    group_id into every guarded write."""
    ops = DreamOperations(
        demotions=[DreamDemotion(edge_uuid="a", reason="stale")],
    )
    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-gid"), pass_id="p-gid", ops=ops
    )

    demotions_mod.supersede_unless_recalled.assert_awaited_once()
    # derive_group_id prefixes user ids with "user_"
    assert (
        demotions_mod.supersede_unless_recalled.await_args.kwargs["group_id"]
        == "user_u-gid"
    )


@pytest.mark.asyncio
async def test_hallucinated_demotion_uuids_dropped_before_cypher():
    """Sync path: demotions targeting edge uuids outside the pass's
    known_fact_uuids are a prompt-constraint violation (hallucination or
    injection) and must never reach the guarded write."""
    ops = DreamOperations(
        demotions=[
            DreamDemotion(edge_uuid="known-1", reason="stale"),
            DreamDemotion(edge_uuid="hallucinated", reason="stale"),
        ],
    )
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-filter"),
        pass_id="p-filter",
        ops=ops,
        known_fact_uuids={"known-1", "known-2"},
    )

    demotions_mod.supersede_unless_recalled.assert_awaited_once()
    sent_uuids = demotions_mod.supersede_unless_recalled.await_args.args[1]
    assert sent_uuids == ["known-1"]
    # The rejected demotion never reaches the snapshot either
    assert [d.edge_uuid for d in stats["snapshot"].demotions] == ["known-1"]
    # The caller supplied the allowlist — no Redis bundle lookup needed
    demotions_mod.read_input_bundle.assert_not_awaited()


@pytest.mark.asyncio
async def test_demotions_without_known_facts_validated_against_persisted_bundle(
    mocker,
):
    """Both routes pass known_fact_uuids from their DreamInput; a caller that
    passes none has the filter fall back to the input bundle persisted at
    submit time."""
    mocker.patch.object(
        demotions_mod,
        "read_input_bundle",
        AsyncMock(return_value=_bundle_with_known_facts("known-1")),
    )
    ops = DreamOperations(
        demotions=[
            DreamDemotion(edge_uuid="known-1", reason="stale"),
            DreamDemotion(edge_uuid="ghost", reason="stale"),
        ],
    )
    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-batch"), pass_id="p-batch", ops=ops
    )

    demotions_mod.read_input_bundle.assert_awaited_once_with("p-batch")
    demotions_mod.supersede_unless_recalled.assert_awaited_once()
    assert demotions_mod.supersede_unless_recalled.await_args.args[1] == ["known-1"]


@pytest.mark.asyncio
async def test_missing_input_bundle_fails_open_and_keeps_demotions():
    """When neither the caller nor Redis can supply known_fact_uuids
    (bundle expired/corrupted), the filter fails open — demotions are
    kept rather than zeroing the pass, matching the clamp's
    unknown-fact-count posture. The autouse fixture's
    read_input_bundle stub returns None."""
    ops = DreamOperations(
        demotions=[DreamDemotion(edge_uuid="unverifiable", reason="stale")],
    )
    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-open"), pass_id="p-open", ops=ops
    )

    demotions_mod.supersede_unless_recalled.assert_awaited_once()
    assert demotions_mod.supersede_unless_recalled.await_args.args[1] == [
        "unverifiable"
    ]


@pytest.mark.asyncio
async def test_redis_blip_on_bundle_fallback_fails_open(mocker, caplog):
    """A Redis error during the input-bundle fallback read must take the
    same fail-open branch as a missing bundle (keep demotions, WARNING)
    instead of raising out of apply_operations — on the batch path the
    at-most-once apply gate is already claimed by the time apply runs,
    so an exception here permanently loses the dream (a retry hits the
    "duplicate" branch and skips apply entirely)."""
    mocker.patch.object(
        demotions_mod,
        "read_input_bundle",
        AsyncMock(side_effect=ConnectionError("redis blip")),
    )
    ops = DreamOperations(
        demotions=[DreamDemotion(edge_uuid="unverifiable", reason="stale")],
    )
    with caplog.at_level(logging.WARNING, logger=demotions_mod.logger.name):
        stats = await apply_mod.apply_operations(
            scope=MemoryScope.for_user("u-blip"), pass_id="p-blip", ops=ops
        )

    demotions_mod.supersede_unless_recalled.assert_awaited_once()
    assert demotions_mod.supersede_unless_recalled.await_args.args[1] == [
        "unverifiable"
    ]
    assert stats["demotion_count"] == 1
    assert any(
        "input bundle read failed" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


@pytest.mark.asyncio
async def test_entity_invalidations_not_filtered_by_known_fact_uuids():
    """The input bundle carries no entity-uuid allowlist (FactRow
    source/target are entity names), so entity invalidations are NOT
    subject to the known-fact pre-flight — they rely on the LD flag +
    count clamp + single-hop guarantee instead."""
    ops = DreamOperations(
        entity_invalidations=[
            EntityInvalidation(entity_uuid="ent-unlisted", reason="r"),
        ],
    )
    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-ent"),
        pass_id="p-ent",
        ops=ops,
        known_fact_uuids={"some-fact"},
    )

    demotions_mod.invalidate_entity_direct_neighbors.assert_awaited_once()


@pytest.mark.asyncio
async def test_entity_invalidation_calls_single_hop_helper():
    ops = DreamOperations(
        entity_invalidations=[
            EntityInvalidation(entity_uuid="ent-x", reason="dead_to_us"),
        ],
    )
    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-z"), pass_id="p-4", ops=ops
    )

    demotions_mod.invalidate_entity_direct_neighbors.assert_awaited_once()
    kwargs = demotions_mod.invalidate_entity_direct_neighbors.await_args.kwargs
    assert kwargs["entity_uuid"] == "ent-x"
    assert kwargs["reason"] == "dead_to_us"


@pytest.mark.asyncio
async def test_entity_invalidation_skipped_when_flag_off(mocker):
    """With DREAM_PASS_INVALIDATE_ENTITY off, proposed invalidations are
    dropped — the destructive single-hop helper must never run and the
    snapshot reflects zero entity edges touched."""
    mocker.patch.object(
        demotions_mod, "is_feature_enabled", AsyncMock(return_value=False)
    )
    ops = DreamOperations(
        entity_invalidations=[
            EntityInvalidation(entity_uuid="ent-x", reason="dead_to_us"),
        ],
    )
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-z"), pass_id="p-off", ops=ops
    )

    demotions_mod.invalidate_entity_direct_neighbors.assert_not_awaited()
    assert stats["entity_invalidation_count"] == 0
    assert stats["snapshot"].entity_invalidations == []


@pytest.mark.asyncio
async def test_empty_pass_creates_no_session_and_no_message():
    """A pass with no writes, proposals, demotions, or entity
    invalidations must not manufacture a user-visible chat — no
    dream-kind ChatSession, no placeholder message, zero stats, and no
    session_id. A non-empty summary alone is NOT an operation.
    Regression: nightly dreams for users with old facts but no new
    activity created one untitled empty chat per user per night."""
    from backend.copilot import db as copilot_db

    from .schemas import DreamOperationsSnapshot

    ops = DreamOperations(summary_for_user="Nothing new today.")
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-a"), pass_id="p-5", ops=ops
    )

    copilot_db.create_chat_session.assert_not_awaited()
    copilot_db.add_chat_message.assert_not_awaited()
    apply_mod.enqueue_episode.assert_not_awaited()
    assert stats.get("session_id") is None
    assert stats["consolidated_count"] == 0
    assert stats["proposal_count"] == 0
    assert stats["demotion_count"] == 0
    assert stats["demotion_failed_count"] == 0
    assert stats["entity_invalidation_count"] == 0
    assert stats["snapshot"] == DreamOperationsSnapshot()


@pytest.mark.asyncio
async def test_ops_with_empty_summary_still_create_session_with_placeholder():
    """Operations landed but the model returned no narrative — the
    session must still be created (the memory ops need their provenance
    + user-visible record) with the fallback placeholder message."""
    from backend.copilot import db as copilot_db

    ops = DreamOperations(
        writes=[_WRITE],
        summary_for_user="",
    )
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-ph"), pass_id="p-ph", ops=ops, **_READ
    )

    copilot_db.create_chat_session.assert_awaited_once()
    copilot_db.add_chat_message.assert_awaited_once()
    msg_kwargs = copilot_db.add_chat_message.await_args.kwargs
    assert msg_kwargs["role"] == "assistant"
    assert msg_kwargs["content"] == "Dream pass completed with no narrative output."
    assert isinstance(stats["session_id"], str) and stats["session_id"]


@pytest.mark.asyncio
async def test_dream_session_titled_with_utc_date():
    """The dream-kind session gets a 'Dream summary — YYYY-MM-DD' title
    (UTC date) via update_chat_session_title, scoped to the owning user,
    so it doesn't render as '(untitled)' in the chat list."""
    from backend.copilot import db as copilot_db

    ops = DreamOperations(
        writes=[_WRITE],
        summary_for_user="ok",
    )
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-title"), pass_id="p-title", ops=ops, **_READ
    )

    copilot_db.update_chat_session_title.assert_awaited_once()
    title_kwargs = copilot_db.update_chat_session_title.await_args.kwargs
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert title_kwargs["title"] == f"Dream summary — {today}"
    assert title_kwargs["session_id"] == stats["session_id"]
    assert title_kwargs["user_id"] == "u-title"


@pytest.mark.asyncio
async def test_title_failure_does_not_abort_apply(mocker):
    """The title write is cosmetic and best-effort — on the batch path the
    at-most-once apply gate is already claimed when apply runs, so an
    exception here would permanently lose the dream. The ops and the
    narrative must still land."""
    from backend.copilot import db as copilot_db

    mocker.patch(
        "backend.copilot.db.update_chat_session_title",
        AsyncMock(side_effect=ConnectionError("db blip")),
    )
    ops = DreamOperations(
        writes=[_WRITE],
        summary_for_user="ok",
    )
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-tf"), pass_id="p-tf", ops=ops, **_READ
    )

    assert stats["consolidated_count"] == 1
    copilot_db.add_chat_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_expert_dream_session_and_write_keep_expert_scope(mocker):
    db = mocker.MagicMock()
    db.create_chat_session = AsyncMock()
    db.update_chat_session_title = AsyncMock(return_value=True)
    mocker.patch("backend.data.db_accessors.chat_db", return_value=db)

    expert_scope = MemoryScope.for_expert("u1", "expert-1")
    await apply_mod._create_dream_session(expert_scope, "p1")
    assert db.create_chat_session.call_args.kwargs["expert_id"] == "expert-1"

    await apply_mod._write_consolidated_fact(
        expert_scope,
        "p1",
        0,
        _WRITE,
        "session-1",
        IngestionCompletion(),
        citations=Citations(episode_uuids=[_READ_EPISODE]),
    )
    assert apply_mod.enqueue_episode.call_args.args[0] == expert_scope


@pytest.mark.asyncio
async def test_summary_written_after_memory_ops(mocker):
    """The user-facing narrative must be written AFTER the memory ops, so a
    partway failure doesn't leave a 'completed' dream narrative with no
    memory behind it."""
    calls: list[str] = []

    async def _track_write(*args, **kwargs):
        calls.append("write")
        return True

    async def _track_summary(*args, **kwargs):
        calls.append("summary")

    mocker.patch.object(
        apply_mod, "_create_dream_session", new_callable=AsyncMock, return_value="s"
    )
    mocker.patch.object(apply_mod, "_write_consolidated_fact", side_effect=_track_write)
    mocker.patch.object(
        apply_mod, "_write_dream_summary_message", side_effect=_track_summary
    )
    mocker.patch.object(
        apply_mod, "apply_demotions", AsyncMock(return_value=DemotionResults())
    )

    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-1"),
        pass_id="p-1",
        ops=DreamOperations(
            writes=[_WRITE],
            summary_for_user="done",
        ),
        **_READ,
    )

    assert calls == ["write", "summary"], calls


# ---------------------------------------------------------------------------
# Ingestion drain — enqueue_episode returning True only proves the episode
# reached the in-process asyncio queue; the real graph write (LLM extraction
# + embedding in _ingestion_worker) happens later. apply_operations must
# await the queue drain BEFORE returning, because the caller holds the dream
# lock only until apply returns — otherwise a scheduler pod restart silently
# discards queued writes while the pass stays recorded successful.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_waits_for_ingestion_drain_before_reporting_counts(mocker):
    """With writes enqueued, apply must await wait_for_ingestion (scoped to
    the pass's own episodes, bounded by the drain timeout) and report
    ingestion_drain_status=drained on success."""
    drain = mocker.patch.object(
        apply_mod, "wait_for_ingestion", AsyncMock(return_value=True)
    )
    ops = DreamOperations(
        writes=[_WRITE],
        summary_for_user="ok",
    )
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-drain"), pass_id="p-drain", ops=ops, **_READ
    )

    drain.assert_awaited_once()
    # Waited on the pass's own completion tracker, not the whole queue, with
    # the sync drain timeout.
    completion, timeout = drain.await_args.args
    assert isinstance(completion, IngestionCompletion)
    assert completion.registered == 1
    assert timeout == apply_mod.INGESTION_DRAIN_TIMEOUT_SECONDS
    assert stats["ingestion_drain_status"] is IngestionDrainStatus.drained


@pytest.mark.asyncio
async def test_drain_timeout_reports_partial_visibility_not_failure(mocker, caplog):
    """When the pass's episodes don't drain inside the cap, the pass must
    still succeed (partial visibility beats a failed pass): counts are
    returned, ingestion_drain_status=timed_out flags the overflow, and a
    WARNING records the revert to fire-and-forget behavior."""
    mocker.patch.object(apply_mod, "wait_for_ingestion", AsyncMock(return_value=False))
    ops = DreamOperations(
        writes=[_WRITE],
        summary_for_user="ok",
    )
    with caplog.at_level(logging.WARNING, logger=apply_mod.logger.name):
        stats = await apply_mod.apply_operations(
            scope=MemoryScope.for_user("u-slow"), pass_id="p-slow", ops=ops, **_READ
        )

    assert stats["ingestion_drain_status"] is IngestionDrainStatus.timed_out
    assert stats["consolidated_count"] == 1
    assert any(
        "did not drain" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


@pytest.mark.asyncio
async def test_zero_drain_timeout_skips_wait_and_reports_skipped(mocker, caplog):
    """The batch path passes ``ingestion_drain_timeout=0`` so apply never
    stalls the shared, serial BatchExecutor.walk_once loop: even with writes
    enqueued, wait_for_ingestion is NOT awaited, and the pass reports
    ingestion_drain_status=skipped (a by-design skip, not a failure)."""
    drain = mocker.patch.object(
        apply_mod, "wait_for_ingestion", AsyncMock(return_value=True)
    )
    ops = DreamOperations(
        writes=[_WRITE],
        summary_for_user="ok",
    )
    with caplog.at_level(logging.INFO, logger=apply_mod.logger.name):
        stats = await apply_mod.apply_operations(
            scope=MemoryScope.for_user("u-batch"),
            pass_id="p-batch",
            ops=ops,
            **_READ,
            ingestion_drain_timeout=apply_mod.BATCH_INGESTION_DRAIN_TIMEOUT_SECONDS,
        )

    drain.assert_not_awaited()
    assert stats["ingestion_drain_status"] is IngestionDrainStatus.skipped
    assert stats["consolidated_count"] == 1
    assert any("drain skipped" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_no_enqueued_writes_skips_ingestion_drain(mocker):
    """A dream with no writes/proposals must not block on the shared
    per-user queue (live chat episodes could be in flight) — the drain is
    skipped and reported vacuously drained."""
    drain = mocker.patch.object(
        apply_mod, "wait_for_ingestion", AsyncMock(return_value=True)
    )
    ops = DreamOperations(summary_for_user="Nothing new today.")
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-empty"), pass_id="p-empty", ops=ops
    )

    drain.assert_not_awaited()
    assert stats["ingestion_drain_status"] is IngestionDrainStatus.drained


@pytest.mark.asyncio
async def test_sync_path_renews_lock_before_drain(mocker):
    """The dream lock is renewed to a fresh budget right before the drain so
    it cannot expire while the pass's writes are still landing (which would
    admit a concurrent pass onto the same graph)."""
    mocker.patch.object(apply_mod, "wait_for_ingestion", AsyncMock(return_value=True))
    lock_handle = mocker.MagicMock()
    lock_handle.extend = AsyncMock(return_value=True)
    ops = DreamOperations(
        writes=[_WRITE],
        summary_for_user="ok",
    )
    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-lock"),
        pass_id="p-lock",
        ops=ops,
        **_READ,
        lock_handle=lock_handle,
    )

    lock_handle.extend.assert_awaited_once_with(apply_mod.LOCK_DRAIN_RENEWAL_SECONDS)


@pytest.mark.asyncio
async def test_failed_lock_renewal_aborts_before_drain_and_demotions(mocker):
    """A failed renewal means the lock expired (a newer pass may own the
    graph), so apply must abort before the drain and the destructive
    demotions/summary writes instead of continuing as if it held exclusive
    ownership. The orchestrator's catch-all turns the raise into an errored
    ``DreamPassResult``."""
    drain = mocker.patch.object(apply_mod, "_drain_ingestion", AsyncMock())
    demote = mocker.patch.object(apply_mod, "apply_demotions", AsyncMock())
    lock_handle = mocker.MagicMock()
    lock_handle.extend = AsyncMock(return_value=False)
    ops = DreamOperations(
        writes=[_WRITE],
        summary_for_user="ok",
    )

    with pytest.raises(DreamLockLostError):
        await apply_mod.apply_operations(
            scope=MemoryScope.for_user("u-lost"),
            pass_id="p-lost",
            ops=ops,
            **_READ,
            lock_handle=lock_handle,
        )

    drain.assert_not_awaited()
    demote.assert_not_awaited()


@pytest.mark.asyncio
async def test_demotions_only_pass_renews_lock_before_destructive_tail(mocker):
    """Demotions and entity invalidations are the most destructive ops in a
    pass, and a sanitizer output can contain them with zero writes. Such a
    pass registers no episodes, so the renewal must not be gated on the
    ingestion tracker alone or it would run under a near-exhausted TTL."""
    lock_handle = mocker.MagicMock()
    lock_handle.extend = AsyncMock(return_value=True)
    ops = DreamOperations(
        demotions=[DreamDemotion(edge_uuid="a", reason="stale")],
        entity_invalidations=[EntityInvalidation(entity_uuid="ent-x", reason="gone")],
        summary_for_user="tidied up",
    )
    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-demote"),
        pass_id="p-demote",
        ops=ops,
        lock_handle=lock_handle,
    )

    lock_handle.extend.assert_awaited_once_with(apply_mod.LOCK_DRAIN_RENEWAL_SECONDS)


@pytest.mark.asyncio
async def test_empty_pass_does_not_renew_lock(mocker):
    """A pass with nothing to drain and nothing to demote needn't touch the
    lock TTL."""
    mocker.patch.object(apply_mod, "wait_for_ingestion", AsyncMock(return_value=True))
    lock_handle = mocker.MagicMock()
    lock_handle.extend = AsyncMock(return_value=None)
    ops = DreamOperations(summary_for_user="Nothing new today.")
    await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-empty"),
        pass_id="p-empty",
        ops=ops,
        lock_handle=lock_handle,
    )

    lock_handle.extend.assert_not_awaited()


# ---------------------------------------------------------------------------
# drain_status_from_stats — the single read point for the drain outcome,
# shared by the sync orchestrator and the batch callback. Deserialized or
# partial stats must never read as a confirmed drain.
# ---------------------------------------------------------------------------


def test_drain_status_from_stats_passes_through_enum():
    assert (
        apply_mod.drain_status_from_stats(
            {"ingestion_drain_status": IngestionDrainStatus.skipped}
        )
        is IngestionDrainStatus.skipped
    )


def test_drain_status_from_stats_coerces_valid_string():
    """Stats that round-tripped through JSON carry the enum's string value."""
    assert (
        apply_mod.drain_status_from_stats({"ingestion_drain_status": "drained"})
        is IngestionDrainStatus.drained
    )


@pytest.mark.parametrize(
    "raw",
    ["", "DRAINED", "nonsense", 1, True, None, ["drained"]],
)
def test_drain_status_from_stats_fails_closed(raw):
    """Malformed, wrong-typed and missing values all read as ``timed_out`` —
    lost observability must not present as a confirmed drain."""
    assert (
        apply_mod.drain_status_from_stats({"ingestion_drain_status": raw})
        is IngestionDrainStatus.timed_out
    )


def test_drain_status_from_stats_missing_key_fails_closed():
    assert apply_mod.drain_status_from_stats({}) is IngestionDrainStatus.timed_out


# ---------------------------------------------------------------------------
# Prisma auto-connect regression (scheduler service starts without an open
# Prisma connection; apply_operations must open one before any DB writes).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_operations_never_auto_connects_prisma(mocker):
    """``apply_operations`` MUST NOT call ``platform_db.connect()``.

    The dream pass runs in the Scheduler subprocess where Prisma is
    intentionally left disconnected so callers route through
    ``chat_db()`` / equivalents (which transparently use the
    DatabaseManager RPC client). Auto-connecting here flips
    ``is_connected()`` to True before the local Prisma engine is
    reachable, racing with concurrent ``platform_cost_db()`` callers
    from ``token_tracking._safe_log`` — they see
    ``is_connected=True``, try the direct Prisma path, and hit
    "All connection attempts failed" while the engine is still
    booting. Regression pin: keep the auto-connect OUT.
    """
    from backend.copilot.dream import apply as apply_mod
    from backend.copilot.dream.schemas import DreamOperations

    mocker.patch.object(
        apply_mod, "_create_dream_session", new_callable=AsyncMock, return_value="s"
    )
    mocker.patch.object(
        apply_mod, "_write_dream_summary_message", new_callable=AsyncMock
    )
    mocker.patch.object(
        apply_mod, "apply_demotions", AsyncMock(return_value=DemotionResults())
    )

    # Whatever state Prisma is in, apply_operations must not touch
    # ``platform_db.connect``. Spy on BOTH states to make the contract
    # explicit.
    for is_conn in (False, True):
        mocker.patch("backend.data.db.is_connected", return_value=is_conn)
        connect_spy = mocker.patch("backend.data.db.connect", new_callable=AsyncMock)

        await apply_mod.apply_operations(
            scope=MemoryScope.for_user("u-1"),
            pass_id="p-1",
            ops=DreamOperations(
                writes=[],
                proposals=[],
                demotions=[],
                entity_invalidations=[],
                summary_for_user="empty",
            ),
        )
        connect_spy.assert_not_called()


# ---------------------------------------------------------------------------
# DreamOperationsSnapshot — eval/UI/SSE consumers need per-operation detail.
# Tested at the apply.py boundary so the contract survives refactors.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_operations_returns_snapshot_with_per_op_detail(mocker):
    """The stats dict must include a ``snapshot: DreamOperationsSnapshot``
    field with one entry per write/proposal and per-demotion detail.

    Consumers (AgentProbe scorers, admin visualizer, future P9 SSE
    event) read this; counts alone aren't enough."""
    from backend.copilot.dream.schemas import DreamOperationsSnapshot

    ops = DreamOperations(
        writes=[
            ConsolidatedFact(
                content="A likes B",
                confidence=0.8,
                scope="real:global",
                source_episode_uuids=["ep-1", "ep-2"],
            )
        ],
        proposals=[
            ProposedFinding(
                content="A trusts B",
                confidence=0.6,
                rationale="implied",
                source_fact_uuids=["f1"],
            )
        ],
        demotions=[
            DreamDemotion(edge_uuid="d1", reason="stale", new_status="superseded"),
        ],
        entity_invalidations=[
            EntityInvalidation(entity_uuid="ent-x", reason="dead_to_us"),
        ],
        summary_for_user="ok",
    )
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-snap"),
        pass_id="p-snap",
        ops=ops,
        known_fact_uuids={"f1", "d1"},
        known_episode_uuids={"ep-1", "ep-2"},
    )

    snap = stats["snapshot"]
    assert isinstance(snap, DreamOperationsSnapshot)
    assert len(snap.writes) == 1
    assert snap.writes[0].content == "A likes B"
    assert snap.writes[0].status == "active"
    assert snap.writes[0].source_episode_uuids == ["ep-1", "ep-2"]
    assert len(snap.proposals) == 1
    assert snap.proposals[0].status == "tentative"
    # Proposal provenance must carry BOTH episode + fact source uuids.
    assert snap.proposals[0].source_fact_uuids == ["f1"]
    assert len(snap.demotions) == 1
    assert snap.demotions[0].edge_uuid == "d1"
    assert snap.demotions[0].new_status == "superseded"
    assert snap.demotions[0].applied is True
    assert len(snap.entity_invalidations) == 1
    assert snap.entity_invalidations[0].entity_uuid == "ent-x"
    # ``invalidate_entity_direct_neighbors`` changes ["e1","e2"] per fixture stub
    assert snap.entity_invalidations[0].edges_touched == ["e1", "e2"]


@pytest.mark.asyncio
async def test_apply_operations_demotion_summary_marks_applied_false_on_miss(mocker):
    """When the guarded write reports the edge failed (no longer live),
    the corresponding DemotionSummary records ``applied=False`` so the
    consumer can render a "stale-uuid skip" without inferring it."""
    from backend.copilot.dream.schemas import DreamOperationsSnapshot

    # Override the default success stub: this uuid lands in the bad list.
    mocker.patch.object(
        demotions_mod,
        "supersede_unless_recalled",
        AsyncMock(return_value=[WriteOutcome.UNMATCHED]),
    )
    ops = DreamOperations(
        demotions=[
            DreamDemotion(
                edge_uuid="d-missing", reason="stale", new_status="superseded"
            ),
        ],
    )
    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-miss"), pass_id="p-miss", ops=ops
    )
    snap = stats["snapshot"]
    assert isinstance(snap, DreamOperationsSnapshot)
    assert len(snap.demotions) == 1
    assert snap.demotions[0].edge_uuid == "d-missing"
    assert snap.demotions[0].applied is False
    assert snap.demotions[0].protected is False
    assert stats["demotion_failed_count"] == 1


@pytest.mark.asyncio
async def test_what_the_guarded_writes_spared_reaches_the_stats_and_snapshot(
    mocker,
):
    """``protected_demotions`` is the distinct facts the writes spared, less
    any the final read finds no longer live (here all are); a spared demotion
    is neither applied nor failed."""
    mocker.patch.object(
        demotions_mod,
        "supersede_unless_recalled",
        AsyncMock(return_value=[WriteOutcome.SPARED, WriteOutcome.CHANGED]),
    )
    mocker.patch.object(
        demotions_mod,
        "invalidate_entity_direct_neighbors",
        AsyncMock(return_value=NeighbourWrites(changed=["n1"], spared=["n2", "n3"])),
    )
    ops = DreamOperations(
        demotions=[
            DreamDemotion(edge_uuid="hot", reason="stale_fact"),
            DreamDemotion(edge_uuid="cold", reason="stale_fact"),
        ],
        entity_invalidations=[EntityInvalidation(entity_uuid="ent", reason="gone")],
        summary_for_user="ok",
    )

    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-spared"), pass_id="p-spared", ops=ops
    )

    assert (
        stats["demotion_count"],
        stats["demotion_failed_count"],
        stats["entity_invalidation_count"],
        stats["protected_demotions"],
        stats["indeterminate_demotion_writes"],
        stats["demotion_accounting_complete"],
    ) == (1, 0, 1, 3, 0, True)
    snap = stats["snapshot"]
    assert isinstance(snap, DreamOperationsSnapshot)
    assert [(d.edge_uuid, d.applied, d.protected) for d in snap.demotions] == [
        ("hot", False, True),
        ("cold", True, False),
    ]
    assert snap.entity_invalidations[0].edges_touched == ["n1"]
    assert snap.entity_invalidations[0].edges_protected == ["n2", "n3"]


@pytest.mark.asyncio
async def test_unknown_writes_and_a_failed_read_reach_the_stats(mocker):
    """A write whose reply was lost and a neighbour statement that raised
    are indeterminate, not failures; with the liveness read failing too, the
    protected count is provisional and marked so."""
    mocker.patch.object(
        demotions_mod,
        "supersede_unless_recalled",
        AsyncMock(return_value=[WriteOutcome.SPARED, WriteOutcome.UNKNOWN]),
    )
    mocker.patch.object(
        demotions_mod,
        "invalidate_entity_direct_neighbors",
        AsyncMock(return_value=NeighbourWrites(unknown=True)),
    )
    mocker.patch.object(
        demotions_mod, "live_fact_uuids", AsyncMock(side_effect=TimeoutError)
    )
    ops = DreamOperations(
        demotions=[
            DreamDemotion(edge_uuid="hot", reason="stale_fact"),
            DreamDemotion(edge_uuid="gone", reason="stale_fact"),
        ],
        entity_invalidations=[EntityInvalidation(entity_uuid="ent", reason="gone")],
        summary_for_user="ok",
    )

    stats = await apply_mod.apply_operations(
        scope=MemoryScope.for_user("u-unknown"), pass_id="p-unknown", ops=ops
    )

    assert (
        stats["demotion_count"],
        stats["demotion_failed_count"],
        stats["indeterminate_demotion_writes"],
        stats["protected_demotions"],
        stats["demotion_accounting_complete"],
    ) == (0, 0, 2, 1, False)
    snap = stats["snapshot"]
    assert isinstance(snap, DreamOperationsSnapshot)
    assert [d.indeterminate for d in snap.demotions] == [False, True]
    assert snap.entity_invalidations[0].indeterminate is True
