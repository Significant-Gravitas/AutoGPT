"""Apply tests — mock the Graphiti/Postgres boundary, verify the right
calls are produced for each operation type.

These tests do NOT touch FalkorDB or Prisma. apply.py is a pure
fan-out: it builds MemoryEnvelopes for writes/proposals and delegates
to ``enqueue_episode`` / ``mark_edges_superseded`` /
``invalidate_entity_direct_neighbors`` / ``create_chat_session`` /
``add_chat_message``. Each of those is mocked here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from . import apply as apply_mod
from .schemas import (
    ConsolidatedFact,
    DreamDemotion,
    DreamOperations,
    EntityInvalidation,
    ProposedFinding,
)


@pytest.fixture(autouse=True)
def _stub_boundaries(mocker):
    """Wire up the apply.py side-effects with AsyncMocks once per test."""
    mocker.patch.object(apply_mod, "enqueue_episode", AsyncMock(return_value=True))
    # The driver constructor + close — apply.py opens a FalkorDB driver for
    # demotions and entity invalidations. Patch where it's used.
    driver = mocker.MagicMock()
    driver.close = AsyncMock(return_value=None)
    mocker.patch.object(
        apply_mod, "AutoGPTFalkorDriver", mocker.MagicMock(return_value=driver)
    )
    # Helper functions
    mocker.patch.object(
        apply_mod,
        "mark_edges_superseded",
        AsyncMock(return_value=(["e1"], [])),
    )
    mocker.patch.object(
        apply_mod,
        "invalidate_entity_direct_neighbors",
        AsyncMock(return_value=["e1", "e2"]),
    )
    # ChatSession + ChatMessage writes — apply.py imports them lazily inside
    # ``_write_dream_session`` to avoid a circular import. Patch where the
    # symbol is looked up (the copilot.db module).
    mocker.patch(
        "backend.copilot.db.create_chat_session",
        AsyncMock(return_value=mocker.MagicMock(session_id="s1")),
    )
    mocker.patch("backend.copilot.db.add_chat_message", AsyncMock(return_value=None))
    # derive_group_id is deterministic; let it run.


@pytest.mark.asyncio
async def test_writes_become_active_envelopes():
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(content="A likes B", confidence=0.8, scope="real:global")
        ],
        summary_for_user="ok",
    )
    stats = await apply_mod.apply_operations(
        user_id="u-1234567890ab", pass_id="p-abc", ops=ops
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
    await apply_mod.apply_operations(user_id="u-x", pass_id="p-2", ops=ops)
    # First call was the consolidate write (there were no writes, so first call IS the proposal)
    call_kwargs = apply_mod.enqueue_episode.await_args.kwargs
    assert call_kwargs["name"].startswith("dream_p-2_recombine_")
    body = call_kwargs["episode_body"]
    assert '"status":"tentative"' in body


@pytest.mark.asyncio
async def test_demotions_group_by_status_and_reason():
    """Bucketed mark_edges_superseded calls — one per (status, reason) pair."""
    ops = DreamOperations(
        demotions=[
            DreamDemotion(edge_uuid="a", reason="stale", new_status="superseded"),
            DreamDemotion(edge_uuid="b", reason="stale", new_status="superseded"),
            DreamDemotion(
                edge_uuid="c", reason="contradicted_by:x", new_status="contradicted"
            ),
        ],
    )
    await apply_mod.apply_operations(user_id="u-y", pass_id="p-3", ops=ops)

    # Three demotions but only TWO buckets: (superseded, stale) and
    # (contradicted, contradicted_by:x)
    assert apply_mod.mark_edges_superseded.await_count == 2
    bucket_args = [
        call.args[1] if len(call.args) > 1 else call.kwargs.get("uuids")
        for call in apply_mod.mark_edges_superseded.await_args_list
    ]
    # One bucket has 2 uuids, the other has 1
    assert sorted(len(b) for b in bucket_args) == [1, 2]


@pytest.mark.asyncio
async def test_entity_invalidation_calls_single_hop_helper():
    ops = DreamOperations(
        entity_invalidations=[
            EntityInvalidation(entity_uuid="ent-x", reason="dead_to_us"),
        ],
    )
    await apply_mod.apply_operations(user_id="u-z", pass_id="p-4", ops=ops)

    apply_mod.invalidate_entity_direct_neighbors.assert_awaited_once()
    kwargs = apply_mod.invalidate_entity_direct_neighbors.await_args.kwargs
    assert kwargs["entity_uuid"] == "ent-x"
    assert kwargs["reason"] == "dead_to_us"


@pytest.mark.asyncio
async def test_no_op_dream_still_writes_summary_session(mocker):
    """Empty DreamOperations still creates the dream-kind ChatSession."""
    from backend.copilot import db as copilot_db

    ops = DreamOperations(summary_for_user="Nothing new today.")
    stats = await apply_mod.apply_operations(user_id="u-a", pass_id="p-5", ops=ops)

    assert stats["consolidated_count"] == 0
    assert stats["proposal_count"] == 0
    assert stats["demotion_count"] == 0
    # ChatSession + ChatMessage were both created
    copilot_db.create_chat_session.assert_awaited_once()
    copilot_db.add_chat_message.assert_awaited_once()
    msg_kwargs = copilot_db.add_chat_message.await_args.kwargs
    assert msg_kwargs["role"] == "assistant"
    assert msg_kwargs["content"] == "Nothing new today."


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
        apply_mod, "_write_dream_session", new_callable=AsyncMock, return_value="s"
    )
    mocker.patch.object(
        apply_mod, "_apply_demotions", new_callable=AsyncMock, return_value=(0, 0, [])
    )
    mocker.patch.object(
        apply_mod,
        "_apply_entity_invalidations",
        new_callable=AsyncMock,
        return_value=(0, []),
    )

    # Whatever state Prisma is in, apply_operations must not touch
    # ``platform_db.connect``. Spy on BOTH states to make the contract
    # explicit.
    for is_conn in (False, True):
        mocker.patch("backend.data.db.is_connected", return_value=is_conn)
        connect_spy = mocker.patch("backend.data.db.connect", new_callable=AsyncMock)

        await apply_mod.apply_operations(
            user_id="u-1",
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

    # The autouse fixture stubs mark_edges_superseded to return ["e1"]
    # in the succeeded list, which doesn't match our test uuid "d1".
    # Override so d1 lands in the succeeded list.
    mocker.patch.object(
        apply_mod,
        "mark_edges_superseded",
        AsyncMock(return_value=(["d1"], [])),
    )

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
        user_id="u-snap", pass_id="p-snap", ops=ops
    )

    snap = stats["snapshot"]
    assert isinstance(snap, DreamOperationsSnapshot)
    assert len(snap.writes) == 1
    assert snap.writes[0].content == "A likes B"
    assert snap.writes[0].status == "active"
    assert snap.writes[0].source_episode_uuids == ["ep-1", "ep-2"]
    assert len(snap.proposals) == 1
    assert snap.proposals[0].status == "tentative"
    assert len(snap.demotions) == 1
    assert snap.demotions[0].edge_uuid == "d1"
    assert snap.demotions[0].new_status == "superseded"
    assert snap.demotions[0].applied is True
    assert len(snap.entity_invalidations) == 1
    assert snap.entity_invalidations[0].entity_uuid == "ent-x"
    # ``invalidate_entity_direct_neighbors`` returns ["e1","e2"] per fixture stub
    assert snap.entity_invalidations[0].edges_touched == ["e1", "e2"]


@pytest.mark.asyncio
async def test_apply_operations_demotion_summary_marks_applied_false_on_miss(mocker):
    """When mark_edges_superseded returns the uuid in the failed list,
    the corresponding DemotionSummary records ``applied=False`` so the
    consumer can render a "stale-uuid skip" without inferring it."""
    from backend.copilot.dream.schemas import DreamOperationsSnapshot

    # Override the default success stub: this uuid lands in the bad list.
    mocker.patch.object(
        apply_mod,
        "mark_edges_superseded",
        AsyncMock(return_value=([], ["d-missing"])),
    )
    ops = DreamOperations(
        demotions=[
            DreamDemotion(
                edge_uuid="d-missing", reason="stale", new_status="superseded"
            ),
        ],
    )
    stats = await apply_mod.apply_operations(
        user_id="u-miss", pass_id="p-miss", ops=ops
    )
    snap = stats["snapshot"]
    assert isinstance(snap, DreamOperationsSnapshot)
    assert len(snap.demotions) == 1
    assert snap.demotions[0].edge_uuid == "d-missing"
    assert snap.demotions[0].applied is False
