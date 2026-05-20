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
