"""A hard forget leaves none of the dream's copies of what it forgot in
Postgres (``dream/forget_copies.py``), end to end: production ingestion,
apply and forget against a live FalkorDB and a real Postgres, with canned
model answers.

These are the reproductions of an independent review of PR 13 (round 1),
which found the forgotten sentence kept by a pass still open when the forget
landed, by a narrative a cancelled apply wrote after it, by a failed pass's
error, and by a pass behind a thousand newer ones. Each now ends with the
sentence gone. A soft forget blanks nothing in Postgres, but a write it
dropped still keeps the narrative out (``dream/apply.py``).

Run with FalkorDB reachable (see ``conftest.py``) and ``DATABASE_URL`` on a
migrated database::

    poetry run pytest -m integration backend/copilot/graphiti/recall_forget_copies_integration_test.py
"""

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from prisma.models import ChatMessage, ChatSession, DreamPass, User
from pytest_mock import MockerFixture

from backend.copilot import db as chat
from backend.copilot.dream import apply, forget_copies, store
from backend.copilot.dream.pass_record import cancelled
from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamOperations,
    DreamPassResult,
    ProposedFinding,
)
from backend.data import db, dream_pass
from backend.data.dream_pass_models import DreamPassDraft
from backend.util.json import SafeJson

from . import recall_forget
from .recall import FORGOTTEN_FACT
from .recall_cascade_fixtures import FLOUR, SUPPLIES, dream, gather
from .recall_integration_fixtures import (
    edge_row,
    ingest_facts,
    live_facts,
    patch_recall_boundaries,
    stop_ingestion_workers,
)
from .scope import MemoryScope

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="function")]

_DERIVED = SUPPLIES[2]


@pytest_asyncio.fixture(loop_scope="function", autouse=True)
async def postgres(
    mocker: MockerFixture, scope_graph, stub_graphiti_client
) -> AsyncIterator[None]:
    """The scope's owner in a real Postgres, and the dream's store, its
    narrative and the forget's blanking on it; the graph as ``conftest.py``
    gives it."""
    _, scope = scope_graph
    await db.connect()
    await User.prisma().create(
        data={
            "id": scope.owner_user_id,
            "email": f"{uuid.uuid4()}@example.invalid",
            "topUpConfig": SafeJson({}),
            "timezone": "UTC",
        }
    )
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)
    mocker.patch.object(
        recall_forget, "blank_dream_copies", forget_copies.blank_dream_copies
    )
    mocker.patch.object(forget_copies, "chat_db", lambda: chat)
    mocker.patch.object(store, "dream_db", lambda: dream_pass)
    mocker.patch("backend.data.db_accessors.chat_db", lambda: chat)
    mocker.patch(
        "backend.copilot.dream.registry.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    mocker.patch.object(apply, "_create_dream_session", _dream_session)
    yield
    await stop_ingestion_workers()
    await User.prisma().delete(where={"id": scope.owner_user_id})
    await db.disconnect()


async def _dream_session(scope: MemoryScope, pass_id: str) -> str:
    session = await ChatSession.prisma().create(
        data={
            "userId": scope.owner_user_id,
            "metadata": SafeJson({"kind": "dream", "dream_pass_id": pass_id}),
        }
    )
    return session.id


async def _open_pass(scope: MemoryScope) -> str:
    pass_id = str(uuid.uuid4())
    await dream_pass.create_dream_pass(
        DreamPassDraft(
            id=pass_id,
            user_id=scope.owner_user_id,
            scope_key=scope.scope_key,
            route=DreamPassRoute.ANTHROPIC_BATCH,
            trigger=DreamPassTrigger.CRON,
            status=DreamPassStatus.QUEUED,
        )
    )
    return pass_id


async def _source(scope_graph, build) -> str:
    driver, scope = scope_graph
    _, facts = await ingest_facts(
        driver, scope, build, [FLOUR], session_id="source-session"
    )
    return facts[FLOUR[2]]


async def _kept(pass_id: str, session_id: str | None) -> list[str]:
    """Where the pass's row or its narrative still holds the derived
    sentence."""
    row = await dream_pass.get_dream_pass(pass_id)
    assert row is not None
    kept = [
        name
        for name, text in (
            ("phase outputs", row.phase_outputs.model_dump_json()),
            ("operations", row.operations.model_dump_json()),
            ("error", row.error or ""),
        )
        if _DERIVED in text
    ]
    if session_id is not None:
        message = await ChatMessage.prisma().find_first(
            where={"sessionId": session_id, "sequence": 0}
        )
        if message is not None and _DERIVED in (message.content or ""):
            kept.append("narrative")
    return kept


def _completed(scope: MemoryScope, pass_id: str, stats: dict, ops) -> DreamPassResult:
    now = datetime.now(timezone.utc)
    return DreamPassResult(
        user_id=scope.owner_user_id,
        pass_id=pass_id,
        started_at=now,
        completed_at=now,
        execution_path="anthropic_batch",
        dropped_forgotten=stats["dropped_forgotten"],
        summary_for_user=ops.summary_for_user,
        dream_session_id=stats["session_id"],
    )


def _derived_ops(root: str, operation: str) -> tuple[DreamOperations, ConsolidatedFact]:
    write = ConsolidatedFact(content=_DERIVED, confidence=0.9, source_fact_uuids=[root])
    if operation == "propose":
        proposal = ProposedFinding(
            content=_DERIVED,
            rationale=FLOUR[2],
            confidence=0.9,
            source_fact_uuids=[root],
        )
        return DreamOperations(proposals=[proposal], summary_for_user=_DERIVED), write
    return DreamOperations(writes=[write], summary_for_user=_DERIVED), write


@pytest.mark.parametrize("operation", ["consolidate", "propose"])
async def test_an_open_pass_that_read_a_hard_forgotten_fact_keeps_none_of_it(
    scope_graph, stub_graphiti_client, operation
):
    """The review's first reproduction: a pass gathers the source, a hard
    forget deletes it while the pass is open, and the pass goes on to record
    its phases, apply (its write is dropped) and complete. The forget
    reports nothing left to do, and the closed pass keeps none of the
    sentence."""
    driver, scope = scope_graph
    root = await _source(scope_graph, stub_graphiti_client)
    pass_id = await _open_pass(scope)
    read = await gather(scope)
    ops, write = _derived_ops(root, operation)

    forgot = await recall_forget.retract(scope, [root], hard=True)

    assert (forgot.deleted, forgot.failures) == ([root], [])
    assert await edge_row(driver, root) == {}
    await store.record_phase_output(
        pass_id, "consolidate", ConsolidationOutput(facts=[write])
    )
    await store.record_applying(pass_id, ops)
    stats = await dream(
        driver, scope, stub_graphiti_client, ops, read, SUPPLIES, pass_id
    )
    assert stats["dropped_forgotten"] == 1
    assert await live_facts(driver) == {}
    await store.record_batch_complete(_completed(scope, pass_id, stats, ops), None)
    await store.record_cleanup_finished(pass_id)

    row = await dream_pass.get_dream_pass(pass_id)
    assert row is not None and row.status == DreamPassStatus.COMPLETE
    assert await _kept(pass_id, stats["session_id"]) == []


async def test_a_soft_forget_keeps_a_dropped_write_out_of_the_narrative(
    scope_graph, stub_graphiti_client
):
    """A soft forget marks no pass, but the write it made the worker drop
    may be restated anywhere in the summary: the narrative is the
    placeholder."""
    driver, scope = scope_graph
    root = await _source(scope_graph, stub_graphiti_client)
    pass_id = await _open_pass(scope)
    read = await gather(scope)
    ops, _ = _derived_ops(root, "consolidate")
    await recall_forget.retract(scope, [root])

    stats = await dream(
        driver, scope, stub_graphiti_client, ops, read, SUPPLIES, pass_id
    )

    assert stats["dropped_forgotten"] == 1
    message = await ChatMessage.prisma().find_first(
        where={"sessionId": stats["session_id"], "sequence": 0}
    )
    assert message is not None and message.content == FORGOTTEN_FACT
    row = await dream_pass.get_dream_pass(pass_id)
    assert row is not None and row.erased_at is None, "a soft forget marks nothing"


async def test_a_cancelled_apply_writes_no_narrative_after_the_forget(
    scope_graph, stub_graphiti_client, mocker: MockerFixture
):
    """The review's race: apply's drain finishes (the write is made), the
    pass is cancelled, a hard forget erases the write and finds no
    narrative yet, then apply resumes and writes it. The narrative goes in
    blank: the forget marked the pass."""
    driver, scope = scope_graph
    root = await _source(scope_graph, stub_graphiti_client)
    pass_id = await _open_pass(scope)
    read = await gather(scope)
    ops, _ = _derived_ops(root, "consolidate")
    await store.record_applying(pass_id, ops)
    drain = apply._drain_ingestion
    forgot = []

    async def close_then_forget(*args, **kwargs):
        status = await drain(*args, **kwargs)
        assert await store.write_stop(
            pass_id, cancelled("owner cancelled", owner_user_id=scope.owner_user_id)
        )
        forgot.append(await recall_forget.retract(scope, [root], hard=True))
        assert (
            await ChatMessage.prisma().count(
                where={
                    "Session": {"is": {"userId": scope.owner_user_id}},
                    "sequence": 0,
                }
            )
            == 0
        ), "no narrative yet"
        return status

    mocker.patch.object(apply, "_drain_ingestion", close_then_forget)
    stats = await dream(
        driver, scope, stub_graphiti_client, ops, read, SUPPLIES, pass_id
    )

    assert forgot[0].failures == []
    assert stats["dropped_forgotten"] == 0, "the write was made, then erased"
    row = await dream_pass.get_dream_pass(pass_id)
    assert row is not None and row.status == DreamPassStatus.CANCELLED
    assert await live_facts(driver) == {}
    assert await _kept(pass_id, stats["session_id"]) == []


async def test_a_failed_passes_error_keeps_only_its_code(
    scope_graph, stub_graphiti_client
):
    """The review's error column: a pass failed with an error from before
    errors were codes, quoting the model's answer, and names no uuid in it.
    The hard forget keeps the phase and blanks the rest."""
    _, scope = scope_graph
    root = await _source(scope_graph, stub_graphiti_client)
    pass_id = await _open_pass(scope)
    write = ConsolidatedFact(content=_DERIVED, confidence=0.9, source_fact_uuids=[root])
    await store.record_phase_output(
        pass_id, "consolidate", ConsolidationOutput(facts=[write])
    )
    quoted = f"recombine: LLM returned non-JSON content — first 200 chars: {_DERIVED}"
    await store.record_sync_outcome(
        DreamPassResult(user_id=scope.owner_user_id, pass_id=pass_id, error=quoted)
    )

    forgot = await recall_forget.retract(scope, [root], hard=True)

    assert (forgot.deleted, forgot.failures) == ([root], [])
    row = await dream_pass.get_dream_pass(pass_id)
    assert row is not None and row.error == f"recombine: {FORGOTTEN_FACT}"
    assert await _kept(pass_id, None) == []


async def test_an_affected_pass_behind_a_thousand_newer_ones_is_blanked(
    scope_graph, stub_graphiti_client
):
    """The review's 1,001 passes: the affected one is the oldest of the
    scope, its narrative named by no erased write."""
    _, scope = scope_graph
    root = await _source(scope_graph, stub_graphiti_client)
    old = await _open_pass(scope)
    write = ConsolidatedFact(content=_DERIVED, confidence=0.9, source_fact_uuids=[root])
    await store.record_phase_output(
        old, "consolidate", ConsolidationOutput(facts=[write])
    )
    await store.record_sync_outcome(
        DreamPassResult(user_id=scope.owner_user_id, pass_id=old)
    )
    await DreamPass.prisma().create_many(
        data=[
            {
                "id": str(uuid.uuid4()),
                "userId": scope.owner_user_id,
                "scopeKey": scope.scope_key,
                "route": DreamPassRoute.SYNC,
                "trigger": DreamPassTrigger.ADMIN,
                "status": DreamPassStatus.SKIPPED,
                "phase": DreamPassPhase.DONE,
            }
            for _ in range(1000)
        ]
    )
    session_id = await _dream_session(scope, old)
    await chat.add_chat_message(session_id, "assistant", 0, _DERIVED)

    forgot = await recall_forget.retract(scope, [root], hard=True)

    assert forgot.failures == []
    assert await _kept(old, session_id) == []
