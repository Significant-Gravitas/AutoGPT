"""A hard forget leaves none of the dream's copies of what it forgot in
Postgres (``dream/forget_copies.py``), end to end: production ingestion,
apply and forget against a live FalkorDB and a real Postgres, with canned
model answers.

These are the reproductions of independent reviews of PR 13. Round 1
found the forgotten sentence kept by a pass still open when the forget
landed, by a narrative a cancelled apply wrote after it, by a failed pass's
error, and by a pass behind a thousand newer ones; round 2 by a pass whose
insert failed, by a summary restating a fact no write cited, and by a
narrative a failed blank left for a repeat that found no pass; round 3 by
a pass from before read sets were recorded, its narrative written before
the deploy or after it, its row kept or aged out. Each now ends with the
sentence gone, as does a narrative whose pass's row aged out and the pass
of a dream write forgotten itself, while a forget of what a pass with a
known read set never read leaves it alone. A soft forget blanks
nothing in Postgres, but a write it dropped still keeps the narrative out
(``dream/apply.py``).

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
from prisma import Prisma
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from prisma.models import ChatMessage, ChatSession, DreamPass, Expert, User
from pytest_mock import MockerFixture

from backend.copilot import db as chat
from backend.copilot.dream import apply, forget_copies, store
from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.pass_errors import UNRECORDED, pass_error
from backend.copilot.dream.pass_record import cancelled
from backend.copilot.dream.pass_run import DreamPassRun
from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamOperations,
    DreamPassResult,
    ProposedFinding,
)
from backend.data import db, dream_pass
from backend.data.dream_pass_models import (
    UNKNOWN_READS,
    DreamPassDraft,
    DreamPassUpdate,
)
from backend.util.json import SafeJson

from . import recall_forget
from .recall import FORGOTTEN_FACT
from .recall_cascade_fixtures import CAFE, FLOUR, SUPPLIES, dream, fact_uuid, gather
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
    gives it.

    Each test runs on a loop of its own, for its FalkorDB driver, so it gets
    a Prisma client of its own, connected on that loop and registered for
    the test: the session's shared client may already be connected on the
    session's loop by an earlier test, and its HTTP pool works only on the
    loop that opened it."""
    _, scope = scope_graph
    client = Prisma(
        http={"timeout": db.HTTP_TIMEOUT}, datasource={"url": db.DATABASE_URL}
    )
    await client.connect()
    try:
        mocker.patch.object(db, "prisma", client)
        mocker.patch("prisma._registry._registered_client", client)
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
    finally:
        await client.disconnect(db.DISCONNECT_TIMEOUT)


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
            pass_id,
            cancelled(
                pass_error("cancelled", "by_admin"),
                owner_user_id=scope.owner_user_id,
            ),
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


async def test_a_failed_passes_error_never_quotes_the_model(
    scope_graph, stub_graphiti_client
):
    """The review's error column: a pass fails with an error quoting the
    model's answer. Only a fixed code is recorded, and the hard forget finds
    the closed pass by what it read and blanks the rest."""
    _, scope = scope_graph
    root = await _source(scope_graph, stub_graphiti_client)
    pass_id, _ = await _gathered_pass(scope)
    write = ConsolidatedFact(content=_DERIVED, confidence=0.9, source_fact_uuids=[root])
    await store.record_phase_output(
        pass_id, "consolidate", ConsolidationOutput(facts=[write])
    )
    quoted = f"recombine: LLM returned non-JSON content — first 200 chars: {_DERIVED}"
    await store.record_sync_outcome(
        DreamPassResult(user_id=scope.owner_user_id, pass_id=pass_id, error=quoted)
    )
    row = await dream_pass.get_dream_pass(pass_id)
    assert row is not None and row.error == UNRECORDED

    forgot = await recall_forget.retract(scope, [root], hard=True)

    assert (forgot.deleted, forgot.failures) == ([root], [])
    assert await _kept(pass_id, None) == []


async def test_an_affected_pass_behind_a_thousand_newer_ones_is_blanked(
    scope_graph, stub_graphiti_client
):
    """The review's 1,001 passes: the affected one is the oldest of the
    scope, its narrative named by no erased write."""
    _, scope = scope_graph
    root = await _source(scope_graph, stub_graphiti_client)
    old, _ = await _gathered_pass(scope)
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
                "readUuids": [],
            }
            for _ in range(1000)
        ]
    )
    session_id = await _dream_session(scope, old)
    await chat.add_chat_message(session_id, "assistant", 0, _DERIVED)

    forgot = await recall_forget.retract(scope, [root], hard=True)

    assert forgot.failures == []
    assert await _kept(old, session_id) == []


async def _gathered_pass(scope: MemoryScope) -> tuple[str, DreamInput]:
    """An open pass that gathered the scope and recorded what it read, as
    every pass does before its first model call."""
    pass_id = await _open_pass(scope)
    read = await gather(scope)
    assert await store.record_gathered(pass_id, read)
    return pass_id, read


async def _cafe_pass(
    scope_graph, build
) -> tuple[str, str, DreamOperations, DreamInput]:
    """The source, an unrelated cafe fact, and an open pass that read both
    and is about to write a consolidation of the cafe (citing it) with a
    summary restating the source, citing nothing: the source's uuid, the
    pass id, its operations and what it read."""
    driver, scope = scope_graph
    root = await _source(scope_graph, build)
    _, other = await ingest_facts(
        driver, scope, build, [CAFE], session_id="other-source"
    )
    pass_id, read = await _gathered_pass(scope)
    assert {root, other[CAFE[2]]} <= read.known_fact_uuids
    write = ConsolidatedFact(
        content=CAFE[2], confidence=0.9, source_fact_uuids=[other[CAFE[2]]]
    )
    await store.record_phase_output(
        pass_id, "consolidate", ConsolidationOutput(facts=[write])
    )
    return (
        root,
        pass_id,
        DreamOperations(writes=[write], summary_for_user=_DERIVED),
        read,
    )


async def _complete(scope: MemoryScope, pass_id: str, stats: dict, ops) -> None:
    await store.record_batch_complete(_completed(scope, pass_id, stats, ops), None)
    assert await store.record_cleanup_finished(pass_id)


async def test_a_pass_whose_record_did_not_go_in_writes_no_narrative(
    scope_graph, stub_graphiti_client, mocker: MockerFixture
):
    """The round-2 review's first reproduction: the pass's insert fails. The
    pass stops before it gathers (``orchestrator``); should its apply run
    anyway, as the reproduction makes it, after a hard forget erased its
    write, the narrative has no row to go by and is not written."""
    driver, scope = scope_graph
    root = await _source(scope_graph, stub_graphiti_client)
    run = DreamPassRun.begin(scope.owner_user_id, "sync_baseline")
    mocker.patch.object(
        dream_pass, "create_dream_pass", AsyncMock(side_effect=OSError("insert"))
    )
    assert not await store.start_pass(run, scope, trigger="cron")
    assert await dream_pass.get_dream_pass(run.pass_id) is None
    read = await gather(scope)
    ops, _ = _derived_ops(root, "consolidate")
    drain = apply._drain_ingestion
    forgot = []

    async def forget_after_the_write(*args, **kwargs):
        status = await drain(*args, **kwargs)
        forgot.append(await recall_forget.retract(scope, [root], hard=True))
        return status

    mocker.patch.object(apply, "_drain_ingestion", forget_after_the_write)
    stats = await dream(
        driver, scope, stub_graphiti_client, ops, read, SUPPLIES, run.pass_id
    )

    assert forgot[0].failures == [] and forgot[0].dream_writes
    assert await live_facts(driver) == {}
    assert (
        await ChatMessage.prisma().count(where={"sessionId": stats["session_id"]}) == 0
    )


async def test_a_summary_restating_a_fact_it_did_not_cite_is_blanked(
    scope_graph, stub_graphiti_client
):
    """The round-2 review's second reproduction: the pass's summary restates
    the source, which no write cites. Completed and cleaned up, the pass is
    found by what it read, and the forget blanks its operations and its
    narrative."""
    driver, scope = scope_graph
    root, pass_id, ops, read = await _cafe_pass(scope_graph, stub_graphiti_client)
    await store.record_applying(pass_id, ops)
    stats = await dream(driver, scope, stub_graphiti_client, ops, read, CAFE, pass_id)
    await _complete(scope, pass_id, stats, ops)
    assert await _kept(pass_id, stats["session_id"]) == ["operations", "narrative"]

    forgot = await recall_forget.retract(scope, [root], hard=True)

    assert forgot.failures == []
    row = await dream_pass.get_dream_pass(pass_id)
    assert row is not None and row.erased_at is not None
    assert await _kept(pass_id, stats["session_id"]) == []


async def test_a_forget_repeated_after_its_narrative_blank_failed_finishes_it(
    scope_graph, stub_graphiti_client, mocker: MockerFixture
):
    """The round-2 review's third reproduction: the forget marks the open
    pass but cannot blank its narrative (a cleanup error, to repeat); the
    pass then completes, its row blanked as it closes. Repeated, the forget
    finds the pass by its mark and blanks the narrative."""
    driver, scope = scope_graph
    root, pass_id, ops, read = await _cafe_pass(scope_graph, stub_graphiti_client)
    await store.record_applying(pass_id, ops)
    stats = await dream(driver, scope, stub_graphiti_client, ops, read, CAFE, pass_id)
    blank = chat.blank_dream_narratives
    mocker.patch.object(
        chat, "blank_dream_narratives", AsyncMock(side_effect=OSError("down"))
    )
    first = await recall_forget.retract(scope, [root], hard=True)
    assert [failure.code for failure in first.failures] == ["cleanup_error"]
    mocker.patch.object(chat, "blank_dream_narratives", blank)
    await _complete(scope, pass_id, stats, ops)
    assert await _kept(pass_id, stats["session_id"]) == ["narrative"]

    again = await recall_forget.retract(scope, [root], hard=True)

    assert again.failures == []
    assert await _kept(pass_id, stats["session_id"]) == []


async def test_a_narrative_whose_pass_row_aged_out_is_found_by_what_it_read(
    scope_graph, stub_graphiti_client
):
    """The pass's row is deleted by retention after it completed; its
    narrative's session kept what the pass read, so a hard forget of the
    source, which names no write of the pass, still blanks it."""
    driver, scope = scope_graph
    root, pass_id, ops, read = await _cafe_pass(scope_graph, stub_graphiti_client)
    stats = await dream(driver, scope, stub_graphiti_client, ops, read, CAFE, pass_id)
    await _complete(scope, pass_id, stats, ops)
    await DreamPass.prisma().delete(where={"id": pass_id})

    forgot = await recall_forget.retract(scope, [root], hard=True)

    assert forgot.failures == []
    message = await ChatMessage.prisma().find_first(
        where={"sessionId": stats["session_id"], "sequence": 0}
    )
    assert message is not None and message.content == FORGOTTEN_FACT


async def _pass_before_read_sets(
    scope_graph, build
) -> tuple[str, str, DreamOperations, DreamInput]:
    """``_cafe_pass``'s pass, written as a release before read sets wrote
    one (the round-3 review's legacy pass): its create states no read set,
    so it reads the column's default, as every row that existed when the
    column was added does, and its gather records the window only."""
    driver, scope = scope_graph
    root = await _source(scope_graph, build)
    _, other = await ingest_facts(
        driver, scope, build, [CAFE], session_id="other-source"
    )
    pass_id = str(uuid.uuid4())
    await DreamPass.prisma().create(
        data={
            "id": pass_id,
            "userId": scope.owner_user_id,
            "scopeKey": scope.scope_key,
            "route": DreamPassRoute.ANTHROPIC_BATCH,
            "trigger": DreamPassTrigger.CRON,
            "status": DreamPassStatus.QUEUED,
        }
    )
    read = await gather(scope)
    assert {root, other[CAFE[2]]} <= read.known_fact_uuids
    assert await dream_pass.update_dream_pass(
        pass_id,
        DreamPassUpdate(window_start=read.window_start, window_end=read.window_end),
    )
    write = ConsolidatedFact(
        content=CAFE[2], confidence=0.9, source_fact_uuids=[other[CAFE[2]]]
    )
    await store.record_phase_output(
        pass_id, "consolidate", ConsolidationOutput(facts=[write])
    )
    row = await dream_pass.get_dream_pass(pass_id)
    assert row is not None and row.read_uuids == [UNKNOWN_READS]
    return (
        root,
        pass_id,
        DreamOperations(writes=[write], summary_for_user=_DERIVED),
        read,
    )


async def _narrative_before_read_sets(
    session_id: str, pass_id: str, content: str, erased: str
) -> bool:
    """The narrative as the release before read sets wrote it: the message
    alone, nothing copied into its session."""
    await chat.add_chat_message(session_id, "assistant", 0, content)
    return True


async def _narrative(session_id: str) -> str | None:
    message = await ChatMessage.prisma().find_first(
        where={"sessionId": session_id, "sequence": 0}
    )
    return message.content if message is not None else None


@pytest.mark.parametrize("row", ["kept", "aged_out"])
@pytest.mark.parametrize("narrative", ["before_the_deploy", "after_the_deploy"])
async def test_a_pass_from_before_read_sets_keeps_none_of_a_hard_forgotten_fact(
    scope_graph, stub_graphiti_client, mocker: MockerFixture, narrative, row
):
    """The round-3 review's two legacy reproductions, turned around. A pass
    from before read sets were recorded gathered the source and an
    unrelated cafe fact, wrote the cafe citing it and a summary restating
    the source, citing nothing. Its narrative was written by the release
    before (the message alone) or by this one, the pass in flight across
    the deploy (its session takes the pass's unknown read set); its row is
    kept or has aged out. What the pass read is unknown, so a hard forget of
    the source, which no write of the pass cites, blanks the narrative and
    the row."""
    driver, scope = scope_graph
    root, pass_id, ops, read = await _pass_before_read_sets(
        scope_graph, stub_graphiti_client
    )
    if narrative == "before_the_deploy":
        mocker.patch.object(chat, "add_dream_narrative", _narrative_before_read_sets)
    await store.record_applying(pass_id, ops)
    stats = await dream(driver, scope, stub_graphiti_client, ops, read, CAFE, pass_id)
    await _complete(scope, pass_id, stats, ops)
    assert await _kept(pass_id, stats["session_id"]) == ["operations", "narrative"]
    if row == "aged_out":
        await DreamPass.prisma().delete(where={"id": pass_id})

    forgot = await recall_forget.retract(scope, [root], hard=True)

    assert (forgot.deleted, forgot.failures) == ([root], [])
    assert await _narrative(stats["session_id"]) == FORGOTTEN_FACT
    if row == "kept":
        kept = await dream_pass.get_dream_pass(pass_id)
        assert kept is not None and kept.erased_at is not None
        assert await _kept(pass_id, stats["session_id"]) == []


async def _old_narrative(user_id: str, expert_id: str | None) -> str:
    """A dream session of the user's scope of *expert_id* (the account's when
    ``None``) with a narrative written before sessions kept a read set."""
    session = await ChatSession.prisma().create(
        data={
            "userId": user_id,
            "metadata": SafeJson({"kind": "dream", "dream_pass_id": "p-before"}),
            **({"expertId": expert_id} if expert_id else {}),
        }
    )
    await chat.add_chat_message(session.id, "assistant", 0, _DERIVED)
    return session.id


async def test_an_unrelated_forget_leaves_a_new_narrative_and_takes_old_ones(
    scope_graph, stub_graphiti_client
):
    """A pass with a known read set is reached only by what it read: a hard
    forget of a fact ingested after it gathered leaves its row and its
    narrative. The same forget blanks a narrative of its scope from before
    sessions kept a read set, and not one of the user's expert's scope."""
    driver, scope = scope_graph
    _, cafe = await ingest_facts(
        driver, scope, stub_graphiti_client, [CAFE], session_id="cafe-source"
    )
    pass_id, read = await _gathered_pass(scope)
    write = ConsolidatedFact(
        content=CAFE[2], confidence=0.9, source_fact_uuids=[cafe[CAFE[2]]]
    )
    ops = DreamOperations(writes=[write], summary_for_user=_DERIVED)
    await store.record_phase_output(
        pass_id, "consolidate", ConsolidationOutput(facts=[write])
    )
    await store.record_applying(pass_id, ops)
    stats = await dream(driver, scope, stub_graphiti_client, ops, read, CAFE, pass_id)
    await _complete(scope, pass_id, stats, ops)
    root = await _source(scope_graph, stub_graphiti_client)
    expert = await Expert.prisma().create(
        data={
            "ownerUserId": scope.owner_user_id,
            "name": "Scout",
            "role": "Research",
            "identity": "Finds things out.",
        }
    )
    old_here = await _old_narrative(scope.owner_user_id, None)
    old_there = await _old_narrative(scope.owner_user_id, expert.id)

    forgot = await recall_forget.retract(scope, [root], hard=True)

    assert (forgot.deleted, forgot.failures) == ([root], [])
    row = await dream_pass.get_dream_pass(pass_id)
    assert row is not None and row.erased_at is None
    assert root not in row.read_uuids and UNKNOWN_READS not in row.read_uuids
    assert await _kept(pass_id, stats["session_id"]) == ["operations", "narrative"]
    assert await _narrative(old_here) == FORGOTTEN_FACT
    assert await _narrative(old_there) == _DERIVED


async def test_forgetting_a_dream_write_blanks_the_pass_that_wrote_it(
    scope_graph, stub_graphiti_client
):
    """A pass does not read what it writes: forgetting the dream's own write
    reaches the pass through the write's episode, which names it."""
    driver, scope = scope_graph
    root = await _source(scope_graph, stub_graphiti_client)
    pass_id, read = await _gathered_pass(scope)
    ops, write = _derived_ops(root, "consolidate")
    await store.record_phase_output(
        pass_id, "consolidate", ConsolidationOutput(facts=[write])
    )
    await store.record_applying(pass_id, ops)
    stats = await dream(
        driver, scope, stub_graphiti_client, ops, read, SUPPLIES, pass_id
    )
    await _complete(scope, pass_id, stats, ops)
    derived = await fact_uuid(driver, _DERIVED)
    row = await dream_pass.get_dream_pass(pass_id)
    assert row is not None and derived not in row.read_uuids

    forgot = await recall_forget.retract(scope, [derived], hard=True)

    assert forgot.failures == []
    assert await _kept(pass_id, stats["session_id"]) == []
