"""Unit tests for ``forget_copies``: which passes a hard forget marks and
what it blanks of the dream's copies in Postgres, now or as a marked pass
closes, against the in-memory pass record (``conftest.py``, which follows
the SQL's rules) and a mocked chat database.

The real statements are ``backend/data/dream_pass_blank_test.py`` (the pass
row and its transitions) and ``backend/copilot/db_dream_narrative_test.py``
(the narrative); the review's reproductions run end to end in
``graphiti/recall_forget_copies_integration_test.py``.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from pytest_mock import MockerFixture

from backend.copilot.graphiti.recall import FORGOTTEN_FACT
from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import (
    ERASED_TEXT,
    DreamPassApplied,
    DreamPassDraft,
    DreamPassOperations,
    DreamPhaseOutputs,
)

from . import forget_copies
from .apply import _episode_name
from .conftest import FakeDreamDb
from .forget_copies import blank_dream_copies, pass_of
from .pass_record import cancelled, cleanup_finished, expired, failed
from .schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DemotionSummary,
    DreamDemotion,
    DreamOperations,
    DreamOperationsSnapshot,
    DreamPassResult,
    ProposedFinding,
    RecombinationOutput,
    WriteSummary,
)
from .store import record_batch_complete, record_sync_outcome, write_stop

_SCOPE = MemoryScope.for_user("u1")
_NOW = datetime.now(timezone.utc)
_FORGOTTEN = "fact-forgotten"
_SENTENCE = "Alice runs the Atlas launch"


@pytest.fixture
def chat(mocker: MockerFixture) -> MagicMock:
    database = MagicMock()
    database.blank_dream_narratives = AsyncMock(return_value=1)
    mocker.patch.object(forget_copies, "chat_db", return_value=database)
    return database


def _seed(
    fake: FakeDreamDb,
    pass_id: str,
    *,
    cites: str = _FORGOTTEN,
    status: DreamPassStatus = DreamPassStatus.COMPLETE,
    scope: MemoryScope = _SCOPE,
    **columns: Any,
) -> None:
    """A pass whose every phase consolidated ``_SENTENCE`` citing ``cites``,
    proposed a finding on it and demoted another fact."""
    write = ConsolidatedFact(
        content=_SENTENCE, confidence=0.8, source_fact_uuids=[cites]
    )
    proposal = ProposedFinding(
        content="Atlas ships this quarter",
        confidence=0.6,
        rationale=f"{_SENTENCE}, so it ships",
        source_fact_uuids=[cites],
    )
    demotion = DreamDemotion(edge_uuid="fact-other", reason=f"stale: {_SENTENCE}")
    ops = DreamOperations(
        writes=[write],
        proposals=[proposal],
        demotions=[demotion],
        summary_for_user=f"I noticed {_SENTENCE}.",
    )
    applied = DreamPassApplied(
        consolidated_count=1,
        summary_for_user=f"I noticed {_SENTENCE}.",
        snapshot=DreamOperationsSnapshot(
            writes=[WriteSummary(content=_SENTENCE, source_fact_uuids=[cites])],
            demotions=[
                DemotionSummary(
                    edge_uuid="fact-other",
                    reason=f"stale: {_SENTENCE}",
                    new_status="superseded",
                )
            ],
        ),
    )
    fake.seed(
        DreamPassDraft(
            id=pass_id,
            user_id=scope.owner_user_id,
            scope_key=scope.scope_key,
            route=DreamPassRoute.SYNC,
            trigger=DreamPassTrigger.CRON,
            status=status,
            phase=DreamPassPhase.SANITIZE,
        ),
        phase_outputs=DreamPhaseOutputs(
            consolidate=ConsolidationOutput(facts=[write]),
            recombine=RecombinationOutput(proposals=[proposal]),
            sanitize=ops,
        ).model_dump(mode="json"),
        operations=DreamPassOperations(planned=ops, applied=applied).model_dump(
            mode="json"
        ),
        **columns,
    )


def _texts(value: Any) -> list[str]:
    """Every string under ``value`` (a row's JSON column)."""
    if isinstance(value, dict):
        return [text for item in value.values() for text in _texts(item)]
    if isinstance(value, list):
        return [text for item in value for text in _texts(item)]
    return [value] if isinstance(value, str) else []


def _row_text(fake: FakeDreamDb, pass_id: str) -> list[str]:
    row = fake.rows[pass_id]
    return _texts(row["phase_outputs"]) + _texts(row["operations"])


@pytest.mark.asyncio
async def test_a_closed_pass_citing_what_was_forgotten_is_blanked(
    fake_dream_db: FakeDreamDb, chat: MagicMock
) -> None:
    """Every sentence goes, in every phase's output, the plan and apply's
    report; the uuids, scope, counts and statuses stay."""
    _seed(fake_dream_db, "p1")

    marked = await blank_dream_copies(_SCOPE, reached={_FORGOTTEN}, dream_writes=[])

    assert marked == ["p1"]
    assert not any(_SENTENCE in text for text in _row_text(fake_dream_db, "p1"))
    record = fake_dream_db.record("p1")
    assert record.erased_at is not None
    assert record.phase_outputs.consolidate is not None
    [write] = record.phase_outputs.consolidate.facts
    assert (write.content, write.source_fact_uuids) == (FORGOTTEN_FACT, [_FORGOTTEN])
    assert record.operations.applied is not None
    assert record.operations.applied.summary_for_user == FORGOTTEN_FACT
    assert record.operations.applied.consolidated_count == 1
    snapshot = record.operations.applied.snapshot
    assert snapshot is not None
    [demoted] = snapshot.demotions
    assert (demoted.edge_uuid, demoted.new_status) == ("fact-other", "superseded")
    chat.blank_dream_narratives.assert_awaited_once_with("u1", ["p1"], FORGOTTEN_FACT)


@pytest.mark.asyncio
async def test_a_pass_whose_write_was_erased_is_blanked_by_its_name(
    fake_dream_db: FakeDreamDb, chat: MagicMock
) -> None:
    """A dream write the cascade erased names its pass, which may cite
    nothing else the forget reached."""
    _seed(fake_dream_db, "p1", cites="fact-unrelated")

    await blank_dream_copies(
        _SCOPE,
        reached={_FORGOTTEN},
        dream_writes=[_episode_name("p1", "consolidate", 1)],
    )

    assert not any(_SENTENCE in text for text in _row_text(fake_dream_db, "p1"))
    chat.blank_dream_narratives.assert_awaited_once_with("u1", ["p1"], FORGOTTEN_FACT)


@pytest.mark.asyncio
async def test_a_pass_the_forget_did_not_reach_is_left_alone(
    fake_dream_db: FakeDreamDb, chat: MagicMock
) -> None:
    _seed(fake_dream_db, "p1", cites="fact-unrelated")
    _seed(fake_dream_db, "p2", scope=MemoryScope.for_expert("u1", "e1"))
    before = {pass_id: _row_text(fake_dream_db, pass_id) for pass_id in ("p1", "p2")}

    marked = await blank_dream_copies(_SCOPE, reached={_FORGOTTEN}, dream_writes=[])

    assert marked == []
    assert {p: _row_text(fake_dream_db, p) for p in ("p1", "p2")} == before
    assert all(fake_dream_db.record(p).erased_at is None for p in ("p1", "p2"))
    chat.blank_dream_narratives.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        DreamPassStatus.QUEUED,
        DreamPassStatus.RUNNING,
        DreamPassStatus.SUBMITTED,
        DreamPassStatus.APPLYING,
    ],
)
async def test_every_running_pass_of_the_scope_is_marked_and_blanked_as_it_ends(
    fake_dream_db: FakeDreamDb, chat: MagicMock, status: DreamPassStatus
) -> None:
    """Its next phase reads its row, and one that gathered its input before
    the forget may not name what it read yet: it is marked whether or not its
    row names the forget, its text kept until it closes, and its narrative
    (one apply may have written already) blanked now with the closed pass's
    beside it."""
    _seed(fake_dream_db, "p-cites", status=status)
    _seed(fake_dream_db, "p-names-nothing", cites="fact-unrelated", status=status)
    _seed(fake_dream_db, "p-closed")
    before = _row_text(fake_dream_db, "p-cites")

    marked = await blank_dream_copies(
        _SCOPE,
        reached={_FORGOTTEN},
        dream_writes=[_episode_name("p-cites", "recombine", 2)],
    )

    assert sorted(marked) == ["p-cites", "p-closed", "p-names-nothing"]
    assert _row_text(fake_dream_db, "p-cites") == before
    assert not any(_SENTENCE in t for t in _row_text(fake_dream_db, "p-closed"))
    chat.blank_dream_narratives.assert_awaited_once_with(
        "u1", ["p-cites", "p-closed", "p-names-nothing"], FORGOTTEN_FACT
    )

    await record_sync_outcome(
        _result("p-names-nothing", summary_for_user=f"I noticed {_SENTENCE}.")
    )

    assert not any(_SENTENCE in t for t in _row_text(fake_dream_db, "p-names-nothing"))
    assert _row_text(fake_dream_db, "p-cites") == before, "still running"


def _result(pass_id: str, **fields: Any) -> DreamPassResult:
    return DreamPassResult(user_id="u1", pass_id=pass_id, **fields)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "close",
    [
        "sync_outcome",
        "sync_failure",
        "batch_complete",
        "batch_failure",
        "cancel",
        "expiry",
    ],
)
async def test_every_way_a_marked_pass_closes_blanks_it(
    fake_dream_db: FakeDreamDb, chat: MagicMock, close: str
) -> None:
    """Whichever transition closes a marked pass blanks its text, what it
    writes as it closes included, and keeps only the code of its error."""
    _seed(fake_dream_db, "p1", status=DreamPassStatus.APPLYING)
    await blank_dream_copies(_SCOPE, reached={_FORGOTTEN}, dream_writes=[])
    said = f"I noticed {_SENTENCE}."
    error = f"recombine: non-JSON — {_SENTENCE}"

    match close:
        case "sync_outcome":
            await record_sync_outcome(_result("p1", summary_for_user=said))
        case "sync_failure":
            await record_sync_outcome(_result("p1", error=error))
        case "batch_complete":
            await record_batch_complete(_result("p1", summary_for_user=said), None)
        case "batch_failure":
            assert await write_stop("p1", failed(error, None, _NOW, marked=True))
        case "cancel":
            assert await write_stop(
                "p1", cancelled("owner cancelled", owner_user_id="u1")
            )
        case "expiry":
            assert await write_stop("p1", expired("superseded", not_updated_since=None))

    row = fake_dream_db.record("p1")
    assert row.status not in (DreamPassStatus.APPLYING,)
    assert not any(_SENTENCE in text for text in _row_text(fake_dream_db, "p1"))
    assert _SENTENCE not in (row.error or "")


@pytest.mark.asyncio
async def test_the_cleanups_end_blanks_a_marked_pass_again(
    fake_dream_db: FakeDreamDb, chat: MagicMock
) -> None:
    """The one write a closed row takes blanks it too, so a pass marked in
    the instant it closed is blanked by the cleanup after it at the latest."""
    _seed(fake_dream_db, "p1", status=DreamPassStatus.COMPLETE, cleanup_pending_at=_NOW)
    fake_dream_db.rows["p1"]["erased_at"] = _NOW

    assert await fake_dream_db.update_dream_pass("p1", cleanup_finished())

    assert not any(_SENTENCE in text for text in _row_text(fake_dream_db, "p1"))


@pytest.mark.asyncio
async def test_a_pass_with_no_mark_closes_as_it_did(
    fake_dream_db: FakeDreamDb, chat: MagicMock
) -> None:
    _seed(fake_dream_db, "p1", status=DreamPassStatus.APPLYING)
    said = f"I noticed {_SENTENCE}."

    await record_sync_outcome(_result("p1", summary_for_user=said))

    record = fake_dream_db.record("p1")
    assert record.operations.applied is not None
    assert record.operations.applied.summary_for_user == said


@pytest.mark.asyncio
async def test_a_closed_pass_with_its_cleanup_pending_is_marked_and_blanked(
    fake_dream_db: FakeDreamDb, chat: MagicMock
) -> None:
    """A stop from outside closes the row while the pass may still be
    applying, and may write the narrative after the forget: it is marked
    however little its row names."""
    _seed(
        fake_dream_db,
        "p1",
        cites="fact-unrelated",
        status=DreamPassStatus.CANCELLED,
        cleanup_pending_at=_NOW,
    )

    assert await blank_dream_copies(_SCOPE, reached={_FORGOTTEN}, dream_writes=[]) == [
        "p1"
    ]
    assert not any(_SENTENCE in text for text in _row_text(fake_dream_db, "p1"))


@pytest.mark.asyncio
async def test_every_failed_pass_of_the_scope_keeps_only_its_errors_code(
    fake_dream_db: FakeDreamDb, chat: MagicMock
) -> None:
    """An error written before errors were codes may quote the model, and no
    uuid finds it: every failed pass of the scope loses all but its code,
    reached or not. An error that is a code alone stays."""
    _seed(
        fake_dream_db, "p-old", cites="fact-unrelated", status=DreamPassStatus.ERRORED
    )
    fake_dream_db.rows["p-old"][
        "error"
    ] = f"recombine: LLM returned non-JSON content — first 200 chars: {_SENTENCE}"
    _seed(
        fake_dream_db, "p-new", cites="fact-unrelated", status=DreamPassStatus.ERRORED
    )
    fake_dream_db.rows["p-new"]["error"] = "consolidate: schema_mismatch"

    await blank_dream_copies(_SCOPE, reached={_FORGOTTEN}, dream_writes=[])

    assert fake_dream_db.record("p-old").error == f"recombine: {ERASED_TEXT}"
    assert fake_dream_db.record("p-new").error == "consolidate: schema_mismatch"
    assert fake_dream_db.record("p-old").erased_at is None, "not reached"


@pytest.mark.asyncio
async def test_every_pass_of_the_scope_is_looked_at(
    fake_dream_db: FakeDreamDb, chat: MagicMock
) -> None:
    """The review's 1,001 passes: the oldest, the one the forget reached,
    is behind a thousand newer ones, and no page or limit leaves it out."""
    _seed(fake_dream_db, "p-old")
    for index in range(1000):
        _seed(fake_dream_db, f"p-{index:04d}", cites="fact-unrelated")

    marked = await blank_dream_copies(_SCOPE, reached={_FORGOTTEN}, dream_writes=[])

    assert marked == ["p-old"]
    assert not any(_SENTENCE in text for text in _row_text(fake_dream_db, "p-old"))


@pytest.mark.asyncio
async def test_a_pass_whose_row_aged_out_keeps_only_its_narrative_to_blank(
    fake_dream_db: FakeDreamDb, chat: MagicMock
) -> None:
    await blank_dream_copies(
        _SCOPE,
        reached={_FORGOTTEN},
        dream_writes=[_episode_name("p-gone", "consolidate", 1)],
    )

    chat.blank_dream_narratives.assert_awaited_once_with(
        "u1", ["p-gone"], FORGOTTEN_FACT
    )


@pytest.mark.asyncio
async def test_a_failed_write_raises(
    fake_dream_db: FakeDreamDb, chat: MagicMock
) -> None:
    fake_dream_db.fail = True

    with pytest.raises(ConnectionError):
        await blank_dream_copies(_SCOPE, reached={_FORGOTTEN}, dream_writes=[])
    chat.blank_dream_narratives.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_narrative_write_that_hangs_runs_out_of_time(
    fake_dream_db: FakeDreamDb, chat: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forget_copies, "NARRATIVE_TIMEOUT_SECONDS", 0.05)
    chat.blank_dream_narratives = AsyncMock(side_effect=_hang)
    _seed(fake_dream_db, "p1")

    with pytest.raises(TimeoutError):
        await blank_dream_copies(_SCOPE, reached={_FORGOTTEN}, dream_writes=[])


async def _hang(*args: object) -> None:
    await asyncio.sleep(5)


@pytest.mark.parametrize(
    "name, pass_id",
    [
        (_episode_name("p-1", "consolidate", 1), "p-1"),
        (_episode_name("0198c5a2-7f5e", "recombine", 12), "0198c5a2-7f5e"),
        ("conversation_s-1", None),
        ("finding_s-1", None),
        ("dream_p-1_sanitize_001", None),
        ("dream_p-1_consolidate_1", None),
    ],
)
def test_a_dream_writes_episode_names_its_pass(name: str, pass_id: str | None) -> None:
    assert pass_of(name) == pass_id
