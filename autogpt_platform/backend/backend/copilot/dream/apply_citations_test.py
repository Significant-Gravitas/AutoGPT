"""apply hands every dream write what it rests on, for the ingestion worker
to check against forgets right before writing it
(``graphiti/recall_citations.py``), and reports the writes it dropped.

The live runs, a forget between the dream's read and its writes, are
``graphiti/recall_dream_citations_integration_test.py``.
"""

from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.ingest import IngestionCompletion
from backend.copilot.graphiti.recall_citations import Citations
from backend.copilot.graphiti.scope import MemoryScope

from . import apply as apply_mod
from .schemas import ConsolidatedFact, DreamOperations, ProposedFinding

_SCOPE = MemoryScope.for_user("u-1234567890ab")


@pytest.fixture(autouse=True)
def enqueue(mocker) -> AsyncMock:
    queued = AsyncMock(return_value=True)
    mocker.patch.object(apply_mod, "enqueue_episode", queued)
    mocker.patch.object(apply_mod, "wait_for_ingestion", AsyncMock(return_value=True))
    mocker.patch.object(apply_mod, "_create_dream_session", AsyncMock(return_value="s"))
    mocker.patch.object(apply_mod, "_write_dream_summary_message", AsyncMock())
    return queued


def _cited(enqueue: AsyncMock) -> list[Citations]:
    return [call.kwargs["citations"] for call in enqueue.await_args_list]


@pytest.mark.asyncio
async def test_each_write_carries_what_it_cites(enqueue: AsyncMock) -> None:
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(
                content="Alice works on Atlas",
                confidence=0.9,
                source_episode_uuids=["ep1"],
            )
        ],
        proposals=[
            ProposedFinding(
                content="Alice may lead Atlas",
                confidence=0.5,
                rationale="implied",
                source_fact_uuids=["f1"],
                source_episode_uuids=["ep2"],
            )
        ],
    )

    await apply_mod.apply_operations(
        _SCOPE, "p1", ops, known_fact_uuids={"f1", "f9"}, known_episode_uuids={"ep1"}
    )

    assert _cited(enqueue) == [
        Citations(episode_uuids=["ep1"]),
        Citations(fact_uuids=["f1"], episode_uuids=["ep2"]),
    ]


@pytest.mark.asyncio
async def test_a_write_citing_nothing_rests_on_everything_the_pass_read(
    enqueue: AsyncMock,
) -> None:
    ops = DreamOperations(
        writes=[ConsolidatedFact(content="Alice works on Atlas", confidence=0.9)],
        proposals=[
            ProposedFinding(content="Alice leads", confidence=0.5, rationale="r")
        ],
    )

    await apply_mod.apply_operations(
        _SCOPE, "p1", ops, known_fact_uuids={"f2", "f1"}, known_episode_uuids={"ep1"}
    )

    assert _cited(enqueue) == [
        Citations(
            fact_uuids=["f1", "f2"],
            episode_uuids=["ep1"],
            statement="Alice works on Atlas",
        ),
        Citations(
            fact_uuids=["f1", "f2"], episode_uuids=["ep1"], statement="Alice leads"
        ),
    ]


@pytest.mark.asyncio
async def test_without_what_the_pass_read_it_is_checked_by_its_statement(
    enqueue: AsyncMock,
) -> None:
    fact = ConsolidatedFact(content="Alice works on Atlas", confidence=0.9)

    await apply_mod._write_consolidated_fact(
        _SCOPE, "p1", 0, fact, "s", IngestionCompletion()
    )

    assert _cited(enqueue) == [Citations(statement="Alice works on Atlas")]


@pytest.mark.asyncio
async def test_the_writes_the_worker_dropped_are_reported(mocker) -> None:
    async def worker_dropped_one(completion: IngestionCompletion, _: float) -> bool:
        completion.dropped_forgotten += 1
        return True

    mocker.patch.object(apply_mod, "wait_for_ingestion", worker_dropped_one)
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(content="one", confidence=0.9),
            ConsolidatedFact(content="two", confidence=0.9),
        ]
    )

    stats = await apply_mod.apply_operations(_SCOPE, "p1", ops)

    assert stats["consolidated_count"] == 2
    assert stats["dropped_forgotten"] == 1


@pytest.mark.asyncio
async def test_an_empty_pass_drops_nothing() -> None:
    stats = await apply_mod.apply_operations(_SCOPE, "p1", DreamOperations())

    assert stats["dropped_forgotten"] == 0
