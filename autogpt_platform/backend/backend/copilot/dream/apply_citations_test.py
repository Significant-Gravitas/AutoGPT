"""apply checks every dream write's citations against what its pass read,
drops and counts one left citing nothing, and hands the rest, with what they
cite, to the ingestion worker, which checks them against forgets right before
writing (``graphiti/recall_citations.py``) and records them
(``graphiti/recall_derivation.py``).

The live runs are ``graphiti/recall_dream_citations_integration_test.py``
(a forget between the dream's read and its writes) and
``graphiti/recall_cascade_integration_test.py`` (a forget after them).
"""

from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.ingest import IngestionCompletion
from backend.copilot.graphiti.migrations.legacy_citations import described_citations
from backend.copilot.graphiti.recall_citations import Citations
from backend.copilot.graphiti.scope import MemoryScope

from . import apply as apply_mod
from .citations import UNSCOPED, source_description, validated_citations
from .schemas import ConsolidatedFact, DreamOperations, ProposedFinding

_SCOPE = MemoryScope.for_user("u-1234567890ab")
_READ = {"known_fact_uuids": {"f1", "f2"}, "known_episode_uuids": {"ep1", "ep2"}}


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


def _checked(
    facts: list[str], episodes: list[str], **known: set[str]
) -> Citations | None:
    """``validated_citations`` for a write in the default scope, every
    source unscoped."""
    return validated_citations(
        facts,
        episodes,
        scope=UNSCOPED,
        known_facts=known["known_facts"],
        known_episodes=known["known_episodes"],
        source_scopes={},
    ).citations


class TestValidatedCitations:
    def test_keeps_what_the_pass_read_in_the_models_order_once_each(self) -> None:
        cited = _checked(
            ["f2", "made-up", "f1", "f2"],
            ["ep2", "ep1", "ep2"],
            known_facts={"f1", "f2"},
            known_episodes={"ep1", "ep2"},
        )

        assert cited == Citations(fact_uuids=["f2", "f1"], episode_uuids=["ep2", "ep1"])

    def test_files_a_uuid_under_the_kind_the_pass_read_it_as(self) -> None:
        cited = _checked(["ep1"], ["f1"], known_facts={"f1"}, known_episodes={"ep1"})

        assert cited == Citations(fact_uuids=["f1"], episode_uuids=["ep1"])

    @pytest.mark.parametrize(
        "facts, episodes", [([], []), (["made-up"], ["also-made-up"])]
    )
    def test_nothing_the_pass_read_is_none(
        self, facts: list[str], episodes: list[str]
    ) -> None:
        cited = _checked(facts, episodes, known_facts={"f1"}, known_episodes={"ep1"})

        assert cited is None


@pytest.mark.asyncio
async def test_each_write_carries_what_it_cites_of_what_its_pass_read(
    enqueue: AsyncMock,
) -> None:
    """A consolidation cites facts now, as a proposal does; a made-up uuid
    is dropped and no write carries a statement to compare."""
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(
                content="Alice works on Atlas",
                confidence=0.9,
                source_episode_uuids=["ep1", "made-up"],
                source_fact_uuids=["f1"],
            )
        ],
        proposals=[
            ProposedFinding(
                content="Alice may lead Atlas",
                confidence=0.5,
                rationale="implied",
                source_fact_uuids=["f2", "made-up"],
                source_episode_uuids=["ep2"],
            )
        ],
    )

    stats = await apply_mod.apply_operations(_SCOPE, "p1", ops, **_READ)

    assert _cited(enqueue) == [
        Citations(fact_uuids=["f1"], episode_uuids=["ep1"]),
        Citations(fact_uuids=["f2"], episode_uuids=["ep2"]),
    ]
    assert stats["uncited_writes_dropped"] == 0


@pytest.mark.asyncio
async def test_writes_citing_nothing_the_pass_read_are_dropped_and_counted(
    enqueue: AsyncMock,
) -> None:
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(content="cites nothing", confidence=0.9),
            ConsolidatedFact(
                content="cites only made-up uuids",
                confidence=0.9,
                source_episode_uuids=["made-up"],
                source_fact_uuids=["also-made-up"],
            ),
            ConsolidatedFact(
                content="Alice works on Atlas",
                confidence=0.9,
                source_episode_uuids=["ep1"],
            ),
        ],
        proposals=[ProposedFinding(content="a hunch", confidence=0.5, rationale="r")],
    )

    stats = await apply_mod.apply_operations(_SCOPE, "p1", ops, **_READ)

    assert [call.kwargs["name"] for call in enqueue.await_args_list] == [
        "dream_p1_consolidate_002"
    ]
    assert stats["consolidated_count"] == 1
    assert stats["proposal_count"] == 0
    assert stats["uncited_writes_dropped"] == 3
    snapshot = stats["snapshot"]
    assert [w.content for w in snapshot.writes] == ["Alice works on Atlas"]


@pytest.mark.asyncio
async def test_a_pass_left_with_only_uncited_writes_creates_no_session(
    enqueue: AsyncMock,
) -> None:
    ops = DreamOperations(
        writes=[ConsolidatedFact(content="cites nothing", confidence=0.9)],
        summary_for_user="I consolidated one fact.",
    )

    stats = await apply_mod.apply_operations(_SCOPE, "p1", ops, **_READ)

    enqueue.assert_not_awaited()
    apply_mod._create_dream_session.assert_not_awaited()
    assert "session_id" not in stats
    assert (stats["consolidated_count"], stats["uncited_writes_dropped"]) == (0, 1)


@pytest.mark.asyncio
async def test_without_what_the_pass_read_every_write_is_dropped(
    enqueue: AsyncMock,
) -> None:
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(
                content="Alice works on Atlas",
                confidence=0.9,
                source_episode_uuids=["ep1"],
            )
        ]
    )

    stats = await apply_mod.apply_operations(_SCOPE, "p1", ops)

    enqueue.assert_not_awaited()
    assert stats["uncited_writes_dropped"] == 1


@pytest.mark.asyncio
async def test_the_snapshot_and_the_episode_description_carry_the_checked_citations(
    enqueue: AsyncMock,
) -> None:
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(
                content="Alice works on Atlas",
                confidence=0.9,
                source_episode_uuids=["ep1", "made-up"],
                source_fact_uuids=["f1"],
            )
        ],
        proposals=[
            ProposedFinding(
                content="Alice may lead Atlas",
                confidence=0.5,
                rationale="she runs the standups",
                source_fact_uuids=["f2"],
            )
        ],
    )

    stats = await apply_mod.apply_operations(_SCOPE, "p1", ops, **_READ)

    write, proposal = stats["snapshot"].writes[0], stats["snapshot"].proposals[0]
    assert (write.source_fact_uuids, write.source_episode_uuids) == (["f1"], ["ep1"])
    assert (proposal.source_fact_uuids, proposal.source_episode_uuids) == (["f2"], [])
    descriptions = [c.kwargs["source_description"] for c in enqueue.await_args_list]
    assert descriptions == [
        "dream-pass consolidation",
        "dream-pass proposal; rationale=she runs the standups",
    ], "the marker and the records carry what it cites"


def test_a_rationale_cannot_forge_a_citation_in_the_description() -> None:
    """Codex's probe: a rationale holding the delimiter and a key. Its
    ``;`` is written as ``,``, so the backfill's reader finds a key where
    the dream never puts one and attributes nothing."""
    description = source_description(
        "proposal", rationale="ordinary; src_facts=11111111-2222"
    )

    assert description == (
        "dream-pass proposal; rationale=ordinary, src_facts=11111111-2222"
    )
    read_back = described_citations(description)
    assert (read_back.facts, read_back.ambiguous) == ([], True)


@pytest.mark.asyncio
async def test_the_writes_the_worker_dropped_are_reported(mocker) -> None:
    async def worker_dropped_one(completion: IngestionCompletion, _: float) -> bool:
        completion.dropped_forgotten += 1
        return True

    mocker.patch.object(apply_mod, "wait_for_ingestion", worker_dropped_one)
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(content="one", confidence=0.9, source_fact_uuids=["f1"]),
            ConsolidatedFact(content="two", confidence=0.9, source_fact_uuids=["f2"]),
        ]
    )

    stats = await apply_mod.apply_operations(_SCOPE, "p1", ops, **_READ)

    assert stats["consolidated_count"] == 2
    assert stats["dropped_forgotten"] == 1


@pytest.mark.asyncio
async def test_an_empty_pass_drops_nothing() -> None:
    stats = await apply_mod.apply_operations(_SCOPE, "p1", DreamOperations())

    assert (stats["dropped_forgotten"], stats["uncited_writes_dropped"]) == (0, 0)
