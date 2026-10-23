"""A hard forget whose cascade stopped short, resumed, on a live FalkorDB
through ``dream/apply.py`` and the production ingestion worker
(``recall_forget.py``, ``migrations/backfill_cascade.py``).

A hard forget purges the fact the user named even when its cascade stops
(a failed step, or a bound), so the fact is gone and repeating the forget
used to find nothing: Codex's finding. Now a repeated forget of the purged
uuid starts a cascade from whatever still names it (a record, an earlier
cascade's reason, the ``redacted_for`` of the episodes it hid), erasing,
and the backfill's ``--cascade-existing-forgets`` finds the purged roots and
the hidden episodes the dream cited. Each stop (before the first query,
between rounds, at the item bound and at the round bound) is resumed both
ways, until everything the dream derived from the bakery's flour supplier
is retracted and erased under that root's name. ``echo`` cites only the
user's chat turn, which the purge emptied: only ``redacted_for`` links it to
the forgotten fact. When that turn also stated another fact, the purge hides
it without emptying it, so no emptied episode carries the erasure: the
backfill must find the root gone from what names it and erase from there.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_cascade_resume_integration_test.py
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from backend.copilot.dream import apply
from backend.copilot.dream.schemas import ProposedFinding

from . import recall_cascade
from .falkordb_driver import AutoGPTFalkorDriver
from .migrations.backfill_derivations import backfill_graph
from .recall import FORGOTTEN_FACT
from .recall_cascade_fixtures import (
    BOULE_FLOUR,
    FLOUR,
    SUPPLIES,
    WEEKLY,
    Bakery,
    build_bakery,
    dream_episode,
    dream_once,
    fact_uuid,
)
from .recall_cascade_queries import DERIVED_FACTS_QUERY, EARLIER_QUERY
from .recall_forget import retract
from .recall_integration_fixtures import (
    Fact,
    edge_row,
    episode_row,
    live_facts,
    patch_recall_boundaries,
    rows,
    sentence_properties,
    stop_ingestion_workers,
)
from .scope import MemoryScope

_ECHO: Fact = (
    "Sunrise Bakery",
    "Flour Order",
    "Sunrise Bakery keeps a standing flour order",
)
_TRIES = 12


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest_asyncio.fixture(loop_scope="function")
async def dream_apply(mocker: MockerFixture) -> AsyncIterator[None]:
    """apply's Postgres side (the dream session and its summary) stubbed."""
    mocker.patch.object(apply, "_create_dream_session", AsyncMock(return_value="s"))
    mocker.patch.object(apply, "_write_dream_summary_message", AsyncMock())
    mocker.patch(
        "backend.copilot.dream.registry.ensure_dream_system_scheduled",
        AsyncMock(return_value=None),
    )
    yield
    await stop_ingestion_workers()


@contextmanager
def _failing_once(query: str, nth: int) -> Iterator[None]:
    """The ``nth`` run of ``query`` raises; every other run goes through."""
    original = AutoGPTFalkorDriver.execute_query
    seen = [0]

    async def failing(self, cypher_query_, **params):
        if cypher_query_ == query:
            seen[0] += 1
            if seen[0] == nth:
                raise RuntimeError("injected cascade failure")
        return await original(self, cypher_query_, **params)

    with patch.object(AutoGPTFalkorDriver, "execute_query", failing):
        yield


@contextmanager
def _stop(point: str) -> Iterator[None]:
    """Where the first hard forget's cascade stops."""
    if point == "first_query":
        with _failing_once(EARLIER_QUERY, 1):
            yield
    elif point == "between_rounds":
        with _failing_once(DERIVED_FACTS_QUERY, 2):
            yield
    else:
        bound = "CASCADE_MAX_ITEMS" if point == "item_bound" else "CASCADE_MAX_ROUNDS"
        with patch.object(recall_cascade, bound, 1):
            yield


@contextmanager
def _bounded(point: str) -> Iterator[None]:
    """A bound stays in place while the forget is resumed; a failure does
    not happen again."""
    if point.endswith("_bound"):
        with _stop(point):
            yield
    else:
        yield


async def _with_echo(driver, scope: MemoryScope, build) -> tuple[Bakery, str, str]:
    """The bakery, plus a proposal citing only the user's chat turn."""
    bakery = await build_bakery(driver, scope, build)
    echo = ProposedFinding(
        content=_ECHO[2],
        confidence=0.6,
        rationale="the chat turn",
        source_episode_uuids=[bakery.said],
    )
    await dream_once(driver, scope, build, echo, _ECHO, "pass-echo")
    return (
        bakery,
        await fact_uuid(driver, _ECHO[2]),
        await dream_episode(driver, "pass-echo"),
    )


async def _resume(scope: MemoryScope, driver, root: str, recovery: str) -> list[dict]:
    """Forget again, or run the backfill, until nothing is left to resume;
    what each try reported."""
    tries: list[dict] = []
    for _ in range(_TRIES):
        if recovery == "forget_again":
            result = await retract(scope, [root], hard=True)
            tries.append({"resumed": result.resumed, "failed": bool(result.failures)})
        else:
            found = await backfill_graph(driver, apply=True, cascade_forgets=True)
            tries.append({"roots": found.roots, "failed": bool(found.failed)})
        if not tries[-1]["failed"]:
            return tries
    raise AssertionError(f"not finished after {_TRIES} tries: {tries}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["forget_again", "backfill"])
@pytest.mark.parametrize(
    "point", ["first_query", "between_rounds", "item_bound", "round_bound"]
)
async def test_a_hard_forget_that_stopped_short_is_resumed_and_erases_everything(
    scope_graph, stub_graphiti_client, dream_apply, point: str, recovery: str
) -> None:
    driver, scope = scope_graph
    bakery, echo, echo_episode = await _with_echo(driver, scope, stub_graphiti_client)
    derived = [*bakery.derived(), echo]

    with _stop(point):
        first = await retract(scope, [bakery.flour], hard=True)

    assert first.failures, "the cascade stopped short"
    assert await edge_row(driver, bakery.flour) == {}, "the root is purged anyway"
    assert set(await live_facts(driver)) - {bakery.boule, bakery.cafe}

    with _bounded(point):
        tries = await _resume(scope, driver, bakery.flour, recovery)

    if recovery == "forget_again":
        assert all(t["resumed"] == [bakery.flour] for t in tries)
    assert set(await live_facts(driver)) == {bakery.boule, bakery.cafe}
    for uuid in derived:
        row = await edge_row(driver, uuid)
        assert row["reason"] == f"derived_from_forgotten:{bakery.flour}", uuid
        assert (row["fact"], row["fact_redacted"]) == (FORGOTTEN_FACT, "")
    embeddings = await rows(
        driver,
        "MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $uuids "
        "AND e.fact_embedding IS NOT NULL RETURN e.uuid AS uuid",
        uuids=derived,
    )
    assert embeddings == []
    for sentence in (FLOUR[2], SUPPLIES[2], BOULE_FLOUR[2], WEEKLY[2], _ECHO[2]):
        assert await sentence_properties(driver, sentence) == set(), sentence
    episodes = [
        bakery.supplies_episode,
        bakery.boule_flour_episode,
        bakery.weekly_episode,
        echo_episode,
    ]
    bodies = await rows(
        driver,
        "MATCH (ep:Episodic) WHERE ep.uuid IN $uuids RETURN ep.content AS content",
        uuids=episodes,
    )
    assert [row["content"] for row in bodies] == [""] * len(episodes)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_backfill_erases_from_a_purged_root_whose_chat_turn_stays(
    scope_graph, stub_graphiti_client, dream_apply
) -> None:
    """The flour supplier's chat turn also named the cafe, so the hard forget
    hides the turn but keeps its text for the cafe and empties no episode:
    no hard seed carries the erasure. The backfill finds the purged root
    from the records and the ``redacted_for`` that name it, and erases
    every derived fact under its name."""
    driver, scope = scope_graph
    bakery, echo, _ = await _with_echo(driver, scope, stub_graphiti_client)
    await driver.execute_query(
        "MATCH (ep:Episodic {uuid: $said}), ()-[e:RELATES_TO {uuid: $cafe}]->() "
        "SET ep.entity_edges = ep.entity_edges + [$cafe], "
        "e.episodes = e.episodes + [$said]",
        said=bakery.said,
        cafe=bakery.cafe,
    )
    derived = [*bakery.derived(), echo]

    with _stop("first_query"):
        first = await retract(scope, [bakery.flour], hard=True)

    assert first.failures and await edge_row(driver, bakery.flour) == {}
    turn = await episode_row(driver, bakery.said)
    assert turn["redacted_at"] is not None and turn["hard_deleted_at"] is None

    tries = await _resume(scope, driver, bakery.flour, "backfill")

    assert tries[0]["roots"] >= 1
    assert set(await live_facts(driver)) == {bakery.boule, bakery.cafe}
    for uuid in derived:
        row = await edge_row(driver, uuid)
        assert row["reason"] == f"derived_from_forgotten:{bakery.flour}", uuid
        assert (row["fact"], row["fact_redacted"]) == (FORGOTTEN_FACT, ""), uuid
    for sentence in (SUPPLIES[2], BOULE_FLOUR[2], WEEKLY[2], _ECHO[2]):
        assert await sentence_properties(driver, sentence) == set(), sentence
