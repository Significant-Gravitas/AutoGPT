"""The dream citation marker on a live FalkorDB (``recall_derivation.py``,
``recall_reconcile.py``, ``recall_landing.py``, ``migrations/dream_markers.py``):
Codex's second-pass attacks on it, as regressions.

- A marker names its write's episode by uuid: a user's episode sharing the
  dream episode's name is never stamped, and a hard forget of what the
  dream cited leaves it whole.
- A pending marker is never deleted for its age: past a day it expires, no
  longer holds forgets up, and is kept; a write landing after that is
  settled on landing, by its writer or by the next reconcile: retracted
  and, its root purged, erased. So is one settling while a hard forget is
  still purging its root.
- A write whose marker an operator resolved is settled by its writer when
  it lands.
- A crashed writer's placed episode is never recalled; its marker holds up
  a forget of what it cites (``cleanup_error``) for a day, then expires,
  and the operator command deletes it with the episode.
- A write whose ``add_episode`` raised marks its marker aborted; reconcile
  drops it with the placed episode, or completes it when graphiti saved the
  episode after all.

The young race, a forget while a write is in flight where the lock does not
hold, is ``recall_marker_race_integration_test.py``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_marker_integration_test.py
"""

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from graphiti_core.nodes import EpisodeType
from pytest_mock import MockerFixture

from backend.copilot.dream import apply
from backend.copilot.dream.schemas import ConsolidatedFact, DreamOperations

from . import recall_forget
from .falkordb_driver import AutoGPTFalkorDriver
from .marked_write import placed
from .memory_model import MemoryForgetFailureCode
from .migrations import dream_markers
from .recall import FORGOTTEN_FACT, forgotten_facts_clause, recallable_episode_predicate
from .recall_cascade_fixtures import FLOUR, SUPPLIES, derivation, dream, gather
from .recall_cascade_walk import derived_reason
from .recall_citations import Citations
from .recall_derivation import MARKER_LABEL, abort, mark, record
from .recall_forget import retract
from .recall_integration_fixtures import (
    BuildClient,
    edge_row,
    episode_row,
    ingest_facts,
    live_facts,
    model_of,
    patch_recall_boundaries,
    rows,
    stop_ingestion_workers,
)
from .recall_reconcile import Reconciled, reconcile

_LONG_AGO = "2000-01-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


async def _seed_root(driver: AutoGPTFalkorDriver, group_id: str, root: str) -> None:
    """A user's fact ``root`` and the episode that stated it."""
    await driver.execute_query(
        """
        CREATE (a:Entity {uuid: $root + '-a', name: 'Root A', group_id: $gid}),
               (b:Entity {uuid: $root + '-b', name: 'Root B', group_id: $gid}),
               (:Episodic {uuid: $root + '-episode', name: 'user turn',
                           group_id: $gid, content: 'The root source sentence',
                           entity_edges: [$root]}),
               (a)-[:RELATES_TO {uuid: $root, group_id: $gid,
                                 fact: 'The root source sentence', name: 'rel',
                                 status: 'active', scope: 'real:global',
                                 episodes: [$root + '-episode'], created_at: $now}]->(b)
        """,
        gid=group_id,
        root=root,
        now=datetime.now(timezone.utc).isoformat(),
    )


async def _land(
    driver: AutoGPTFalkorDriver,
    group_id: str,
    *,
    episode: str,
    name: str,
    edge: str,
    content: str,
    missing: tuple[str, ...] = (),
) -> None:
    """A write landing as graphiti saves one: its episode saved over by
    uuid (every property replaced), then the fact it produced. ``missing``
    are facts the episode lists that were never saved."""
    await driver.execute_query(
        """
        MERGE (ep:Episodic {uuid: $episode})
        SET ep = {uuid: $episode, name: $name, group_id: $gid, content: $content,
                  source: 'json', source_description: 'dream-pass consolidation',
                  entity_edges: [$edge] + $missing}
        CREATE (a:Entity {uuid: $edge + '-a', name: $edge + ' A', group_id: $gid}),
               (b:Entity {uuid: $edge + '-b', name: $edge + ' B', group_id: $gid}),
               (a)-[:RELATES_TO {uuid: $edge, group_id: $gid, fact: $content,
                                 name: 'derived', status: 'active',
                                 scope: 'real:global', episodes: [$episode],
                                 created_at: $now, fact_embedding: [0.1, 0.2]}]->(b)
        """,
        gid=group_id,
        episode=episode,
        name=name,
        edge=edge,
        content=content,
        missing=list(missing),
        now=datetime.now(timezone.utc).isoformat(),
    )


def _payload(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "episode_body": '{"content": "A dream fact"}',
        "source": EpisodeType.json,
        "source_description": "dream-pass consolidation",
        "reference_time": datetime.now(timezone.utc),
    }


async def _markers(driver: AutoGPTFalkorDriver) -> list[dict[str, Any]]:
    return await rows(
        driver,
        f"MATCH (m:{MARKER_LABEL}) RETURN m.uuid AS uuid, m.state AS state "
        "ORDER BY uuid",
    )


async def _age(driver: AutoGPTFalkorDriver, marker: str) -> None:
    await driver.execute_query(
        f"MATCH (m:{MARKER_LABEL} {{uuid: $uuid}}) SET m.created_at = $long_ago",
        uuid=marker,
        long_ago=_LONG_AGO,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_user_episode_named_like_the_dream_episode_is_never_touched(
    scope_graph,
) -> None:
    """Codex's collision: both were stamped, then both bodies erased."""
    driver, scope = scope_graph
    gid, shared = scope.group_id, "dream_same_name"
    await _seed_root(driver, gid, "c-root")
    user_says = "A sentence supplied independently by the user"
    await _land(
        driver,
        gid,
        episode="c-user",
        name=shared,
        edge="c-user-fact",
        content=user_says,
    )
    await _land(
        driver,
        gid,
        episode="c-dream",
        name=shared,
        edge="c-dream-fact",
        content="Dream",
    )
    await mark(driver, gid, "c-dream", shared, Citations(fact_uuids=["c-root"]))

    reconciled = await reconcile(driver, gid)
    result = await retract(scope, ["c-root"], hard=True)

    assert reconciled == Reconciled(completed=1)
    assert (await derivation(driver, "c-user"))["facts"] is None, "never stamped"
    assert (await derivation(driver, "c-dream"))["facts"] == ["c-root"]
    assert (result.failures, result.derived) == ([], ["c-dream-fact"])
    assert (await episode_row(driver, "c-user"))["content"] == user_says
    assert (await episode_row(driver, "c-dream"))["content"] == ""
    assert "c-user-fact" in await live_facts(driver)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("settled_by", ["writer", "reconcile"])
async def test_an_old_marker_is_kept_and_its_late_write_settled_on_landing(
    scope_graph, settled_by: str
) -> None:
    """Codex's slow write: its marker was deleted after an hour, and the
    write that landed later kept a live fact no forget could reach."""
    driver, scope = scope_graph
    gid, cited = scope.group_id, Citations(fact_uuids=["o-root"])
    await _seed_root(driver, gid, "o-root")
    marker = await mark(driver, gid, "o-episode", "dream_old", cited)
    await _age(driver, marker)

    expired = await reconcile(driver, gid)
    forgotten = await retract(scope, ["o-root"], hard=True)
    kept = await _markers(driver)
    await _land(
        driver,
        gid,
        episode="o-episode",
        name="dream_old",
        edge="o-fact",
        content="Late",
    )
    if settled_by == "writer":
        assert await record(driver, gid, marker, "o-episode", ["o-fact"], cited)
    else:
        assert await reconcile(driver, gid) == Reconciled(completed=1)

    assert expired == Reconciled(expired=1)
    assert forgotten.failures == [], "an expired marker holds no forget up"
    assert kept == [{"uuid": marker, "state": "expired"}], "never deleted for age"
    assert "o-fact" not in await live_facts(driver), "settled, before any retry"
    fact = await edge_row(driver, "o-fact")
    assert (fact["status"], fact["reason"]) == ("retracted", derived_reason("o-root"))
    assert (fact["fact"], fact["fact_redacted"]) == (FORGOTTEN_FACT, ""), "erased"
    assert (await episode_row(driver, "o-episode"))["content"] == ""
    assert await _markers(driver) == []
    retried = await retract(scope, ["o-root"], hard=True)
    assert (retried.failures, retried.resumed) == ([], ["o-root"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_write_settled_while_a_hard_forget_purges_is_erased(
    scope_graph,
) -> None:
    """Where the lock does not hold, a write can land and settle after a
    hard forget retracted its root and before it purged it: the root's
    ``hard_forgotten_at`` makes that settle erase, as one after the purge
    would."""
    driver, scope = scope_graph
    gid, cited = scope.group_id, Citations(fact_uuids=["h-root"])
    await _seed_root(driver, gid, "h-root")
    marker = await mark(driver, gid, "h-episode", "dream_purging", cited)
    purge = recall_forget.purge

    async def landing_then_purge(*args: Any) -> None:
        await _land(
            driver,
            gid,
            episode="h-episode",
            name="dream_purging",
            edge="h-fact",
            content="Late",
        )
        assert await record(driver, gid, marker, "h-episode", ["h-fact"], cited)
        await purge(*args)

    with patch.object(recall_forget, "purge", landing_then_purge):
        forgotten = await retract(scope, ["h-root"], hard=True)

    assert [f.code for f in forgotten.failures] == [
        MemoryForgetFailureCode.CLEANUP_ERROR
    ], "its marker was there when the forget looked"
    assert "h-fact" not in await live_facts(driver)
    fact = await edge_row(driver, "h-fact")
    assert (fact["fact"], fact["fact_redacted"]) == (FORGOTTEN_FACT, "")
    assert await edge_row(driver, "h-root") == {}, "purged all the same"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_write_whose_marker_was_resolved_is_settled_by_its_writer(
    scope_graph,
) -> None:
    driver, scope = scope_graph
    gid, cited = scope.group_id, Citations(fact_uuids=["r-root"])
    await _seed_root(driver, gid, "r-root")
    marker = await mark(driver, gid, "r-episode", "dream_resolved", cited)
    await placed(driver, gid, _payload("dream_resolved"), "r-episode")

    resolved = await dream_markers.resolve_graph(
        driver, gid, dream_markers.Selection(uuids={marker}), apply=True
    )
    forgotten = await retract(scope, ["r-root"])
    await _land(
        driver,
        gid,
        episode="r-episode",
        name="dream_resolved",
        edge="r-fact",
        content="Late",
    )
    recorded = await record(driver, gid, marker, "r-episode", ["r-fact"], cited)

    assert resolved == dream_markers.Resolved(deleted=1)
    assert forgotten.failures == []
    assert recorded, "settled from the citations its writer holds"
    assert "r-fact" not in await live_facts(driver)
    fact = await edge_row(driver, "r-fact")
    assert (fact["reason"], fact["fact_redacted"]) == (derived_reason("r-root"), "Late")
    assert (await derivation(driver, "r-fact"))["facts"] == ["r-root"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_crashed_writer_holds_forgets_up_for_a_day_then_is_resolved(
    scope_graph,
) -> None:
    driver, scope = scope_graph
    gid = scope.group_id
    await _seed_root(driver, gid, "x-root")
    marker = await mark(
        driver, gid, "x-episode", "dream_crash", Citations(fact_uuids=["x-root"])
    )
    await placed(driver, gid, _payload("dream_crash"), "x-episode")
    [shown] = await rows(
        driver,
        forgotten_facts_clause() + "MATCH (ep:Episodic {uuid: 'x-episode'}) "
        f"RETURN {recallable_episode_predicate('ep')} AS recallable",
    )

    first = await retract(scope, ["x-root"])
    waiting = await reconcile(driver, gid)
    await _age(driver, marker)
    expired = await reconcile(driver, gid)
    second = await retract(scope, ["x-root"])
    listed = await dream_markers.list_markers(driver, gid)
    resolved = await dream_markers.resolve_graph(
        driver, gid, dream_markers.Selection(expired=True), apply=True
    )

    assert shown == {"recallable": False}, "a placed episode is never recalled"
    assert [(f.uuid, f.code) for f in first.failures] == [
        ("x-root", MemoryForgetFailureCode.CLEANUP_ERROR)
    ]
    assert "may still land" in first.failures[0].reason
    assert (waiting, expired) == (Reconciled(waiting=1), Reconciled(expired=1))
    assert second.failures == [], "past the bound it holds nothing up"
    assert [(m.uuid, m.state, m.saved, m.landed) for m in listed] == [
        (marker, "expired", False, False)
    ]
    assert resolved == dream_markers.Resolved(deleted=1)
    assert await _markers(driver) == []
    assert await episode_row(driver, "x-episode") == {}, "deleted with its marker"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_an_aborted_write_graphiti_saved_in_part_is_completed(
    scope_graph,
) -> None:
    driver, scope = scope_graph
    gid, cited = scope.group_id, Citations(fact_uuids=["s-root"])
    await _seed_root(driver, gid, "s-root")
    marker = await mark(driver, gid, "s-episode", "dream_saved", cited)
    await _land(
        driver,
        gid,
        episode="s-episode",
        name="dream_saved",
        edge="s-fact",
        content="Saved",
        missing=("s-never-saved",),
    )
    await abort(driver, marker)

    done = await reconcile(driver, gid)
    forgotten = await retract(scope, ["s-root"])

    assert done == Reconciled(completed=1), "an episode was saved: completed"
    assert (await derivation(driver, "s-fact"))["facts"] == ["s-root"]
    assert (forgotten.failures, forgotten.derived) == ([], ["s-fact"])
    assert await _markers(driver) == []


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


def _model_down(build: BuildClient, mocker: MockerFixture) -> BuildClient:
    """``build``, its model failing every call."""

    def built(driver: AutoGPTFalkorDriver, responses: dict[str, dict]) -> Any:
        client = build(driver, responses)
        failing = AsyncMock(side_effect=RuntimeError("model down"))
        mocker.patch.object(model_of(client), "_generate_response", failing)
        return client

    return built


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_write_whose_add_episode_raised_is_dropped_with_its_episode(
    scope_graph, stub_graphiti_client, dream_apply, mocker: MockerFixture
) -> None:
    """Through the worker: graphiti's model fails once the episode is placed."""
    driver, scope = scope_graph
    said, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [FLOUR], session_id="s-1"
    )
    write = ConsolidatedFact(
        content=SUPPLIES[2],
        confidence=0.9,
        source_fact_uuids=[edges[FLOUR[2]]],
        source_episode_uuids=[said],
    )
    read = await gather(scope)

    stats = await dream(
        driver,
        scope,
        _model_down(stub_graphiti_client, mocker),
        DreamOperations(writes=[write]),
        read,
        SUPPLIES,
        "p-abort",
    )
    [left] = await rows(
        driver,
        f"MATCH (m:{MARKER_LABEL}) MATCH (ep:Episodic {{uuid: m.episode_uuid}}) "
        "RETURN m.state AS state, ep.uuid AS episode, ep.write_pending AS pending",
    )
    done = await reconcile(driver, scope.group_id)

    assert (stats["failed_writes"], stats["provenance_pending"]) == (1, 0)
    assert (left["state"], left["pending"]) == ("aborted", True)
    assert done == Reconciled(dropped=1)
    assert await _markers(driver) == []
    assert await episode_row(driver, left["episode"]) == {}
