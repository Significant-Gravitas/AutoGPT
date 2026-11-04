"""Dream writes whose writer crashed or whose ``add_episode`` raised, on a
live FalkorDB (``recall_reconcile.py``, ``migrations/dream_markers.py``).

- A crashed writer's placed episode is never recalled; its marker holds up
  a forget of what it cites (``cleanup_error``) for a day, then expires,
  and the operator command deletes it with the episode.
- A write whose ``add_episode`` raised marks its marker aborted; reconcile
  completes it when graphiti saved the episode, even in part, and
  otherwise drops it with the placed episode (through the production
  worker, its model failing once the episode is placed).

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_marker_crash_integration_test.py
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from backend.copilot.dream import apply
from backend.copilot.dream.schemas import ConsolidatedFact, DreamOperations

from .falkordb_driver import AutoGPTFalkorDriver
from .marked_write import placed
from .memory_model import MemoryForgetFailureCode
from .migrations import dream_markers
from .recall import forgotten_facts_clause, recallable_episode_predicate
from .recall_cascade_fixtures import FLOUR, SUPPLIES, derivation, dream, gather
from .recall_citations import Citations
from .recall_derivation import MARKER_LABEL, abort, mark
from .recall_forget import retract
from .recall_integration_fixtures import (
    BuildClient,
    episode_row,
    ingest_facts,
    model_of,
    patch_recall_boundaries,
    rows,
    stop_ingestion_workers,
)
from .recall_marker_fixtures import age, land, markers, placed_payload, seed_root
from .recall_reconcile import Reconciled, reconcile


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_crashed_writer_holds_forgets_up_for_a_day_then_is_resolved(
    scope_graph,
) -> None:
    driver, scope = scope_graph
    gid = scope.group_id
    await seed_root(driver, gid, "x-root")
    marker = await mark(
        driver, gid, "x-episode", "dream_crash", Citations(fact_uuids=["x-root"])
    )
    await placed(driver, gid, placed_payload("dream_crash"), "x-episode")
    [shown] = await rows(
        driver,
        forgotten_facts_clause() + "MATCH (ep:Episodic {uuid: 'x-episode'}) "
        f"RETURN {recallable_episode_predicate('ep')} AS recallable",
    )

    first = await retract(scope, ["x-root"])
    waiting = await reconcile(driver, gid)
    await age(driver, marker)
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
    assert await markers(driver) == []
    assert await episode_row(driver, "x-episode") == {}, "deleted with its marker"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_an_aborted_write_graphiti_saved_in_part_is_completed(
    scope_graph,
) -> None:
    driver, scope = scope_graph
    gid, cited = scope.group_id, Citations(fact_uuids=["s-root"])
    await seed_root(driver, gid, "s-root")
    marker = await mark(driver, gid, "s-episode", "dream_saved", cited)
    await land(
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
    assert await markers(driver) == []


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
    assert await markers(driver) == []
    assert await episode_row(driver, left["episode"]) == {}
