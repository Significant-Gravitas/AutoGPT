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

Crashed and aborted writers are ``recall_marker_crash_integration_test.py``;
the young race, a forget while a write is in flight where the lock does not
hold, is ``recall_marker_race_integration_test.py``.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/recall_marker_integration_test.py
"""

from typing import Any
from unittest.mock import patch

import pytest

from . import recall_forget
from .marked_write import placed
from .memory_model import MemoryForgetFailureCode
from .migrations import dream_markers
from .recall import FORGOTTEN_FACT
from .recall_cascade_fixtures import derivation
from .recall_cascade_walk import derived_reason
from .recall_citations import Citations
from .recall_derivation import mark, record
from .recall_forget import retract
from .recall_integration_fixtures import (
    edge_row,
    episode_row,
    live_facts,
    patch_recall_boundaries,
)
from .recall_marker_fixtures import age, land, markers, placed_payload, seed_root
from .recall_reconcile import Reconciled, reconcile


@pytest.fixture(autouse=True)
def boundaries(mocker, scope_graph, stub_graphiti_client):
    patch_recall_boundaries(mocker, scope_graph[0], stub_graphiti_client)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_user_episode_named_like_the_dream_episode_is_never_touched(
    scope_graph,
) -> None:
    """Codex's collision: both were stamped, then both bodies erased."""
    driver, scope = scope_graph
    gid, shared = scope.group_id, "dream_same_name"
    await seed_root(driver, gid, "c-root")
    user_says = "A sentence supplied independently by the user"
    await land(
        driver,
        gid,
        episode="c-user",
        name=shared,
        edge="c-user-fact",
        content=user_says,
    )
    await land(
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
    await seed_root(driver, gid, "o-root")
    marker = await mark(driver, gid, "o-episode", "dream_old", cited)
    await age(driver, marker)

    expired = await reconcile(driver, gid)
    forgotten = await retract(scope, ["o-root"], hard=True)
    kept = await markers(driver)
    await land(
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
    assert await markers(driver) == []
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
    await seed_root(driver, gid, "h-root")
    marker = await mark(driver, gid, "h-episode", "dream_purging", cited)
    purge = recall_forget.purge

    async def landing_then_purge(*args: Any) -> None:
        await land(
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
    await seed_root(driver, gid, "r-root")
    marker = await mark(driver, gid, "r-episode", "dream_resolved", cited)
    await placed(driver, gid, placed_payload("dream_resolved"), "r-episode")

    resolved = await dream_markers.resolve_graph(
        driver, gid, dream_markers.Selection(uuids={marker}), apply=True
    )
    forgotten = await retract(scope, ["r-root"])
    await land(
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
