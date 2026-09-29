"""Unit tests for ``recall_reconcile``: the dream citation markers a graph
still holds, each acted on as its state says. One whose write landed (found
by its episode's uuid, never its name) is recorded, settled and deleted; an
aborted one without a saved episode is deleted with the episode its writer
placed, the check and the delete in one statement; a pending one not landed
waits, then expires, and is never deleted for its age; an expired one is
kept. ``in_flight`` counts the writes that could still land. The reaper's sweep is in ``provenance_pending_test.py``;
the live runs are ``recall_provenance_integration_test.py`` and
``recall_marker_integration_test.py``.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import recall_reconcile
from .recall_citations import Citations

_NOW = datetime.now(timezone.utc)


def _marker(
    uuid: str,
    rank: int,
    *,
    state: str = "pending",
    age: timedelta = timedelta(0),
    saved: bool = False,
) -> dict:
    return {
        "uuid": uuid,
        "episode": f"episode-{uuid}",
        "facts": ["f1"],
        "episodes": ["ep1"],
        "created_at": (_NOW - age).isoformat(),
        "state": state,
        "saved": saved,
        "rank": rank,
    }


def _driver(*results) -> MagicMock:
    driver = MagicMock()
    driver.execute_query = AsyncMock(side_effect=[(r, [], None) for r in results])
    return driver


def _queries(driver: MagicMock) -> list[str]:
    return [call.args[0] for call in driver.execute_query.await_args_list]


class TestReconcile:
    @pytest.mark.asyncio
    async def test_a_graph_with_no_marker_reads_once(self) -> None:
        driver = _driver([])

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled()
        [read] = driver.execute_query.await_args_list
        assert read.args[0] == recall_reconcile.MARKERS_QUERY
        assert read.kwargs == {"group_id": "user_a", "limit": 201}

    @pytest.mark.asyncio
    async def test_a_landed_marker_is_recorded_settled_and_deleted_by_uuid(
        self,
    ) -> None:
        driver = _driver([_marker("m1", recall_reconcile.LANDED)], [{"uuid": "e1"}])
        record = AsyncMock(return_value=True)

        with patch.object(recall_reconcile, "record", record):
            done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled(completed=1)
        touched = driver.execute_query.await_args_list[1]
        assert touched.args[0] == recall_reconcile.TOUCHED_FACTS_QUERY
        assert touched.kwargs == {"episodes": ["episode-m1"]}
        record.assert_awaited_once_with(
            driver,
            "user_a",
            "m1",
            "episode-m1",
            ["e1"],
            Citations(fact_uuids=["f1"], episode_uuids=["ep1"]),
        )

    @pytest.mark.asyncio
    async def test_one_whose_record_or_settle_stops_short_is_kept(self) -> None:
        driver = _driver([_marker("m1", recall_reconcile.LANDED)], [])

        with patch.object(recall_reconcile, "record", AsyncMock(return_value=False)):
            done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled(unsettled=1)
        assert done.incomplete()

    @pytest.mark.asyncio
    async def test_an_aborted_one_whose_write_never_landed_is_dropped(self) -> None:
        """With the episode its writer placed, when graphiti never saved it."""
        driver = _driver(
            [_marker("m1", recall_reconcile.DROPPABLE, state="aborted")],
            [{"dropped": 1}],
        )

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled(dropped=1)
        drop = driver.execute_query.await_args_list[1]
        assert (drop.args[0], drop.kwargs) == (
            recall_reconcile.DROP_UNLANDED_QUERY,
            {"uuid": "m1"},
        )

    @pytest.mark.asyncio
    async def test_an_aborted_one_whose_episode_was_saved_since_is_kept(
        self,
    ) -> None:
        """The drop tests it again in its own statement: an episode graphiti
        saved after the read keeps the marker, for the next reconcile to
        complete."""
        driver = _driver(
            [_marker("m1", recall_reconcile.DROPPABLE, state="aborted")],
            [{"dropped": 0}],
        )

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled(waiting=1)

    @pytest.mark.asyncio
    async def test_a_pending_one_not_landed_waits_within_the_bound(self) -> None:
        driver = _driver(
            [_marker("m1", recall_reconcile.UNLANDED, age=timedelta(hours=23))]
        )

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled(waiting=1)
        assert _queries(driver) == [recall_reconcile.MARKERS_QUERY]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("created_at", ["long ago", None])
    async def test_past_the_bound_it_expires_and_is_never_deleted(
        self, created_at: str | None
    ) -> None:
        old = _marker("m1", recall_reconcile.UNLANDED, age=timedelta(hours=25))
        stale = {**old, "created_at": created_at} if created_at != "long ago" else old
        driver = _driver([stale], [])

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled(expired=1)
        expire = driver.execute_query.await_args_list[1]
        assert expire.args[0] == recall_reconcile.EXPIRE_QUERY
        assert (expire.kwargs["uuid"], expire.kwargs["state"]) == ("m1", "expired")
        assert "DELETE" not in recall_reconcile.EXPIRE_QUERY

    @pytest.mark.asyncio
    async def test_an_expired_one_is_kept(self) -> None:
        driver = _driver(
            [
                _marker(
                    "m1", recall_reconcile.KEPT, state="expired", age=timedelta(days=9)
                )
            ]
        )

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled(expired=1)
        assert _queries(driver) == [recall_reconcile.MARKERS_QUERY]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("next_rank", "left"),
        [(0, True), (2, False), (3, False)],
        ids=["a landed one", "one waiting", "one expired"],
    )
    async def test_left_only_when_a_landed_one_remains(
        self, next_rank: int, left: bool
    ) -> None:
        """Landed markers come first, so what is beyond the limit is landed
        only when every one taken was."""
        rows = [_marker("m1", recall_reconcile.KEPT), _marker("m2", next_rank)]
        driver = _driver(rows)

        done = await recall_reconcile.reconcile(driver, "user_a", limit=1)

        assert done.left is left
        assert driver.execute_query.await_args_list[0].kwargs["limit"] == 2


class TestTheMarkersRead:
    def test_finds_the_episode_by_uuid_in_the_graph_never_by_name(self) -> None:
        query = recall_reconcile.MARKERS_QUERY
        assert "OPTIONAL MATCH (ep:Episodic {uuid: m.episode_uuid})" in query
        assert "WHERE ep.group_id = $group_id" in query
        assert "episode_name" not in query and "ep.name" not in query

    def test_landed_means_graphiti_saved_the_episode_and_every_fact(self) -> None:
        query = recall_reconcile.MARKERS_QUERY
        assert "ep.write_pending IS NULL AND found = wanted AS landed" in query
        assert "ORDER BY rank, created_at" in query

    def test_a_marker_goes_only_while_no_saved_episode_has_its_uuid(self) -> None:
        query = recall_reconcile.DROP_UNLANDED_QUERY
        assert "WHERE all(ep IN placed WHERE ep.write_pending IS NOT NULL)" in query
        assert query.index("WHERE all(") < query.index("DELETE m")
        assert "FOREACH (ep IN placed | DETACH DELETE ep)" in query


class TestInFlight:
    @pytest.mark.asyncio
    async def test_counts_the_pending_writes_citing_what_it_is_given(self) -> None:
        driver = _driver([{"count": 2}])

        count = await recall_reconcile.in_flight(driver, ["f1"], ["ep1"])

        assert count == 2
        [read] = driver.execute_query.await_args_list
        assert read.args[0] == recall_reconcile.IN_FLIGHT_QUERY
        cutoff = datetime.fromisoformat(read.kwargs["cutoff"])
        assert timedelta(hours=23, minutes=59) < _NOW - cutoff < timedelta(hours=25)
        assert (read.kwargs["facts"], read.kwargs["episodes"]) == (["f1"], ["ep1"])
        assert read.kwargs["state"] == "pending"

    @pytest.mark.asyncio
    async def test_nothing_to_cite_reads_nothing(self) -> None:
        driver = _driver()

        assert await recall_reconcile.in_flight(driver, [], []) == 0
        driver.execute_query.assert_not_awaited()
