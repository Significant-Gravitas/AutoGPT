"""Unit tests for ``recall_derivation``: a dream write's citations are marked
before the ingestion worker writes it, under the uuid its episode is given,
failing closed; then recorded on that episode (found by uuid in its graph)
and on the facts only dream episodes state, the write settled, and only then
the marker dropped; a record or settle that fails leaves the marker. An
``add_episode`` that raised marks the marker aborted; a write dropped before
it has its marker withdrawn.

The worker's side is in ``recall_derivation_worker_test.py``, the settle in
``recall_landing_test.py``. On FalkorDB, through ``dream/apply.py`` and the
production worker: ``recall_cascade_integration_test.py`` (the facts a
dream write produced or merged into), ``recall_provenance_integration_test.py``
(a record that failed, reconciled before the next forget) and
``recall_marker_integration_test.py`` (a colliding episode name, writes that
land late).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import recall_derivation
from .recall_citations import Citations

_CITED = Citations(fact_uuids=["f1", "f1", "f2"], episode_uuids=["ep1"])


def _driver() -> MagicMock:
    driver = MagicMock()
    driver.execute_query = AsyncMock(return_value=([], [], None))
    return driver


class TestMark:
    @pytest.mark.asyncio
    async def test_marks_the_complete_citations_under_the_episodes_uuid(
        self,
    ) -> None:
        driver = _driver()

        marker = await recall_derivation.mark(
            driver, "user_a", "episode-uuid", "dream_p_1", _CITED
        )

        [call] = driver.execute_query.await_args_list
        assert call.args[0] == recall_derivation.MARK_QUERY
        params = dict(call.kwargs)
        assert params.pop("now")
        assert params == {
            "uuid": marker,
            "group_id": "user_a",
            "episode": "episode-uuid",
            "name": "dream_p_1",
            "facts": ["f1", "f2"],
            "episodes": ["ep1"],
            "state": "pending",
        }

    @pytest.mark.asyncio
    async def test_a_marker_that_cannot_be_written_raises(self) -> None:
        driver = _driver()
        driver.execute_query.side_effect = RuntimeError("down")

        with pytest.raises(RuntimeError):
            await recall_derivation.mark(
                driver, "user_a", "episode-uuid", "dream_p_1", _CITED
            )

    def test_a_marker_holds_uuids_and_names_only(self) -> None:
        query = recall_derivation.MARK_QUERY
        assert "CREATE (:DreamCitations {" in query
        assert "episode_uuid: $episode" in query
        for text in ("content", "fact:", "rationale", "source_description"):
            assert text not in query

    def test_the_episode_is_found_by_its_uuid_in_its_graph_never_its_name(
        self,
    ) -> None:
        query = recall_derivation.RECORD_EPISODE_QUERY
        assert "MATCH (ep:Episodic {uuid: $episode})" in query
        assert "ep.group_id = $group_id" in query
        assert "name" not in query


class TestAbort:
    @pytest.mark.asyncio
    async def test_marks_the_marker_aborted(self) -> None:
        driver = _driver()

        await recall_derivation.abort(driver, "m1")

        [call] = driver.execute_query.await_args_list
        assert call.args[0] == recall_derivation.ABORT_QUERY
        assert (call.kwargs["uuid"], call.kwargs["state"]) == ("m1", "aborted")

    @pytest.mark.asyncio
    async def test_never_raises(self, caplog: pytest.LogCaptureFixture) -> None:
        driver = _driver()
        driver.execute_query.side_effect = RuntimeError("down")

        await recall_derivation.abort(driver, "m1")

        assert "Could not mark dream marker m1 aborted" in caplog.text


class TestWithdraw:
    """The marker of a write dropped before ``add_episode`` goes."""

    @pytest.mark.asyncio
    async def test_deletes_the_marker(self) -> None:
        driver = _driver()

        await recall_derivation.withdraw(driver, "m1")

        [call] = driver.execute_query.await_args_list
        assert (call.args[0], call.kwargs) == (
            recall_derivation.DROP_MARKER_QUERY,
            {"uuid": "m1"},
        )

    @pytest.mark.asyncio
    async def test_one_that_cannot_be_deleted_is_marked_aborted(self) -> None:
        """For reconcile to drop: it has no episode."""
        driver = _driver()
        driver.execute_query.side_effect = [RuntimeError("down"), ([], [], None)]

        await recall_derivation.withdraw(driver, "m1")

        _, abort = driver.execute_query.await_args_list
        assert (abort.args[0], abort.kwargs["state"]) == (
            recall_derivation.ABORT_QUERY,
            "aborted",
        )


class TestRecord:
    @pytest.mark.asyncio
    async def test_records_the_episode_the_facts_settles_then_drops_the_marker(
        self,
    ) -> None:
        driver = _driver()
        settled: list[int] = []

        async def settle(driver_, group_id, citations) -> bool:
            settled.append(driver.execute_query.await_count)
            assert (group_id, citations) == ("user_a", _CITED)
            return True

        with patch.object(recall_derivation, "settle", settle):
            recorded = await recall_derivation.record(
                driver, "user_a", "m1", "dream-ep", ["e1"], _CITED
            )

        episode, facts, drop = driver.execute_query.await_args_list
        assert episode.args[0] == recall_derivation.RECORD_EPISODE_QUERY
        assert episode.kwargs == {
            "episode": "dream-ep",
            "group_id": "user_a",
            "facts": ["f1", "f2"],
            "episodes": ["ep1"],
        }
        assert facts.args[0] == recall_derivation.STAMP_FACTS_QUERY
        assert facts.kwargs == {"uuids": ["e1"], "group_id": "user_a"}
        assert settled == [2], "settled after the record, before the drop"
        assert (drop.args[0], drop.kwargs) == (
            recall_derivation.DROP_MARKER_QUERY,
            {"uuid": "m1"},
        )
        assert recorded is True

    @pytest.mark.asyncio
    async def test_a_write_that_touched_no_fact_records_its_episode_only(
        self,
    ) -> None:
        driver = _driver()

        with patch.object(recall_derivation, "settle", AsyncMock(return_value=True)):
            await recall_derivation.record(
                driver, "user_a", "m1", "dream-ep", [], _CITED
            )

        queries = [call.args[0] for call in driver.execute_query.await_args_list]
        assert queries == [
            recall_derivation.RECORD_EPISODE_QUERY,
            recall_derivation.DROP_MARKER_QUERY,
        ]

    @pytest.mark.asyncio
    async def test_a_settle_left_unfinished_keeps_the_marker(self) -> None:
        driver = _driver()

        with patch.object(recall_derivation, "settle", AsyncMock(return_value=False)):
            recorded = await recall_derivation.record(
                driver, "user_a", "m1", "dream-ep", ["e1"], _CITED
            )

        assert recorded is False
        queries = [call.args[0] for call in driver.execute_query.await_args_list]
        assert recall_derivation.DROP_MARKER_QUERY not in queries

    @pytest.mark.asyncio
    async def test_a_failure_leaves_the_marker_and_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        driver = _driver()
        driver.execute_query.side_effect = [([], [], None), RuntimeError("down")]

        recorded = await recall_derivation.record(
            driver, "user_a", "m1", "dream-ep", ["e1"], _CITED
        )

        assert recorded is False
        assert driver.execute_query.await_count == 2, "the marker is not dropped"
        assert "its marker stays for reconcile" in caplog.text


class TestTheStamp:
    def test_a_fact_is_stamped_only_when_every_source_is_a_recorded_episode(
        self,
    ) -> None:
        query = recall_derivation.STAMP_FACTS_QUERY
        assert "all(x IN coalesce(e.episodes, []) WHERE x IN found)" in query
        assert "all(s IN sources WHERE s.derived_from_facts IS NOT NULL)" in query
        assert "e.forgotten_at IS NULL" in query

    def test_the_stamp_is_the_union_of_its_sources_records(self) -> None:
        query = recall_derivation.STAMP_FACTS_QUERY
        assert "acc + [x IN s.derived_from_facts WHERE NOT x IN acc]" in query
        assert "SET e.derived_from_facts = reduce(" in query
