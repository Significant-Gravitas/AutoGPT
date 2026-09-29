"""Unit tests for ``recall_derivation``: a dream write's citations are marked
before the ingestion worker writes it, failing closed, then recorded on its
episode and on the facts only dream episodes state, and the marker dropped,
all under the graph's write lock; a record that fails leaves the marker and
is reported as ``provenance_pending``.

The worker's side is in ``recall_derivation_worker_test.py``. On FalkorDB,
through ``dream/apply.py`` and the production worker:
``recall_cascade_integration_test.py`` (the facts a dream write produced or
merged into) and ``recall_provenance_integration_test.py`` (a record that
failed, reconciled before the next forget).
"""

from unittest.mock import AsyncMock, MagicMock

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
    async def test_marks_the_complete_citations_under_the_episodes_name(
        self,
    ) -> None:
        driver = _driver()

        marker = await recall_derivation.mark(driver, "user_a", "dream_p_1", _CITED)

        [call] = driver.execute_query.await_args_list
        assert call.args[0] == recall_derivation.MARK_QUERY
        params = dict(call.kwargs)
        assert params.pop("now")
        assert params == {
            "uuid": marker,
            "group_id": "user_a",
            "name": "dream_p_1",
            "facts": ["f1", "f2"],
            "episodes": ["ep1"],
        }

    @pytest.mark.asyncio
    async def test_a_marker_that_cannot_be_written_raises(self) -> None:
        driver = _driver()
        driver.execute_query.side_effect = RuntimeError("down")

        with pytest.raises(RuntimeError):
            await recall_derivation.mark(driver, "user_a", "dream_p_1", _CITED)

    def test_a_marker_holds_uuids_and_names_only(self) -> None:
        query = recall_derivation.MARK_QUERY
        assert "CREATE (:DreamCitations {" in query
        for text in ("content", "fact:", "rationale", "source_description"):
            assert text not in query


class TestRecord:
    @pytest.mark.asyncio
    async def test_records_the_episode_then_the_facts_then_drops_the_marker(
        self,
    ) -> None:
        driver = _driver()

        recorded = await recall_derivation.record(
            driver, "user_a", "m1", "dream-ep", ["e1"], _CITED
        )

        episode, facts, drop = driver.execute_query.await_args_list
        assert episode.args[0] == recall_derivation.RECORD_EPISODE_QUERY
        assert episode.kwargs == {
            "episode": "dream-ep",
            "facts": ["f1", "f2"],
            "episodes": ["ep1"],
        }
        assert facts.args[0] == recall_derivation.STAMP_FACTS_QUERY
        assert facts.kwargs == {"uuids": ["e1"], "group_id": "user_a"}
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

        await recall_derivation.record(driver, "user_a", "m1", "dream-ep", [], _CITED)

        queries = [call.args[0] for call in driver.execute_query.await_args_list]
        assert queries == [
            recall_derivation.RECORD_EPISODE_QUERY,
            recall_derivation.DROP_MARKER_QUERY,
        ]

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
