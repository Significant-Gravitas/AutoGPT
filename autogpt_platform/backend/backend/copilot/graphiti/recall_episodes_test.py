"""Unit tests for the recall policy's episode reads: ``recent_episodes``
for the assistant, ``previous_episode_uuids`` for ingestion's extraction
context (``recall_ingest.py``), and ``record_hit``.

The predicates and the fact search are pinned in ``recall_test.py``; the
live runs are ``recall_integration_test.py`` and
``ingest_recall_integration_test.py``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT

from . import recall, recall_ingest
from .scope import MemoryScope

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_SCOPE = MemoryScope.for_user("user-abc")


def _episode_record(uuid: str, content: str) -> dict:
    return {
        "uuid": uuid,
        "name": uuid,
        "group_id": _SCOPE.group_id,
        "created_at": _NOW.isoformat(),
        "source": "text",
        "source_description": "chat",
        "content": content,
        "valid_at": _NOW.isoformat(),
        "entity_edges": [],
    }


def _driver_returning(records: list[dict]) -> AsyncMock:
    driver = AsyncMock()
    driver.execute_query.return_value = (records, [], None)
    return driver


class TestRecentEpisodes:
    @pytest.mark.asyncio
    async def test_asks_for_recallable_episodes_and_returns_oldest_first(
        self,
    ) -> None:
        driver = _driver_returning(
            [_episode_record("newest", "b"), _episode_record("older", "a")]
        )
        get_client = AsyncMock(return_value=MagicMock(driver=driver))
        with patch.object(recall, "get_graphiti_client", get_client):
            episodes = await recall.recent_episodes(_SCOPE, 5)

        get_client.assert_awaited_once_with(_SCOPE.group_id)
        query = driver.execute_query.await_args.args[0]
        kwargs = driver.execute_query.await_args.kwargs
        assert query.startswith(recall.forgotten_facts_clause())
        assert recall.recallable_episode_predicate("e") in query
        assert "ORDER BY e.valid_at DESC" in query
        assert kwargs["group_id"] == _SCOPE.group_id
        assert kwargs["limit"] == 5
        assert kwargs["source"] is None, "recall reads every source"
        assert [ep.uuid for ep in episodes] == ["older", "newest"]

    @pytest.mark.asyncio
    async def test_reads_on_the_cached_clients_driver_and_leaves_it_open(
        self,
    ) -> None:
        """No connection per read: the scope's cached client is shared with
        the fact search and the recheck, so the read must not close it, even
        when it fails."""
        driver = AsyncMock()
        driver.execute_query.side_effect = RuntimeError("falkordb down")
        get_client = AsyncMock(return_value=MagicMock(driver=driver))
        with patch.object(recall, "get_graphiti_client", get_client):
            with pytest.raises(RuntimeError):
                await recall.recent_episodes(_SCOPE, 5)

        driver.close.assert_not_awaited()


class TestPreviousEpisodeUuids:
    @pytest.mark.asyncio
    async def test_graphitis_own_pick_through_the_policy_oldest_first(
        self,
    ) -> None:
        """graphiti's ``retrieve_episodes`` window (same source, up to the
        new episode's time, ``RELEVANT_SCHEMA_LIMIT`` newest) with the
        recallable-episode test added."""
        driver = _driver_returning(
            [_episode_record("newest", "b"), _episode_record("older", "a")]
        )

        uuids = await recall_ingest.previous_episode_uuids(
            driver, _SCOPE.group_id, _NOW, EpisodeType.message
        )

        assert uuids == ["older", "newest"]
        query = driver.execute_query.await_args.args[0]
        assert recall.recallable_episode_predicate("e") in query
        assert "($source IS NULL OR e.source = $source)" in query
        assert driver.execute_query.await_args.kwargs == {
            "group_id": _SCOPE.group_id,
            "reference_time": _NOW,
            "source": "message",
            "limit": RELEVANT_SCHEMA_LIMIT,
        }

    @pytest.mark.asyncio
    async def test_a_failed_read_means_no_earlier_episodes_not_graphitis(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = RuntimeError("falkordb down")

        uuids = await recall_ingest.previous_episode_uuids(
            driver, _SCOPE.group_id, _NOW, EpisodeType.text
        )

        assert uuids == [], "None would let graphiti pick, forgotten text included"
        assert "extracting without earlier episodes" in caplog.text


class TestRecordHit:
    @pytest.mark.asyncio
    async def test_counts_each_edge_once(self) -> None:
        record_memory_hit = AsyncMock()
        with patch.object(recall, "record_memory_hit", record_memory_hit):
            await recall.record_hit(_SCOPE, ["e1", "e2", "e1"])

        assert [c.args for c in record_memory_hit.await_args_list] == [
            (_SCOPE, "e1"),
            (_SCOPE, "e2"),
        ]
