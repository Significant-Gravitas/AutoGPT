"""Unit tests for ``recall_recheck``: what a read is about to show is read
again by uuid, and only what is still live or recallable is kept.

The live runs, a forget answering while warm context's cross-encoder or the
``memory_search`` search is paused, are in ``recall_inflight_read_integration_test.py``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode

from . import recall, recall_recheck
from .scope import MemoryScope

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_SCOPE = MemoryScope.for_user("user-abc")


def _fact(uuid: str) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id=_SCOPE.group_id,
        source_node_uuid="alice",
        target_node_uuid="atlas",
        created_at=_NOW,
        name="works_on",
        fact=f"fact {uuid}",
    )


def _episode(uuid: str) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=uuid,
        group_id=_SCOPE.group_id,
        source=EpisodeType.text,
        source_description="chat",
        content=f"episode {uuid}",
        created_at=_NOW,
        valid_at=_NOW,
    )


def _client(still: set[str]) -> MagicMock:
    """The scope's graphiti client, whose driver's re-read finds only the
    uuids in ``still``."""

    async def execute_query(query: str, *, uuids: list[str]):
        return [{"uuid": uuid} for uuid in uuids if uuid in still], [], None

    client = MagicMock()
    client.driver.execute_query = AsyncMock(side_effect=execute_query)
    return client


class TestRecheck:
    @pytest.mark.asyncio
    async def test_keeps_only_what_is_still_live_or_recallable_in_order(
        self,
    ) -> None:
        client = _client({"f3", "f1", "ep2"})
        facts = [_fact("f1"), _fact("f2"), _fact("f3")]
        episodes = [_episode("ep1"), _episode("ep2")]
        get_client = AsyncMock(return_value=client)
        with patch.object(recall, "get_graphiti_client", get_client):
            kept, recalled = await recall_recheck.recheck(_SCOPE, facts, episodes)

        get_client.assert_awaited_once_with(_SCOPE.group_id)
        assert [f.uuid for f in kept] == ["f1", "f3"]
        assert [e.uuid for e in recalled] == ["ep2"]

    @pytest.mark.asyncio
    async def test_reads_facts_with_the_live_test_and_episodes_with_the_recallable_one(
        self,
    ) -> None:
        client = _client(set())
        with patch.object(
            recall, "get_graphiti_client", AsyncMock(return_value=client)
        ):
            await recall_recheck.recheck(
                _SCOPE, [_fact("f1")], [_episode("ep1")], include_tentative=False
            )

        queries = {
            call.kwargs["uuids"][0]: call.args[0]
            for call in client.driver.execute_query.await_args_list
        }
        assert recall.live_fact_predicate("e", include_tentative=False) in queries["f1"]
        assert queries["ep1"].startswith(recall.forgotten_facts_clause())
        assert recall.recallable_episode_predicate("e") in queries["ep1"]
        assert "e.uuid IN $uuids" in queries["ep1"]

    @pytest.mark.asyncio
    async def test_nothing_to_show_reads_nothing(self) -> None:
        get_client = AsyncMock()
        with patch.object(recall, "get_graphiti_client", get_client):
            assert await recall_recheck.recheck(_SCOPE, [], []) == ([], [])

        get_client.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_read_raises_for_the_caller_to_degrade(self) -> None:
        """Warm context then shows no context and memory_search reports
        itself unavailable: nothing is shown unchecked."""
        client = _client(set())
        client.driver.execute_query.side_effect = RuntimeError("falkordb down")
        with patch.object(
            recall, "get_graphiti_client", AsyncMock(return_value=client)
        ):
            with pytest.raises(RuntimeError):
                await recall_recheck.recheck(_SCOPE, [_fact("f1")], [])
