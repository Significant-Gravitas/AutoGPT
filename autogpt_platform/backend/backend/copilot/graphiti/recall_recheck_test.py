"""Unit tests for ``recall_recheck``: what a read is about to show is checked
again by uuid, facts and episodes in one statement, and only what is still
live or recallable is kept.

The live runs, a forget answering while warm context's cross-encoder or the
``memory_search`` search is paused, or once warm context's fact check has
passed, are in ``recall_inflight_read_integration_test.py``.
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
    """The scope's graphiti client, whose driver finds only the uuids in
    ``still``, in one row as the statement returns them."""

    async def execute_query(
        query: str, *, fact_uuids: list[str], episode_uuids: list[str]
    ):
        facts = [uuid for uuid in fact_uuids if uuid in still]
        episodes = [uuid for uuid in episode_uuids if uuid in still]
        return [{"facts": facts, "episodes": episodes}], [], None

    client = MagicMock()
    client.driver.execute_query = AsyncMock(side_effect=execute_query)
    return client


async def _recheck(client: MagicMock, *args, **kwargs):
    with patch.object(recall, "get_graphiti_client", AsyncMock(return_value=client)):
        return await recall_recheck.recheck(_SCOPE, *args, **kwargs)


class TestRecheck:
    @pytest.mark.asyncio
    async def test_keeps_only_what_is_still_live_or_recallable_in_order(
        self,
    ) -> None:
        facts = [_fact("f1"), _fact("f2"), _fact("f3")]
        episodes = [_episode("ep1"), _episode("ep2")]

        kept, recalled = await _recheck(_client({"f3", "f1", "ep2"}), facts, episodes)

        assert [f.uuid for f in kept] == ["f1", "f3"]
        assert [e.uuid for e in recalled] == ["ep2"]

    @pytest.mark.asyncio
    async def test_facts_and_episodes_are_checked_in_one_statement(self) -> None:
        """No forget can land between the check of a fact and of an episode
        (a split check let one through: ``r6-read-split-check.py``)."""
        client = _client(set())

        await _recheck(
            client, [_fact("f1")], [_episode("ep1")], include_tentative=False
        )

        [call] = client.driver.execute_query.await_args_list
        query = call.args[0]
        assert query.startswith(recall.forgotten_facts_clause())
        assert recall.live_fact_predicate("fact", include_tentative=False) in query
        assert recall.recallable_episode_predicate("episode") in query
        assert call.kwargs == {"fact_uuids": ["f1"], "episode_uuids": ["ep1"]}

    @pytest.mark.asyncio
    async def test_nothing_to_show_reads_nothing(self) -> None:
        get_client = AsyncMock()
        with patch.object(recall, "get_graphiti_client", get_client):
            assert await recall_recheck.recheck(_SCOPE, [], []) == ([], [])

        get_client.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_answer_shows_nothing(self) -> None:
        client = MagicMock()
        client.driver.execute_query = AsyncMock(return_value=([], [], None))

        assert await _recheck(client, [_fact("f1")], [_episode("ep1")]) == ([], [])

    @pytest.mark.asyncio
    async def test_a_failed_read_raises_for_the_caller_to_degrade(self) -> None:
        """Warm context then shows no context and memory_search reports
        itself unavailable: nothing is shown unchecked."""
        client = _client(set())
        client.driver.execute_query.side_effect = RuntimeError("falkordb down")

        with pytest.raises(RuntimeError):
            await _recheck(client, [_fact("f1")], [])
