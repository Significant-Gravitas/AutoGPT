"""Unit tests for the recall policy's read side (``recall.py``).

The live-graph counterpart is ``recall_integration_test.py``; these pin the
policy itself — which statuses survive, what the search is asked for, how a
retired fact renders — against mocks.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config import EdgeReranker, SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER
from graphiti_core.search.search_filters import ComparisonOperator

from . import recall
from .memory_model import MemoryEnvelope, MemoryKind, SourceKind
from .scope import MemoryScope

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_SCOPE = MemoryScope.for_user("user-abc")


def _fact(
    uuid: str = "e1",
    *,
    status: str | None = "active",
    expired_at: datetime | None = None,
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
    fact: str = "Alice works on Atlas",
) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id=_SCOPE.group_id,
        source_node_uuid="alice",
        target_node_uuid="atlas",
        created_at=_NOW,
        name="works_on",
        fact=fact,
        expired_at=expired_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        attributes={} if status is None else {"status": status},
    )


def _episode(content: str = "talked about coffee", uuid: str = "ep1") -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=uuid,
        group_id=_SCOPE.group_id,
        source=EpisodeType.text,
        source_description="chat",
        content=content,
        created_at=_NOW,
        valid_at=_NOW,
    )


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


class TestLiveFactPredicate:
    def test_cypher_matches_live_statuses_and_unset_expiry(self) -> None:
        assert recall.live_fact_predicate("e") == (
            "e.expired_at IS NULL"
            " AND (e.status IS NULL OR e.status IN ['active', 'tentative'])"
        )

    def test_cypher_without_tentative(self) -> None:
        assert recall.live_fact_predicate("live", include_tentative=False) == (
            "live.expired_at IS NULL"
            " AND (live.status IS NULL OR live.status IN ['active'])"
        )

    @pytest.mark.parametrize(
        ("status", "expired_at", "include_tentative", "live"),
        [
            ("active", None, True, True),
            (None, None, True, True),  # predates MemoryFact: counts as active
            ("tentative", None, True, True),
            ("tentative", None, False, False),
            ("superseded", None, True, False),
            ("contradicted", None, True, False),
            ("retracted", None, True, False),
            ("active", _NOW, True, False),  # graphiti-expired, status untouched
            ("retracted", _NOW, True, False),
        ],
    )
    def test_is_live(
        self,
        status: str | None,
        expired_at: datetime | None,
        include_tentative: bool,
        live: bool,
    ) -> None:
        fact = _fact(status=status, expired_at=expired_at)
        assert recall.is_live(fact, include_tentative=include_tentative) is live


class TestSearchFacts:
    @staticmethod
    def _client(edges: list[EntityEdge]) -> MagicMock:
        client = MagicMock()
        client.search_ = AsyncMock(return_value=SearchResults(edges=edges))
        return client

    @pytest.mark.asyncio
    async def test_keeps_live_facts_and_drops_retired_ones(self) -> None:
        client = self._client(
            [
                _fact("active"),
                _fact("no-status", status=None),
                _fact("tentative", status="tentative"),
                _fact("superseded", status="superseded"),
                _fact("contradicted", status="contradicted"),
                _fact("retracted", status="retracted"),
            ]
        )
        with patch.object(
            recall, "get_graphiti_client", AsyncMock(return_value=client)
        ):
            facts = await recall.search_facts(_SCOPE, "atlas", limit=10)

        assert [f.uuid for f in facts] == ["active", "no-status", "tentative"]

    @pytest.mark.asyncio
    async def test_tentative_dropped_when_not_wanted(self) -> None:
        client = self._client([_fact("active"), _fact("tentative", status="tentative")])
        with patch.object(
            recall, "get_graphiti_client", AsyncMock(return_value=client)
        ):
            facts = await recall.search_facts(
                _SCOPE, "atlas", limit=10, include_tentative=False
            )

        assert [f.uuid for f in facts] == ["active"]

    @pytest.mark.asyncio
    async def test_asks_graphiti_for_unexpired_facts_in_the_scope(self) -> None:
        client = self._client([])
        scope = MemoryScope.for_expert("user-abc", "expert-1")
        get_client = AsyncMock(return_value=client)
        with patch.object(recall, "get_graphiti_client", get_client):
            await recall.search_facts(scope, "atlas", limit=7)

        get_client.assert_awaited_once_with(scope.group_id)
        kwargs = client.search_.await_args.kwargs
        assert kwargs["query"] == "atlas"
        assert kwargs["group_ids"] == [scope.group_id]
        [[date_filter]] = kwargs["search_filter"].expired_at
        assert date_filter.comparison_operator == ComparisonOperator.is_null
        # Default recipe: the hybrid RRF config ``Graphiti.search`` uses, so
        # ranking matches what memory_search did before.
        assert kwargs["config"].edge_config.reranker == EdgeReranker.rrf
        assert kwargs["config"].limit == 7

    @pytest.mark.asyncio
    async def test_recipe_limit_set_on_a_copy(self) -> None:
        client = self._client([])
        recipe_limit = EDGE_HYBRID_SEARCH_CROSS_ENCODER.limit
        with patch.object(
            recall, "get_graphiti_client", AsyncMock(return_value=client)
        ):
            await recall.search_facts(
                _SCOPE, "atlas", limit=3, recipe=EDGE_HYBRID_SEARCH_CROSS_ENCODER
            )

        config = client.search_.await_args.kwargs["config"]
        assert config.edge_config.reranker == EdgeReranker.cross_encoder
        assert config.limit == 3
        assert EDGE_HYBRID_SEARCH_CROSS_ENCODER.limit == recipe_limit


class TestRecentEpisodes:
    @pytest.mark.asyncio
    async def test_skips_redacted_episodes_and_returns_oldest_first(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = (
            [_episode_record("newest", "b"), _episode_record("older", "a")],
            [],
            None,
        )
        open_driver = MagicMock(return_value=driver)
        with patch.object(recall, "open_driver", open_driver):
            episodes = await recall.recent_episodes(_SCOPE, 5)

        open_driver.assert_called_once_with(_SCOPE)
        query = driver.execute_query.await_args.args[0]
        kwargs = driver.execute_query.await_args.kwargs
        assert "e.redacted_at IS NULL" in query
        assert "ORDER BY e.valid_at DESC" in query
        assert kwargs["group_id"] == _SCOPE.group_id
        assert kwargs["limit"] == 5
        assert [ep.uuid for ep in episodes] == ["older", "newest"]
        driver.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_driver_when_the_read_fails(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = RuntimeError("falkordb down")
        with patch.object(recall, "open_driver", MagicMock(return_value=driver)):
            with pytest.raises(RuntimeError):
                await recall.recent_episodes(_SCOPE, 5)

        driver.close.assert_awaited_once()


class TestRender:
    def test_live_fact_shows_its_validity(self) -> None:
        fact = _fact(valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert recall.render(fact) == (
            "Alice works on Atlas (valid: 2025-01-01 00:00:00+00:00 — present)"
        )

    def test_live_fact_with_an_end_date(self) -> None:
        fact = _fact(invalid_at=datetime(2025, 6, 1, tzinfo=timezone.utc))
        assert recall.render(fact) == (
            "Alice works on Atlas (valid: unknown — 2025-06-01 00:00:00+00:00)"
        )

    @pytest.mark.parametrize("status", ["retracted", "superseded", "contradicted"])
    def test_retired_fact_is_labelled_never_present(self, status: str) -> None:
        fact = _fact(status=status, expired_at=_NOW)
        rendered = recall.render(fact)
        assert rendered == f"Alice works on Atlas ({status} 2026-09-26 12:00:00+00:00)"
        assert "present" not in rendered

    def test_expired_fact_with_a_live_status_is_labelled_expired(self) -> None:
        # What graphiti's own contradiction handling leaves behind: expired,
        # status never changed.
        fact = _fact(status="active", expired_at=_NOW)
        assert recall.render(fact) == (
            "Alice works on Atlas (expired 2026-09-26 12:00:00+00:00)"
        )

    def test_retired_status_without_a_timestamp(self) -> None:
        fact = _fact(status="retracted")
        assert (
            recall.render(fact) == "Alice works on Atlas (retracted at an unknown time)"
        )

    def test_fact_text_falls_back_to_the_relation_name(self) -> None:
        assert recall.fact_text(_fact(fact="")) == "works_on"


class TestRenderEpisode:
    def test_timestamp_and_body(self) -> None:
        assert recall.render_episode(_episode("talked about coffee")) == (
            "[2026-09-26 12:00:00+00:00] talked about coffee"
        )

    def test_body_cut_to_display_length(self) -> None:
        body = "x" * recall.EPISODE_DISPLAY_CHARS
        assert recall.render_episode(_episode("x" * 1000)) == (
            f"[2026-09-26 12:00:00+00:00] {body}"
        )


class TestEpisodeScope:
    @pytest.mark.parametrize(
        "content",
        ["plain conversation text", "[1, 2, 3]", '"just a string"', "null"],
    )
    def test_non_envelope_bodies_are_global(self, content: str) -> None:
        assert recall.episode_scope(_episode(content)) == recall.GLOBAL_SCOPE

    def test_envelope_scope_is_read(self) -> None:
        envelope = MemoryEnvelope(content="project note", scope="project:crm")
        episode = _episode(envelope.model_dump_json())
        assert recall.episode_scope(episode) == "project:crm"

    def test_envelope_without_scope_is_global(self) -> None:
        assert recall.episode_scope(_episode('{"content": "x"}')) == "real:global"

    def test_long_envelope_is_parsed_whole(self) -> None:
        """An envelope longer than the display cut is still parsed: scoping
        the truncated body would leak a project memory into global recall."""
        envelope = MemoryEnvelope(
            content="x" * 600,
            source_kind=SourceKind.user_asserted,
            scope="project:crm",
            memory_kind=MemoryKind.fact,
        )
        body = envelope.model_dump_json()
        assert len(body) > recall.EPISODE_DISPLAY_CHARS

        assert recall.episode_scope(_episode(body)) == "project:crm"


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
