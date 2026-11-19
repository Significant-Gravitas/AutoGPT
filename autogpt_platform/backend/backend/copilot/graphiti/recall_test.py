"""Unit tests for the recall policy's read side (``recall.py``).

The live-graph counterparts are ``recall_integration_test.py`` and
``recall_forget_integration_test.py``; these pin the policy itself — which
facts are live or forgotten, which episodes stay recallable and what the fact
search is asked for — against mocks. The episode reads are pinned in
``recall_episodes_test.py``, rendering in ``recall_render_test.py``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import EdgeReranker, SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER
from graphiti_core.search.search_filters import ComparisonOperator
from graphiti_core.search.search_utils import DEFAULT_MIN_SCORE

from . import recall
from .scope import MemoryScope

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_SCOPE = MemoryScope.for_user("user-abc")


def _fact(
    uuid: str = "e1",
    *,
    status: str | None = "active",
    reason: str | None = None,
    expired_at: datetime | None = None,
    invalid_at: datetime | None = None,
    forgotten_at: str | None = None,
) -> EntityEdge:
    attributes = {
        "status": status,
        "expiration_reason": reason,
        "forgotten_at": forgotten_at,
    }
    return EntityEdge(
        uuid=uuid,
        group_id=_SCOPE.group_id,
        source_node_uuid="alice",
        target_node_uuid="atlas",
        created_at=_NOW,
        name="works_on",
        fact="Alice works on Atlas",
        expired_at=expired_at,
        invalid_at=invalid_at,
        attributes={k: v for k, v in attributes.items() if v is not None},
    )


def _record(uuid: str) -> dict:
    """A forgotten edge as the graph returns it to graphiti's record reader."""
    return {
        "uuid": uuid,
        "source_node_uuid": "alice",
        "target_node_uuid": "atlas",
        "group_id": _SCOPE.group_id,
        "created_at": _NOW.isoformat(),
        "name": "[forgotten]",
        "fact": "[forgotten]",
        "episodes": [],
        "expired_at": _NOW.isoformat(),
        "valid_at": None,
        "invalid_at": None,
        "attributes": {"forgotten_at": _NOW.isoformat()},
    }


class TestLiveFactPredicate:
    def test_cypher_matches_live_statuses_and_unset_expiry(self) -> None:
        assert recall.live_fact_predicate("e") == (
            "e.expired_at IS NULL AND e.forgotten_at IS NULL"
            " AND (e.status IS NULL OR e.status IN ['active', 'tentative'])"
        )

    def test_cypher_without_tentative(self) -> None:
        assert recall.live_fact_predicate("live", include_tentative=False) == (
            "live.expired_at IS NULL AND live.forgotten_at IS NULL"
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

    def test_a_forgotten_fact_is_never_live_whatever_its_status(self) -> None:
        """Only a forget writes ``forgotten_at``; a later write that set the
        status back (graphiti's attribute rewrite) cannot make it live."""
        fact = _fact(status="active", forgotten_at=_NOW.isoformat())
        assert not recall.is_live(fact)


class TestForgottenFactPredicate:
    def test_cypher(self) -> None:
        assert recall.forgotten_fact_predicate("f") == (
            "(f.forgotten_at IS NOT NULL"
            " OR (f.expired_at IS NOT NULL AND f.invalid_at IS NULL"
            " AND f.expiration_reason IS NULL))"
        )

    @pytest.mark.parametrize(
        ("status", "reason", "expired_at", "invalid_at", "forgotten"),
        [
            ("active", None, None, None, False),
            (None, None, None, None, False),
            # A status or reason alone is no forget: graphiti's extraction
            # fills both from an episode's text, and a dream picks its reason.
            ("retracted", None, None, None, False),  # "I retracted my offer"
            ("retracted", "user_signal", _NOW, None, False),
            ("retracted", "settings_page", _NOW, None, False),
            ("superseded", "user_signal", _NOW, None, False),  # a dream demotion
            ("superseded", "stale_fact", _NOW, None, False),  # dream's own call
            ("active", None, _NOW, _NOW, False),  # graphiti's contradiction
            ("active", None, _NOW, None, True),  # the pre-policy forget
            ("superseded", None, _NOW, None, True),  # ...once backfilled a status
        ],
    )
    def test_is_forgotten(
        self,
        status: str | None,
        reason: str | None,
        expired_at: datetime | None,
        invalid_at: datetime | None,
        forgotten: bool,
    ) -> None:
        fact = _fact(
            status=status, reason=reason, expired_at=expired_at, invalid_at=invalid_at
        )
        assert recall.is_forgotten(fact) is forgotten
        assert not (forgotten and recall.is_live(fact)), "a forgotten fact is live"

    @pytest.mark.parametrize(
        ("status", "reason", "invalid_at"),
        [
            ("retracted", "user_signal", None),  # as the forget left it
            ("superseded", "stale_fact", _NOW),  # a dream write landed over it
            ("active", None, None),  # graphiti rewrote its attributes
        ],
    )
    def test_the_forget_marker_outlasts_any_other_write(
        self, status: str, reason: str | None, invalid_at: datetime | None
    ) -> None:
        fact = _fact(
            status=status,
            reason=reason,
            expired_at=_NOW,
            invalid_at=invalid_at,
            forgotten_at=_NOW.isoformat(),
        )
        assert recall.is_forgotten(fact)
        assert not recall.is_live(fact)


class TestRecallableEpisodePredicate:
    def test_cypher(self) -> None:
        assert recall.recallable_episode_predicate("ep", "gone") == (
            "ep.redacted_at IS NULL"
            " AND ep.write_pending IS NULL"
            " AND none(x IN coalesce(ep.entity_edges, []) WHERE x IN gone)"
        )

    def test_clause_collects_every_forgotten_fact(self) -> None:
        clause = recall.forgotten_facts_clause("gone")
        assert clause.startswith("OPTIONAL MATCH ()-[forgotten_fact:RELATES_TO]->()")
        assert recall.forgotten_fact_predicate("forgotten_fact") in clause
        assert clause.rstrip().endswith("WITH collect(forgotten_fact.uuid) AS gone")

    @pytest.mark.parametrize(
        ("entity_edges", "redacted", "pending", "recallable"),
        [
            (["e1", "e2"], False, False, True),
            (["e1", "gone-1"], False, False, False),  # one forgotten fact hides it
            ([], True, False, False),  # stamped by a forget
            ([], False, True, False),  # a dream write still being written
            ([], False, False, True),
        ],
    )
    def test_is_recallable_episode(
        self, entity_edges: list[str], redacted: bool, pending: bool, recallable: bool
    ) -> None:
        assert (
            recall.is_recallable_episode(
                entity_edges, {"gone-1"}, redacted=redacted, write_pending=pending
            )
            is recallable
        )


class TestSearchFacts:
    @staticmethod
    def _client(edges: list[EntityEdge], still: list[str] | None = None) -> MagicMock:
        """graphiti found ``edges``; reading them again by uuid finds
        ``still`` (every one of them unless given)."""
        client = MagicMock()
        client.search_ = AsyncMock(return_value=SearchResults(edges=edges))
        found = [edge.uuid for edge in edges] if still is None else still
        rows = [{"uuid": uuid} for uuid in found]
        client.driver.execute_query = AsyncMock(return_value=(rows, [], None))
        return client

    @pytest.mark.asyncio
    async def test_a_fact_forgotten_while_the_search_ran_is_not_returned(
        self,
    ) -> None:
        """The search's last step reads the facts again by uuid under the
        live test (``live_now``); what a forget took meanwhile is dropped."""
        client = self._client([_fact("forgotten"), _fact("kept")], still=["kept"])
        with patch.object(
            recall, "get_graphiti_client", AsyncMock(return_value=client)
        ):
            facts = await recall.search_facts(_SCOPE, "atlas", limit=10)

        assert [f.uuid for f in facts] == ["kept"]
        query = client.driver.execute_query.await_args.args[0]
        assert "e.uuid IN $uuids" in query
        assert recall.live_fact_predicate("e") in query
        assert client.driver.execute_query.await_args.kwargs == {
            "uuids": ["forgotten", "kept"]
        }

    @pytest.mark.asyncio
    async def test_nothing_found_reads_nothing_again(self) -> None:
        client = self._client([])
        with patch.object(
            recall, "get_graphiti_client", AsyncMock(return_value=client)
        ):
            assert await recall.search_facts(_SCOPE, "atlas", limit=10) == []

        client.driver.execute_query.assert_not_awaited()

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
                _fact("forgotten", forgotten_at=_NOW.isoformat()),
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
        query = client.driver.execute_query.await_args.args[0]
        assert recall.live_fact_predicate("e", include_tentative=False) in query

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


class TestSearchForgottenFacts:
    @pytest.mark.asyncio
    async def test_ranks_the_forgotten_facts_alone_by_meaning(self) -> None:
        """For a soft forget made permanent: the request is embedded and only
        forgotten facts are ranked against it, best first, above graphiti's
        similarity threshold, up to ``limit``, so however many live facts
        match better, none takes a forgotten one's slot."""
        client = MagicMock()
        client.embedder.create = AsyncMock(return_value=[0.25, 0.75])
        client.driver.provider = GraphProvider.FALKORDB
        rows = [_record("first"), _record("pre-policy")]
        client.driver.execute_query = AsyncMock(return_value=(rows, [], None))
        get_client = AsyncMock(return_value=client)
        with patch.object(recall, "get_graphiti_client", get_client):
            facts = await recall.search_forgotten_facts(_SCOPE, "the\natlas", limit=2)

        assert [f.uuid for f in facts] == ["first", "pre-policy"]
        get_client.assert_awaited_once_with(_SCOPE.group_id)
        client.embedder.create.assert_awaited_once_with(input_data=["the atlas"])
        query = client.driver.execute_query.await_args.args[0]
        assert f"AND {recall.forgotten_fact_predicate('e')}" in query
        assert "vec.cosineDistance(e.fact_embedding" in query
        assert query.endswith("ORDER BY score DESC LIMIT $limit")
        assert client.driver.execute_query.await_args.kwargs == {
            "group_id": _SCOPE.group_id,
            "search_vector": [0.25, 0.75],
            "min_score": DEFAULT_MIN_SCORE,
            "limit": 2,
        }
