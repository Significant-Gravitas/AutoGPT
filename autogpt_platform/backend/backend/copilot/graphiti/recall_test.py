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
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import EdgeReranker, SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER
from graphiti_core.search.search_filters import ComparisonOperator

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
            " OR f.status = 'retracted'"
            " OR f.expiration_reason = 'user_signal'"
            " OR (f.expired_at IS NOT NULL AND f.invalid_at IS NULL"
            " AND f.expiration_reason IS NULL))"
        )

    @pytest.mark.parametrize(
        ("status", "reason", "expired_at", "invalid_at", "forgotten"),
        [
            ("active", None, None, None, False),
            (None, None, None, None, False),
            ("retracted", "user_signal", _NOW, None, True),
            ("retracted", "settings_page", _NOW, None, True),
            ("superseded", "user_signal", _NOW, None, True),  # dream, user's word
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
            " AND none(x IN coalesce(ep.entity_edges, []) WHERE x IN gone)"
        )

    def test_clause_collects_every_forgotten_fact(self) -> None:
        clause = recall.forgotten_facts_clause("gone")
        assert clause.startswith("OPTIONAL MATCH ()-[forgotten_fact:RELATES_TO]->()")
        assert recall.forgotten_fact_predicate("forgotten_fact") in clause
        assert clause.rstrip().endswith("WITH collect(forgotten_fact.uuid) AS gone")

    @pytest.mark.parametrize(
        ("entity_edges", "redacted", "recallable"),
        [
            (["e1", "e2"], False, True),
            (["e1", "gone-1"], False, False),  # one forgotten fact hides it all
            ([], True, False),  # stamped by a forget
            ([], False, True),
        ],
    )
    def test_is_recallable_episode(
        self, entity_edges: list[str], redacted: bool, recallable: bool
    ) -> None:
        assert (
            recall.is_recallable_episode(entity_edges, {"gone-1"}, redacted=redacted)
            is recallable
        )


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
