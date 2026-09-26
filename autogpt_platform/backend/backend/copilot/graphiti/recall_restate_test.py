"""Unit tests for ``recall_restate``: which re-extracted statements become
new live edges, against a mock driver and a stubbed extraction.

The live run is ``recall_repair_integration_test.py`` (two facts restated
between the same entities).
"""

from collections import Counter
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

from . import recall_restate
from .recall import live_fact_predicate
from .recall_restate import normalized, unmatched
from .types import EDGE_TYPE_MAP, EDGE_TYPES

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_PAIR = ("alice", "atlas")


def _said(fact: str, source: str = "alice") -> EntityEdge:
    return EntityEdge(
        group_id="user_test",
        source_node_uuid=source,
        target_node_uuid="atlas",
        created_at=_NOW,
        name="MemoryFact",
        fact=fact,
        episodes=["ep-new"],
    )


class TestUnmatched:
    def test_every_restated_fact_gets_its_own_edge(self) -> None:
        """Two forgotten facts between the same entities, both said again:
        each keeps its own text (the first used to be taken twice)."""
        said = [_said("Alice works on Atlas"), _said("Alice runs the Atlas budget")]

        chosen = unmatched(said, {_PAIR}, Counter())

        assert [x.fact for x in chosen] == [
            "Alice works on Atlas",
            "Alice runs the Atlas budget",
        ]

    def test_a_statement_already_live_is_not_created_again(self) -> None:
        said = [_said("Alice works on Atlas"), _said("Alice leads Atlas")]
        live = Counter({(*_PAIR, "alice works on atlas"): 1})

        chosen = unmatched(said, {_PAIR}, live)

        assert [x.fact for x in chosen] == ["Alice leads Atlas"]

    def test_a_live_match_is_used_once(self) -> None:
        said = [_said("Alice works on Atlas"), _said("alice  works on atlas")]
        live = Counter({(*_PAIR, "alice works on atlas"): 1})

        chosen = unmatched(said, {_PAIR}, live)

        assert [x.fact for x in chosen] == ["alice  works on atlas"]

    def test_which_forgotten_edge_took_a_statement_does_not_matter(self) -> None:
        """Three statements, whatever edges graphiti merged them into."""
        said = [_said("one"), _said("two"), _said("three")]

        assert len(unmatched(said, {_PAIR}, Counter())) == 3

    def test_statements_between_other_entities_are_graphitis_to_keep(self) -> None:
        said = [_said("Bob leads Atlas", source="bob")]

        assert unmatched(said, {_PAIR}, Counter()) == []

    def test_statements_compare_without_case_or_spacing(self) -> None:
        assert normalized("  Alice   Works\non Atlas ") == "alice works on atlas"
        assert normalized(None) == ""


class TestRestate:
    @pytest.mark.asyncio
    async def test_extracts_once_and_saves_a_live_edge_per_statement(self) -> None:
        client = MagicMock()
        client.driver.execute_query = AsyncMock(return_value=([], [], None))
        result = _result()
        said = [_said("Alice works on Atlas"), _said("Alice runs the Atlas budget")]
        extract = AsyncMock(return_value=said)
        save = AsyncMock()

        with (
            patch.object(recall_restate, "extract_edges", extract),
            patch.object(recall_restate.EntityEdge, "save", save),
            patch.object(recall_restate.EntityEdge, "generate_embedding", AsyncMock()),
        ):
            edges = await recall_restate.restate(
                client, result, {_PAIR}, [], "instructions"
            )

        extract.assert_awaited_once_with(
            client.clients,
            result.episode,
            result.nodes,
            [],
            EDGE_TYPE_MAP,
            "user_test",
            EDGE_TYPES,
            "instructions",
        )
        assert [edge.fact for edge in edges] == [x.fact for x in said]
        assert all(edge.episodes == ["ep-new"] for edge in edges)
        assert all(edge.valid_at == _NOW and edge.expired_at is None for edge in edges)
        assert edges[0].attributes["status"] == "active"
        assert save.await_count == 2

    def test_live_facts_are_read_with_recalls_own_test(self) -> None:
        query = recall_restate._LIVE_STATEMENTS_QUERY
        assert f"WHERE {live_fact_predicate('e')}" in query
        assert "UNWIND $pairs AS pair" in query


def _result() -> AddEpisodeResults:
    episode = EpisodicNode(
        uuid="ep-new",
        name="conversation_s2",
        group_id="user_test",
        source=EpisodeType.message,
        source_description="User message in session s2",
        content="Alice works on Atlas. Alice runs the Atlas budget.",
        valid_at=_NOW,
    )
    nodes = [
        EntityNode(uuid=uuid, name=name, group_id="user_test")
        for uuid, name in (("alice", "Alice"), ("atlas", "Atlas"))
    ]
    return AddEpisodeResults(
        episode=episode,
        episodic_edges=[],
        nodes=nodes,
        edges=[],
        communities=[],
        community_edges=[],
    )
