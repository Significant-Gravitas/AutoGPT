"""Unit tests for ``recall_ingest``: how ingestion keeps a forget through
graphiti's ``add_episode``, against a mock driver and a stubbed extraction.

The live runs, through the production worker, are
``recall_ingest_integration_test.py``.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

from . import recall_ingest
from .recall import FORGOTTEN_FACT, forgotten_fact_predicate
from .types import EDGE_TYPE_MAP, EDGE_TYPES

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_GROUP = "user_test"
_SENTENCE = "Alice works on Atlas"


def _row(uuid: str = "f1", **fields: Any) -> dict[str, Any]:
    """A forgotten edge as the snapshot reads it."""
    row: dict[str, Any] = {field: None for field in recall_ingest._FIELDS}
    row.update(
        uuid=uuid,
        source="alice",
        target="atlas",
        status="retracted",
        expiration_reason="user_signal",
        forgotten_at="2026-09-25T00:00:00+00:00",
        expired_at="2026-09-25T00:00:00+00:00",
        fact=FORGOTTEN_FACT,
        fact_redacted=_SENTENCE,
        name=FORGOTTEN_FACT,
        name_redacted="MemoryFact",
        episodes=["ep-old"],
    )
    row.update(fields)
    return row


def _edge(uuid: str, source: str = "alice", fact: str = _SENTENCE) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id=_GROUP,
        source_node_uuid=source,
        target_node_uuid="atlas",
        created_at=_NOW,
        name="MemoryFact",
        fact=fact,
        episodes=["ep-new"],
    )


def _result(*edges: EntityEdge, cites: list[str]) -> AddEpisodeResults:
    episode = EpisodicNode(
        uuid="ep-new",
        name="conversation_s2",
        group_id=_GROUP,
        source=EpisodeType.message,
        source_description="User message in session s2",
        content=_SENTENCE,
        valid_at=_NOW,
        entity_edges=cites,
    )
    nodes = [
        EntityNode(uuid=uuid, name=name, group_id=_GROUP)
        for uuid, name in (("alice", "Alice"), ("atlas", "Atlas"))
    ]
    return AddEpisodeResults(
        episode=episode,
        episodic_edges=[],
        nodes=nodes,
        edges=list(edges),
        communities=[],
        community_edges=[],
    )


def _client(after: list[dict[str, Any]]) -> MagicMock:
    """A client whose graph reads ``after`` back and accepts every write."""

    async def execute_query(query: str, **params: object):
        return (after if query == recall_ingest._READ_QUERY else []), [], None

    client = MagicMock()
    client.driver.execute_query = AsyncMock(side_effect=execute_query)
    return client


def _before(*rows: dict[str, Any]) -> dict[str, recall_ingest.ForgottenEdge]:
    return {row["uuid"]: recall_ingest._edge(row) for row in rows}


# The forgotten edge as graphiti left it after merging the new episode in.
_ABSORBED = _row(episodes=["ep-old", "ep-new"], status="active")
_FORGOTTEN = recall_ingest._edge(_row()).fields


async def _keep_restating(
    client: MagicMock,
    result: AddEpisodeResults,
    extracted: list[EntityEdge],
    instructions: str | None = None,
) -> tuple[AsyncMock, AsyncMock]:
    """``keep_forgotten`` over the snapshot ``_row()``, with graphiti's edge
    extraction answering ``extracted``: the extraction and save mocks."""
    extract = AsyncMock(return_value=extracted)
    save = AsyncMock()
    with (
        patch.object(recall_ingest, "extract_edges", extract),
        patch.object(recall_ingest.EntityEdge, "save", save),
        patch.object(recall_ingest.EntityEdge, "generate_embedding", AsyncMock()),
    ):
        await recall_ingest.keep_forgotten(
            client, _before(_row()), result, [], instructions
        )
    return extract, save


def _writes(client: MagicMock) -> list[tuple[str, dict]]:
    return [
        (call.args[0], call.kwargs)
        for call in client.driver.execute_query.await_args_list
        if call.args[0] != recall_ingest._READ_QUERY
    ]


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_reads_every_forgotten_fact_by_the_policy_predicate(self) -> None:
        driver = AsyncMock()
        driver.execute_query.return_value = ([_row()], [], None)

        snapshot = await recall_ingest.snapshot_forgotten(driver)

        assert list(snapshot) == ["f1"]
        assert snapshot["f1"].fields["fact_redacted"] == _SENTENCE
        query = driver.execute_query.await_args.args[0]
        assert f"WHERE {forgotten_fact_predicate('e')}" in query
        for field in ("forgotten_at", "fact_redacted", "episodes", "invalid_at"):
            assert f"e.{field} AS {field}" in query

    @pytest.mark.asyncio
    async def test_a_failed_snapshot_guards_nothing_and_ingestion_goes_on(
        self,
    ) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = RuntimeError("down")

        assert await recall_ingest.snapshot_forgotten(driver) == {}


class TestKeepForgotten:
    @pytest.mark.asyncio
    async def test_nothing_forgotten_means_nothing_to_repair(self) -> None:
        client = _client([])

        await recall_ingest.keep_forgotten(client, {}, _result(cites=[]), [], None)

        client.driver.execute_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_untouched_forgotten_fact_is_left_alone(self) -> None:
        client = _client([_row()])
        result = _result(_edge("live"), cites=["live"])

        await recall_ingest.keep_forgotten(client, _before(_row()), result, [], None)

        assert _writes(client) == []
        assert [edge.uuid for edge in result.edges] == ["live"]

    @pytest.mark.asyncio
    async def test_a_forgotten_fact_graphiti_invalidated_is_put_back(self) -> None:
        """A contradiction stamps ``invalid_at`` and lists the edge on the new
        episode, which would hide the episode with it."""
        stamped = _row(invalid_at=_NOW.isoformat())
        client = _client([stamped])
        result = _result(_edge("live"), _edge("f1"), cites=["live", "f1"])
        extract = AsyncMock()

        with patch.object(recall_ingest, "extract_edges", extract):
            await recall_ingest.keep_forgotten(
                client, _before(_row()), result, [], None
            )

        assert _writes(client) == [
            (recall_ingest._RESTORE_QUERY, {"uuid": "f1", "fields": _FORGOTTEN}),
            (
                recall_ingest._REPOINT_QUERY,
                {"uuid": "ep-new", "forgotten": ["f1"], "added": []},
            ),
        ]
        extract.assert_not_awaited()
        assert [edge.uuid for edge in result.edges] == ["live"]

    @pytest.mark.asyncio
    async def test_a_fact_stated_again_gets_a_new_live_edge(self) -> None:
        """graphiti merged the new episode into the forgotten edge: the
        episode's own sentence becomes a new live fact."""
        client = _client([_ABSORBED])
        result = _result(_edge("f1"), cites=["f1"])

        _, save = await _keep_restating(client, result, [_edge("extracted")])

        [new] = result.edges
        assert new.uuid not in ("f1", "extracted")
        assert (new.source_node_uuid, new.target_node_uuid) == ("alice", "atlas")
        assert (new.fact, new.episodes, new.valid_at) == (_SENTENCE, ["ep-new"], _NOW)
        assert new.expired_at is None and new.invalid_at is None
        assert new.attributes == {
            "status": "active",
            "source_kind": "user_asserted",
            "scope": "real:global",
        }
        save.assert_awaited_once_with(client.driver)

    @pytest.mark.asyncio
    async def test_the_sentence_is_graphitis_own_extraction_of_the_episode(
        self,
    ) -> None:
        """Extracted as ``add_episode`` would, with its earlier episodes and
        instructions; the forgotten edge is put back and the new episode
        cites the new edge instead."""
        client = _client([_ABSORBED])
        result = _result(_edge("f1"), cites=["f1"])

        extract, _ = await _keep_restating(
            client, result, [_edge("extracted")], "instructions"
        )

        extract.assert_awaited_once_with(
            client.clients,
            result.episode,
            result.nodes,
            [],
            EDGE_TYPE_MAP,
            _GROUP,
            EDGE_TYPES,
            "instructions",
        )
        [new] = result.edges
        assert _writes(client) == [
            (recall_ingest._RESTORE_QUERY, {"uuid": "f1", "fields": _FORGOTTEN}),
            (
                recall_ingest._REPOINT_QUERY,
                {"uuid": "ep-new", "forgotten": ["f1"], "added": [new.uuid]},
            ),
        ]

    @pytest.mark.asyncio
    async def test_no_sentence_found_still_frees_the_new_episode(self) -> None:
        client = _client([_ABSORBED])
        result = _result(_edge("f1"), cites=["f1"])

        await _keep_restating(client, result, [_edge("extracted", source="bob")])

        assert result.edges == []
        assert _writes(client)[-1] == (
            recall_ingest._REPOINT_QUERY,
            {"uuid": "ep-new", "forgotten": ["f1"], "added": []},
        )

    @pytest.mark.asyncio
    async def test_a_failed_repair_leaves_the_result_as_graphiti_wrote_it(
        self,
    ) -> None:
        client = MagicMock()
        client.driver.execute_query = AsyncMock(side_effect=RuntimeError("down"))
        result = _result(_edge("f1"), cites=["f1"])

        await recall_ingest.keep_forgotten(client, _before(_row()), result, [], None)

        assert [edge.uuid for edge in result.edges] == ["f1"]


class TestQueries:
    def test_the_repair_reads_by_uuid_not_by_the_markers_it_repairs(self) -> None:
        assert "WHERE e.uuid IN $uuids" in recall_ingest._READ_QUERY
        assert "forgotten_at IS NOT NULL" not in recall_ingest._READ_QUERY

    def test_a_restore_sets_the_snapshot_back_and_keeps_the_embedding(self) -> None:
        assert "SET e += $fields" in recall_ingest._RESTORE_QUERY
        assert "fact_embedding" not in recall_ingest._FIELDS

    def test_the_new_episode_stops_citing_forgotten_facts(self) -> None:
        query = recall_ingest._REPOINT_QUERY
        assert "MATCH (ep:Episodic {uuid: $uuid})" in query
        assert (
            "[x IN coalesce(ep.entity_edges, []) WHERE NOT x IN $forgotten] + $added"
            in query
        )
