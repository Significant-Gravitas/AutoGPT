"""Unit tests for ``recall_orphans.purge``, a hard forget's last step,
against a mock driver: the tombstones, the per-edge delete, and that every
step before the delete is safe to repeat.

The live run is ``recall_hard_forget_integration_test.py``.
"""

import json
from unittest.mock import AsyncMock

import pytest

from . import recall_orphans
from .memory_model import ForgetResult, MemoryForgetFailureCode
from .scope import MemoryScope

_SCOPE = MemoryScope.for_user("user-abc")
_NOW = "2026-09-26T12:00:00+00:00"
_CLEANUP = MemoryForgetFailureCode.CLEANUP_ERROR
_ENVELOPE = json.dumps(
    {"content": "Alice works on Atlas", "provenance": "session:s1#msg:3"}
)


def _driver(*results) -> AsyncMock:
    driver = AsyncMock()
    driver.execute_query.side_effect = [
        r if isinstance(r, Exception) else (r, [], None) for r in results
    ]
    return driver


def _deleted(uuid: str, *entities: str) -> list[dict]:
    return [{"uuid": uuid, "deleted_entities": list(entities)}]


class TestPurge:
    @pytest.mark.asyncio
    async def test_tombstones_then_deletes_each_edge(self) -> None:
        driver = _driver(
            [
                {"uuid": "ep-chat", "content": "Alice works on Atlas"},
                {"uuid": "ep-memory", "content": _ENVELOPE},
            ],
            [{"uuid": "ep-chat"}, {"uuid": "ep-memory"}],
            _deleted("u1", "alice"),
            _deleted("u2"),
        )
        result = ForgetResult(redacted_episodes=["ep-chat", "ep-memory", "ep-kept"])

        await recall_orphans.purge(driver, _SCOPE, ["u1", "u2"], _NOW, result)

        assert result.deleted == ["u1", "u2"] and result.failures == []
        assert result.tombstoned_episodes == ["ep-chat", "ep-memory"]
        assert result.redacted_episodes == [
            "ep-kept"
        ], "a tombstone is not listed twice"
        assert result.deleted_entities == ["alice"]
        citing, tombstone, *deletes = driver.execute_query.await_args_list
        assert citing.args == (recall_orphans._CITING_EPISODES_QUERY,)
        assert citing.kwargs == {"uuids": ["u1", "u2"]}
        assert tombstone.args == (recall_orphans._TOMBSTONE_QUERY,)
        assert tombstone.kwargs == {
            "uuids": ["u1", "u2"],
            "provenance": [["ep-memory", "session:s1#msg:3"]],
            "now": _NOW,
        }
        assert [(d.args, d.kwargs) for d in deletes] == [
            (
                (recall_orphans._DELETE_EDGE_QUERY,),
                {"uuid": u, "group_id": _SCOPE.group_id},
            )
            for u in ("u1", "u2")
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing", [0, 1], ids=["citing-read", "tombstone"])
    async def test_a_failed_tombstone_step_deletes_nothing(self, failing: int) -> None:
        """The edges stay, so forgetting again finds them and finishes."""
        results: list = [[], []]
        results[failing] = RuntimeError("down")
        driver = _driver(*results)
        result = ForgetResult()

        await recall_orphans.purge(driver, _SCOPE, ["u1", "u2"], _NOW, result)

        assert result.deleted == []
        assert [(f.uuid, f.code) for f in result.failures] == [
            ("u1", _CLEANUP),
            ("u2", _CLEANUP),
        ]
        queries = [call.args[0] for call in driver.execute_query.await_args_list]
        assert recall_orphans._DELETE_EDGE_QUERY not in queries

    @pytest.mark.asyncio
    async def test_a_failed_delete_is_that_edges_cleanup_error(self) -> None:
        driver = _driver([], [], _deleted("u1"), RuntimeError("down"))
        result = ForgetResult()

        await recall_orphans.purge(driver, _SCOPE, ["u1", "u2"], _NOW, result)

        assert result.deleted == ["u1"]
        assert [(f.uuid, f.code) for f in result.failures] == [("u2", _CLEANUP)]

    @pytest.mark.asyncio
    async def test_an_edge_gone_before_its_delete_is_a_no_match(self) -> None:
        driver = _driver([], [], [])
        result = ForgetResult()

        await recall_orphans.purge(driver, _SCOPE, ["u1"], _NOW, result)

        assert result.deleted == []
        assert [f.code for f in result.failures] == [MemoryForgetFailureCode.NO_MATCH]


class TestTombstoneQuery:
    def test_an_episode_is_emptied_only_when_no_remaining_edge_cites_it(
        self,
    ) -> None:
        """Ownership, whatever the citing edge's status (a retracted edge kept
        for audit still needs its source), checked in the writing query."""
        query = recall_orphans._TOMBSTONE_QUERY
        assert (
            "AND (ref.uuid IN ep.entity_edges OR ep.uuid IN coalesce(ref.episodes, []))"
            in query
        )
        assert "WHERE NOT ref.uuid IN $uuids" in query
        assert "WHERE refs = 0" in query
        assert "status" not in query and "expired_at" not in query

    def test_a_tombstone_keeps_what_names_its_session(self) -> None:
        query = recall_orphans._TOMBSTONE_QUERY
        assert "ep.content = ''" in query
        assert "DELETE" not in query, "an episode is never deleted"
        for kept in ("ep.name", "ep.source_description", "ep.entity_edges"):
            assert f"{kept} =" not in query
        assert "ep.provenance = coalesce(" in query
        assert "[p IN $provenance WHERE p[0] = ep.uuid | p[1]][0]" in query

    def test_a_repeat_keeps_the_first_stamps(self) -> None:
        query = recall_orphans._TOMBSTONE_QUERY
        assert "ep.redacted_at = coalesce(ep.redacted_at, $now)" in query
        assert "ep.hard_deleted_at = coalesce(ep.hard_deleted_at, $now)" in query


class TestDeleteEdgeQuery:
    def test_one_query_deletes_the_edge_and_what_it_alone_kept(self) -> None:
        query = recall_orphans._DELETE_EDGE_QUERY
        assert query.count("MATCH (source)-[e:MENTIONS|RELATES_TO|HAS_MEMBER") == 1
        assert "WHERE e.group_id = $group_id OR e.group_id IS NULL" in query
        assert "SET ep.entity_edges = [x IN ep.entity_edges WHERE x <> uuid]" in query
        assert "WHERE tomb.hard_deleted_at IS NOT NULL" in query
        assert "FOREACH (m IN mentions | DELETE m)" in query
        assert "FOREACH (orphan IN orphans | DETACH DELETE orphan)" in query
        assert "RETURN uuid, orphan_uuids AS deleted_entities" in query

    def test_ids_are_read_before_anything_is_deleted(self) -> None:
        """FalkorDB cannot read a deleted element."""
        query = recall_orphans._DELETE_EDGE_QUERY
        capture = query.index(
            "WITH e, e.uuid AS uuid, source.uuid AS source_uuid, "
            "target.uuid AS target_uuid"
        )
        assert capture < query.index("DELETE e\n")

    def test_an_entity_goes_only_with_no_fact_and_no_mention(self) -> None:
        query = recall_orphans._DELETE_EDGE_QUERY
        assert "OPTIONAL MATCH (candidate)-[link:RELATES_TO|MENTIONS]-()" in query
        assert "CASE WHEN links = 0 THEN candidate END" in query
