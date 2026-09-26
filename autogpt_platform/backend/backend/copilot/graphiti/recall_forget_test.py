"""Unit tests for ``recall_forget.retract`` against a mock driver.

Pin the Cypher each forget mode issues, its order and the per-uuid failure
reporting; ``recall_forget_integration_test.py`` and
``recall_integration_test.py`` run the same calls against FalkorDB.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import recall_forget, recall_orphans
from .memory_model import FORGET_NO_MATCH_REASON, MemoryForgetFailureCode
from .recall import forgotten_facts_clause, recallable_episode_predicate
from .scope import MemoryScope

_SCOPE = MemoryScope.for_user("user-abc")
_CLEANUP = MemoryForgetFailureCode.CLEANUP_ERROR
_EDGE_ROW = {"uuid": "u1", "source_uuid": "alice", "target_uuid": "atlas"}


def _driver(*results) -> AsyncMock:
    """A driver whose queries return (or raise) ``results`` in order."""
    driver = AsyncMock()
    driver.execute_query.side_effect = [
        r if isinstance(r, Exception) else (r, [], None) for r in results
    ]
    return driver


async def _retract(driver: AsyncMock, uuids: list[str], **kwargs):
    with patch.object(recall_forget, "open_driver", MagicMock(return_value=driver)):
        return await recall_forget.retract(_SCOPE, uuids, **kwargs)


def _call(driver: AsyncMock, index: int) -> tuple[str, dict]:
    call = driver.execute_query.await_args_list[index]
    return call.args[0], call.kwargs


class TestSoftRetract:
    @pytest.mark.asyncio
    async def test_marks_edge_retracted_and_redacts_its_episodes(self) -> None:
        driver = _driver([{"uuid": "u1"}], [{"uuid": "u1"}], [{"uuid": "ep1"}])

        result = await _retract(driver, ["u1"])

        assert result.deleted == ["u1"]
        assert result.failures == []
        assert result.redacted_episodes == ["ep1"]
        assert driver.execute_query.await_count == 3
        driver.close.assert_awaited_once()

        lookup, lookup_kwargs = _call(driver, 0)
        assert "RETURN DISTINCT e.uuid AS uuid" in lookup
        assert "SET" not in lookup, "the lookup must be a read"
        assert lookup_kwargs == {"uuids": ["u1"], "group_id": _SCOPE.group_id}

        write, write_kwargs = _call(driver, 1)
        assert "SET e.expired_at = coalesce(e.expired_at, $now)," in write
        assert "e.status = $status," in write
        assert "e.expiration_reason = $reason" in write
        assert "invalid_at" not in write, "a forget is not a world change"
        assert "datetime()" not in write, "FalkorDB has no no-arg datetime()"
        assert write_kwargs["uuid"] == "u1"
        assert write_kwargs["group_id"] == _SCOPE.group_id
        assert write_kwargs["status"] == "retracted"
        assert write_kwargs["reason"] == "user_signal"

    @pytest.mark.asyncio
    async def test_redacts_every_episode_the_policy_now_hides(self) -> None:
        """Any episode naming a forgotten fact, not only one left with no
        live fact: the redaction and the read side share one predicate."""
        driver = _driver([{"uuid": "u1"}], [{"uuid": "u1"}], [{"uuid": "ep1"}])

        await _retract(driver, ["u1"])

        redact, redact_kwargs = _call(driver, 2)
        assert redact.startswith(forgotten_facts_clause())
        assert "any(x IN coalesce(ep.entity_edges, []) WHERE x IN $uuids)" in redact
        assert f"NOT ({recallable_episode_predicate('ep')})" in redact
        assert "SET ep.redacted_at = coalesce(ep.redacted_at, $now)" in redact
        assert redact_kwargs == {"uuids": ["u1"], "now": _call(driver, 1)[1]["now"]}

    @pytest.mark.asyncio
    async def test_reason_is_recorded(self) -> None:
        driver = _driver([{"uuid": "u1"}], [{"uuid": "u1"}], [])

        await _retract(driver, ["u1"], reason="settings_page")

        assert _call(driver, 1)[1]["reason"] == "settings_page"

    @pytest.mark.asyncio
    async def test_unknown_uuid_is_a_no_match_and_nothing_is_written(self) -> None:
        driver = _driver([])

        result = await _retract(driver, ["missing"])

        assert result.deleted == []
        assert [(f.uuid, f.code) for f in result.failures] == [
            ("missing", MemoryForgetFailureCode.NO_MATCH)
        ]
        assert result.failures[0].reason == FORGET_NO_MATCH_REASON
        assert driver.execute_query.await_count == 1

    @pytest.mark.asyncio
    async def test_lookup_error_fails_every_uuid_with_its_reason(self) -> None:
        driver = _driver(RuntimeError("Unknown function 'datetime'"))

        result = await _retract(driver, ["u1", "u2"])

        assert result.deleted == []
        assert [f.uuid for f in result.failures] == ["u1", "u2"]
        for failure in result.failures:
            assert failure.code == MemoryForgetFailureCode.QUERY_ERROR
            assert failure.reason == (
                "Deletion query failed: RuntimeError: Unknown function 'datetime'"
            )
        driver.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_error_and_vanished_edge_are_told_apart(self) -> None:
        driver = _driver(
            [{"uuid": "errored"}, {"uuid": "vanished"}],
            RuntimeError("boom"),
            [],  # deleted between the lookup and the write
        )

        result = await _retract(driver, ["errored", "vanished"])

        by_uuid = {f.uuid: f for f in result.failures}
        assert by_uuid["errored"].code == MemoryForgetFailureCode.QUERY_ERROR
        assert "boom" in by_uuid["errored"].reason
        assert by_uuid["vanished"].code == MemoryForgetFailureCode.NO_MATCH
        # Nothing was retracted, so no episode is looked at.
        assert driver.execute_query.await_count == 3

    @pytest.mark.asyncio
    async def test_redaction_failure_is_a_cleanup_error_on_each_edge(self) -> None:
        driver = _driver(
            [{"uuid": "u1"}, {"uuid": "u2"}],
            [{"uuid": "u1"}],
            [{"uuid": "u2"}],
            RuntimeError("down"),
        )

        result = await _retract(driver, ["u1", "u2"])

        assert result.deleted == ["u1", "u2"], "the facts themselves are forgotten"
        assert [(f.uuid, f.code) for f in result.failures] == [
            ("u1", _CLEANUP),
            ("u2", _CLEANUP),
        ]
        assert "RuntimeError: down" in result.failures[0].reason
        assert result.redacted_episodes == []

    @pytest.mark.asyncio
    async def test_repeated_uuids_are_forgotten_once(self) -> None:
        driver = _driver([{"uuid": "u1"}], [{"uuid": "u1"}], [])

        result = await _retract(driver, ["u1", "u1"])

        assert result.deleted == ["u1"]
        assert _call(driver, 0)[1]["uuids"] == ["u1"]

    @pytest.mark.asyncio
    async def test_empty_request_opens_no_driver(self) -> None:
        open_driver = MagicMock()
        with patch.object(recall_forget, "open_driver", open_driver):
            result = await recall_forget.retract(_SCOPE, [])

        assert result.deleted == [] and result.failures == []
        open_driver.assert_not_called()


class TestHardRetract:
    @pytest.mark.asyncio
    async def test_hides_everything_before_it_deletes_anything(self) -> None:
        driver = _driver(
            [{"uuid": "u1"}],  # lookup
            [{"uuid": "u1"}],  # retract
            [{"uuid": "ep1"}, {"uuid": "ep2"}],  # redact
            [_EDGE_ROW],  # delete the edge
            [{"uuid": "ep1", "mentioned": ["alice", "carol"]}],  # orphan episodes
            [],  # drop back-references
            [{"uuid": "alice"}, {"uuid": "carol"}],  # orphan entities
        )

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == ["u1"] and result.failures == []
        assert result.deleted_episodes == ["ep1"]
        assert result.redacted_episodes == ["ep2"], "kept for another edge, hidden"
        assert result.deleted_entities == ["alice", "carol"]
        driver.close.assert_awaited_once()
        queries = [call.args[0] for call in driver.execute_query.await_args_list]
        assert "e.status = $status" in queries[1]
        assert "SET ep.redacted_at" in queries[2]
        assert "DELETE e\n" in queries[3] and "WITH e, e.uuid AS uuid" in queries[3]
        assert queries[4] == recall_orphans._DELETE_ORPHANED_EPISODES_QUERY
        assert "SET ep.entity_edges" in queries[5]
        assert "DETACH DELETE n" in queries[6]
        assert _call(driver, 3)[1] == {"uuid": "u1", "group_id": _SCOPE.group_id}
        assert _call(driver, 4)[1] == {"uuids": ["u1"]}
        assert _call(driver, 5)[1] == {"uuids": ["u1"]}
        # Endpoints of the deleted edge plus what the deleted episode mentioned.
        assert _call(driver, 6)[1] == {"uuids": ["alice", "atlas", "carol"]}

    def test_an_episode_goes_only_when_no_remaining_edge_cites_it(self) -> None:
        """Ownership, whatever the citing edge's status (a retracted edge kept
        for audit still needs its source), checked in the deleting query."""
        query = recall_orphans._DELETE_ORPHANED_EPISODES_QUERY

        assert (
            "WHERE ref.uuid IN ep.entity_edges OR ep.uuid IN coalesce(ref.episodes, [])"
            in query
        )
        assert "WHERE refs = 0" in query
        assert "status" not in query and "expired_at" not in query
        assert "DETACH DELETE ep" in query

    @pytest.mark.asyncio
    async def test_redaction_failure_stops_it_before_any_delete(self) -> None:
        driver = _driver([{"uuid": "u1"}], [{"uuid": "u1"}], RuntimeError("down"))

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == [], "a deleted edge could no longer hide its text"
        assert [(f.uuid, f.code) for f in result.failures] == [("u1", _CLEANUP)]
        assert driver.execute_query.await_count == 3

    @pytest.mark.asyncio
    async def test_clean_up_failure_is_a_cleanup_error_on_each_deleted_edge(
        self,
    ) -> None:
        driver = _driver(
            [{"uuid": "u1"}],
            [{"uuid": "u1"}],
            [{"uuid": "ep1"}],
            [_EDGE_ROW],
            RuntimeError("down"),
        )

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == ["u1"]
        assert [(f.uuid, f.code) for f in result.failures] == [("u1", _CLEANUP)]
        assert result.redacted_episodes == ["ep1"], "still hidden"

    @pytest.mark.asyncio
    async def test_unmatched_delete_is_a_no_match(self) -> None:
        driver = _driver([{"uuid": "u1"}], [{"uuid": "u1"}], [], [])

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == []
        assert [f.code for f in result.failures] == [MemoryForgetFailureCode.NO_MATCH]
        assert driver.execute_query.await_count == 4
