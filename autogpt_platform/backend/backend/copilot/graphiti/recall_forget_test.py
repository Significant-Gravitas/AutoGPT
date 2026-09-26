"""Unit tests for ``recall_forget.retract`` against a mock driver.

Pin the Cypher each forget mode issues and the per-uuid failure reporting;
``recall_integration_test.py`` runs the same calls against FalkorDB.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import recall_forget
from .memory_model import FORGET_NO_MATCH_REASON, MemoryForgetFailureCode
from .recall import live_fact_predicate
from .scope import MemoryScope

_SCOPE = MemoryScope.for_user("user-abc")


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
        assert (
            "SET e.expired_at = $now, e.status = $status, "
            "e.expiration_reason = $reason" in write
        )
        assert "invalid_at" not in write, "a forget is not a world change"
        assert "datetime()" not in write, "FalkorDB has no no-arg datetime()"
        assert write_kwargs["uuid"] == "u1"
        assert write_kwargs["group_id"] == _SCOPE.group_id
        assert write_kwargs["status"] == "retracted"
        assert write_kwargs["reason"] == "user_signal"

        redact, redact_kwargs = _call(driver, 2)
        assert "SET ep.redacted_at = coalesce(ep.redacted_at, $now)" in redact
        assert "any(x IN ep.entity_edges WHERE x IN $uuids)" in redact
        assert live_fact_predicate("live") in redact
        assert redact_kwargs == {"uuids": ["u1"], "now": write_kwargs["now"]}

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
    async def test_redaction_failure_still_reports_the_retraction(self) -> None:
        driver = _driver([{"uuid": "u1"}], [{"uuid": "u1"}], RuntimeError("down"))

        result = await _retract(driver, ["u1"])

        assert result.deleted == ["u1"]
        assert result.failures == []
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
    async def test_deletes_edge_then_orphaned_episodes_then_orphaned_entities(
        self,
    ) -> None:
        driver = _driver(
            [{"uuid": "u1"}],
            [{"uuid": "u1", "source_uuid": "alice", "target_uuid": "atlas"}],
            [{"uuid": "ep1", "mentioned": ["alice", "carol"]}],
            [],  # delete orphaned episodes
            [],  # drop back-references
            [{"uuid": "alice"}, {"uuid": "carol"}],
        )

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == ["u1"]
        assert result.deleted_episodes == ["ep1"]
        assert result.deleted_entities == ["alice", "carol"]
        assert result.redacted_episodes == []
        driver.close.assert_awaited_once()

        delete_edge, delete_kwargs = _call(driver, 1)
        assert "WITH e, e.uuid AS uuid" in delete_edge
        assert "DELETE e" in delete_edge
        assert delete_kwargs == {"uuid": "u1", "group_id": _SCOPE.group_id}

        orphans, orphans_kwargs = _call(driver, 2)
        assert live_fact_predicate("live") in orphans
        assert "collect(n.uuid) AS mentioned" in orphans
        assert orphans_kwargs == {"uuids": ["u1"]}

        delete_episodes, episodes_kwargs = _call(driver, 3)
        assert "DETACH DELETE ep" in delete_episodes
        assert episodes_kwargs == {"uuids": ["ep1"]}

        backrefs, backrefs_kwargs = _call(driver, 4)
        assert "SET ep.entity_edges = [x IN ep.entity_edges WHERE NOT x IN $uuids]" in (
            backrefs
        )
        assert backrefs_kwargs == {"uuids": ["u1"]}

        entities, entities_kwargs = _call(driver, 5)
        assert "OPTIONAL MATCH (n)-[r:RELATES_TO|MENTIONS]-()" in entities
        assert "DETACH DELETE n" in entities
        # Endpoints of the deleted edge plus what the deleted episode mentioned.
        assert entities_kwargs == {"uuids": ["alice", "atlas", "carol"]}

    @pytest.mark.asyncio
    async def test_no_orphaned_episode_skips_the_episode_delete(self) -> None:
        driver = _driver(
            [{"uuid": "u1"}],
            [{"uuid": "u1", "source_uuid": "alice", "target_uuid": "atlas"}],
            [],  # every episode still has a live fact
            [],  # drop back-references
            [],  # no orphaned entity
        )

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == ["u1"]
        assert result.deleted_episodes == []
        assert driver.execute_query.await_count == 5
        assert "SET ep.entity_edges" in _call(driver, 3)[0]

    @pytest.mark.asyncio
    async def test_clean_up_failure_still_reports_the_deletion(self) -> None:
        driver = _driver(
            [{"uuid": "u1"}],
            [{"uuid": "u1", "source_uuid": "alice", "target_uuid": "atlas"}],
            RuntimeError("down"),
        )

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == ["u1"]
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_unmatched_delete_is_a_no_match(self) -> None:
        driver = _driver([{"uuid": "u1"}], [])

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == []
        assert [f.code for f in result.failures] == [MemoryForgetFailureCode.NO_MATCH]
        assert driver.execute_query.await_count == 2
