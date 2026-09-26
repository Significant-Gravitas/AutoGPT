"""Unit tests for ``recall_forget.retract`` against a mock driver and the
in-memory forget stash (``conftest.forget_stash``).

Pin the Cypher each forget mode issues, its order, what it stashes before
touching the graph, and the per-uuid failure reporting. What hiding and
purging send is pinned in ``recall_hide_test.py`` and
``recall_orphans_test.py``; the live runs are
``recall_forget_integration_test.py``, ``recall_hard_forget_integration_test.py``
and ``recall_integration_test.py``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import recall_forget, recall_hide, recall_orphans
from .memory_model import FORGET_NO_MATCH_REASON, MemoryForgetFailureCode
from .recall import FORGOTTEN_FACT
from .recall_fake_redis import FakeRedis
from .recall_stash import ForgetRecord, read_forgets, stash_forgets
from .scope import MemoryScope

_SCOPE = MemoryScope.for_user("user-abc")
_GROUP = _SCOPE.group_id
_CLEANUP = MemoryForgetFailureCode.CLEANUP_ERROR
_EDGE = {
    "uuid": "u1",
    "fact": "Alice works on Atlas",
    "name": "MemoryFact",
    "episodes": ["ep1"],
    "source": "alice",
    "target": "atlas",
    "source_name": "Alice",
    "target_name": "Atlas",
    "citing": ["ep1"],
}


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


def _queries(driver: AsyncMock) -> list[str]:
    return [call.args[0] for call in driver.execute_query.await_args_list]


def _call(driver: AsyncMock, index: int) -> tuple[str, dict]:
    call = driver.execute_query.await_args_list[index]
    return call.args[0], call.kwargs


# lookup, retract, scrub facts, entity keys (none), redact
_SOFT = ([_EDGE], [{"uuid": "u1"}], [{"ends": []}], [], [{"uuid": "ep1"}])


class TestSoftRetract:
    @pytest.mark.asyncio
    async def test_stashes_then_marks_the_edge_then_hides_its_text(
        self, forget_stash: FakeRedis
    ) -> None:
        driver = _driver(*_SOFT)

        result = await _retract(driver, ["u1"])

        assert (result.deleted, result.failures) == (["u1"], [])
        assert result.redacted_episodes == ["ep1"]
        driver.close.assert_awaited_once()
        record = (await read_forgets(_GROUP))["u1"]
        assert record.fact_redacted == "Alice works on Atlas"
        assert record.name_redacted == "MemoryFact"
        assert (record.episodes, record.redacted_episodes) == (["ep1"], ["ep1"])
        assert (record.source_name, record.target_name) == ("Alice", "Atlas")
        write, kwargs = _call(driver, 1)
        assert "SET e.forgotten_at = coalesce(e.forgotten_at, $forgotten_at)," in write
        assert "e.expired_at = coalesce(e.expired_at, $expired_at)," in write
        assert "invalid_at" not in write, "a forget is not a world change"
        assert kwargs["forgotten_at"] == record.forgotten_at
        assert (kwargs["status"], kwargs["reason"]) == ("retracted", "user_signal")
        assert _queries(driver)[2] == recall_hide.SCRUB_FACTS_QUERY
        assert _call(driver, 2)[1]["recovered"] == [
            ["u1", "Alice works on Atlas", "MemoryFact"]
        ]
        assert _queries(driver)[4] == recall_hide.REDACT_EPISODES_QUERY

    @pytest.mark.asyncio
    async def test_the_lookup_is_a_read_that_finds_the_edges_sources(self) -> None:
        driver = _driver(*_SOFT)

        await _retract(driver, ["u1"])

        lookup, kwargs = _call(driver, 0)
        assert "SET" not in lookup, "the lookup must be a read"
        assert "collect(ep.uuid) AS citing" in lookup
        assert kwargs == {"uuids": ["u1"], "group_id": _GROUP}

    @pytest.mark.asyncio
    async def test_a_repeat_recovers_what_a_failed_repair_lost(self) -> None:
        """graphiti took the edge's marker, audit copies and expiry and
        stamped its own ``invalid_at``: the first forget's stashed values
        are written back and stashed again, not new ones."""
        first = ForgetRecord(
            uuid="u1",
            forgotten_at="2026-01-01T00:00:00+00:00",
            expired_at="2026-01-01T00:00:00+00:00",
            fact_redacted="Alice works on Atlas",
            name_redacted="MemoryFact",
            dropped_episodes=["ep-again"],
            stashed_at=datetime.now(timezone.utc).isoformat(),
        )
        await stash_forgets(_GROUP, [first])
        wiped = {**_EDGE, "fact": FORGOTTEN_FACT, "name": FORGOTTEN_FACT}
        wiped["invalid_at"] = "2026-06-01T00:00:00+00:00"
        driver = _driver([wiped], [{"uuid": "u1"}], [{"ends": []}], [], [])

        await _retract(driver, ["u1"])

        _, kwargs = _call(driver, 1)
        assert kwargs["forgotten_at"] == first.forgotten_at
        assert kwargs["expired_at"] == first.expired_at
        assert kwargs["dropped"] == ["ep-again"]
        assert (await read_forgets(_GROUP))["u1"].invalid_at is None
        assert _call(driver, 2)[1]["recovered"] == [
            ["u1", "Alice works on Atlas", "MemoryFact"]
        ]

    @pytest.mark.asyncio
    async def test_the_stash_is_written_before_the_graph_is_touched(self) -> None:
        driver = _driver(*_SOFT)
        order = MagicMock()
        order.attach_mock(driver.execute_query, "query")
        order.attach_mock(AsyncMock(return_value=True), "stash")

        with patch.object(recall_forget, "stash_forgets", order.stash):
            await _retract(driver, ["u1"])

        steps = [name for name, _, _ in order.mock_calls[:3]]
        assert steps == ["query", "stash", "query"], "lookup, stash, write"

    @pytest.mark.asyncio
    async def test_redis_being_down_does_not_stop_the_forget(self) -> None:
        driver = _driver(*_SOFT)

        with patch.object(
            recall_forget, "stash_forgets", AsyncMock(return_value=False)
        ):
            result = await _retract(driver, ["u1"])

        assert (result.deleted, result.failures) == (["u1"], [])

    @pytest.mark.asyncio
    async def test_reason_is_recorded(self) -> None:
        driver = _driver(*_SOFT)

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
            [{**_EDGE, "uuid": "errored"}, {**_EDGE, "uuid": "vanished"}],
            RuntimeError("boom"),
            [],  # deleted between the lookup and the write
        )

        result = await _retract(driver, ["errored", "vanished"])

        by_uuid = {f.uuid: f for f in result.failures}
        assert by_uuid["errored"].code == MemoryForgetFailureCode.QUERY_ERROR
        assert "boom" in by_uuid["errored"].reason
        assert by_uuid["vanished"].code == MemoryForgetFailureCode.NO_MATCH
        assert driver.execute_query.await_count == 3, "nothing to hide"

    @pytest.mark.asyncio
    async def test_a_failed_hide_is_a_cleanup_error_on_each_edge(self) -> None:
        driver = _driver(
            [_EDGE, {**_EDGE, "uuid": "u2"}],
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

    @pytest.mark.asyncio
    async def test_repeated_uuids_are_forgotten_once(self) -> None:
        driver = _driver(*_SOFT)

        result = await _retract(driver, ["u1", "u1"])

        assert result.deleted == ["u1"] and result.failures == []
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
    async def test_hides_everything_before_it_purges_anything(self) -> None:
        driver = _driver(
            *_SOFT[:4],
            [{"uuid": "ep1"}, {"uuid": "ep2"}],  # redact
            [{"uuid": "ep1", "content": "Alice works on Atlas"}],  # citing
            [{"uuid": "ep1"}],  # tombstone
            [{"uuid": "u1", "deleted_entities": ["alice", "carol"]}],  # delete
        )

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == ["u1"] and result.failures == []
        assert result.tombstoned_episodes == ["ep1"]
        assert result.redacted_episodes == ["ep2"], "kept for another edge, hidden"
        assert result.deleted_entities == ["alice", "carol"]
        assert _queries(driver)[4:] == [
            recall_hide.REDACT_EPISODES_QUERY,
            recall_orphans._CITING_EPISODES_QUERY,
            recall_orphans._TOMBSTONE_QUERY,
            recall_orphans._DELETE_EDGE_QUERY,
        ]
        assert (await read_forgets(_GROUP))["u1"].hard

    @pytest.mark.asyncio
    async def test_a_failed_hide_stops_it_before_any_purge(self) -> None:
        driver = _driver([_EDGE], [{"uuid": "u1"}], RuntimeError("down"))

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == [], "a deleted edge could no longer hide its text"
        assert [(f.uuid, f.code) for f in result.failures] == [("u1", _CLEANUP)]
        assert driver.execute_query.await_count == 3

    @pytest.mark.asyncio
    async def test_unmatched_delete_is_a_no_match(self) -> None:
        driver = _driver(*_SOFT[:4], [], [], [], [])

        result = await _retract(driver, ["u1"], hard=True)

        assert result.deleted == []
        assert [f.code for f in result.failures] == [MemoryForgetFailureCode.NO_MATCH]
        assert driver.execute_query.await_count == 8
