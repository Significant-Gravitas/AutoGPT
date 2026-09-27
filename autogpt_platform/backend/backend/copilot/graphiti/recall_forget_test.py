"""Unit tests for ``recall_forget.retract`` against a mock driver and the
in-memory write lock (``conftest.lock_redis``).

Pin the Cypher each forget mode issues, its order, the write lock it holds
around every write, and the per-uuid failure reporting. What hiding and
purging send is pinned in ``recall_hide_test.py`` and
``recall_orphans_test.py``; the live runs are
``recall_forget_integration_test.py``, ``recall_hard_forget_integration_test.py``,
``recall_inflight_integration_test.py`` and ``recall_integration_test.py``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import recall_forget, recall_hide, recall_orphans, scope_lock
from .memory_model import (
    FORGET_BUSY_REASON,
    FORGET_NO_MATCH_REASON,
    MemoryForgetFailureCode,
)
from .recall import FORGOTTEN_FACT
from .recall_fake_redis import FakeRedis
from .scope import MemoryScope, write_lock_key

_SCOPE = MemoryScope.for_user("user-abc")
_LOCK = write_lock_key(_SCOPE.group_id)
_CLEANUP = MemoryForgetFailureCode.CLEANUP_ERROR


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
_SOFT = ([{"uuid": "u1"}], [{"uuid": "u1"}], [{"ends": []}], [], [{"uuid": "ep1"}])


class TestSoftRetract:
    @pytest.mark.asyncio
    async def test_marks_the_edge_forgotten_then_hides_its_text(self) -> None:
        driver = _driver(*_SOFT)

        result = await _retract(driver, ["u1"])

        assert (result.deleted, result.failures) == (["u1"], [])
        assert result.redacted_episodes == ["ep1"]
        driver.close.assert_awaited_once()
        lookup, lookup_kwargs = _call(driver, 0)
        assert "SET" not in lookup, "the lookup must be a read"
        assert lookup_kwargs == {"uuids": ["u1"], "group_id": _SCOPE.group_id}
        write, kwargs = _call(driver, 1)
        assert "SET e.forgotten_at = coalesce(e.forgotten_at, $now)," in write
        assert "e.expired_at = coalesce(e.expired_at, $now)," in write
        assert "invalid_at" not in write, "a forget is not a world change"
        assert (kwargs["status"], kwargs["reason"]) == ("retracted", "user_signal")
        assert _call(driver, 2) == (
            recall_hide.SCRUB_FACTS_QUERY,
            {"uuids": ["u1"], "placeholder": FORGOTTEN_FACT},
        )
        assert _call(driver, 4) == (
            recall_hide.REDACT_EPISODES_QUERY,
            {"uuids": ["u1"], "now": kwargs["now"]},
        )

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
            [{"uuid": "errored"}, {"uuid": "vanished"}],
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


class TestWriteLock:
    @pytest.mark.asyncio
    async def test_every_write_happens_under_the_graphs_lock(
        self, lock_redis: FakeRedis
    ) -> None:
        held: list[bool] = []
        results = list(_SOFT)

        async def query(cypher: str, **params: object):
            held.append(_LOCK in lock_redis.values)
            return results.pop(0), [], None

        driver = AsyncMock()
        driver.execute_query.side_effect = query

        await _retract(driver, ["u1"])

        assert held == [True] * len(_SOFT)
        assert _LOCK not in lock_redis.values, "released afterwards"

    @pytest.mark.asyncio
    async def test_a_lock_held_for_the_whole_wait_fails_as_busy_and_writes_nothing(
        self, lock_redis: FakeRedis
    ) -> None:
        lock_redis.values[_LOCK] = "an ingestion's token"
        open_driver = MagicMock()

        with (
            patch.object(recall_forget, "FORGET_LOCK_WAIT_SECONDS", 0),
            patch.object(recall_forget, "open_driver", open_driver),
        ):
            result = await recall_forget.retract(_SCOPE, ["u1", "u2"])

        assert result.deleted == []
        assert [(f.uuid, f.code) for f in result.failures] == [
            ("u1", MemoryForgetFailureCode.BUSY),
            ("u2", MemoryForgetFailureCode.BUSY),
        ]
        assert result.failures[0].reason == FORGET_BUSY_REASON
        open_driver.assert_not_called()
        assert lock_redis.values[_LOCK] == "an ingestion's token", "left alone"

    @pytest.mark.asyncio
    async def test_redis_unreachable_forgets_without_the_lock(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        driver = _driver(*_SOFT)
        down = AsyncMock(side_effect=ConnectionError("redis down"))

        with patch.object(scope_lock, "get_redis_async", down):
            result = await _retract(driver, ["u1"])

        assert (result.deleted, result.failures) == (["u1"], [])
        assert "write lock unavailable" in caplog.text


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

    @pytest.mark.asyncio
    async def test_a_failed_hide_stops_it_before_any_purge(self) -> None:
        driver = _driver([{"uuid": "u1"}], [{"uuid": "u1"}], RuntimeError("down"))

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
