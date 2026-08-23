"""Tests for the delegated-task wake.

Covers the trigger contract:

* a terminal sub-session with parent linkage enqueues exactly one parent turn
* a retry / restart of the same completion enqueues none (Redis claim)
* a parent with a turn in flight receives the wake on its pending buffer
* no linkage, a handoff, a missing parent, an inline waiter, or a flag-off
  user are all silent no-ops
* an enqueue that blows up is swallowed — the sub's completion still stands
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import stream_registry, subsession_wake
from backend.copilot.subsession_wake import _wake_parent, schedule_parent_wake

_SUB = "sub-session-1"
_PARENT = "parent-session-1"
_USER = "user-1"


class _FakeRedis:
    """Async-Redis fake covering only the calls the wake path makes."""

    def __init__(self, store: dict[str, str] | None = None):
        self.store: dict[str, str] = dict(store or {})

    async def set(self, key: str, value: str, *, nx: bool = False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, _ttl: int, value: str):
        self.store[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        for key in keys:
            self.store.pop(key, None)
        return len(keys)


def _session_info(
    *,
    user_id: str = _USER,
    delegated_by: str | None = None,
    handed_off_from: str | None = None,
) -> MagicMock:
    info = MagicMock()
    info.user_id = user_id
    info.metadata.delegated_by_session_id = delegated_by
    info.metadata.handed_off_from_expert_id = handed_off_from
    return info


def _lookup(sessions: dict[str, MagicMock] | None = None) -> AsyncMock:
    """Fake ``get_chat_session_metadata`` keyed by session id."""
    known = sessions or {}

    async def _get(session_id: str, _user_id: str | None = None):
        return known.get(session_id)

    return AsyncMock(side_effect=_get)


def _patched(
    *,
    lookup: AsyncMock,
    redis: _FakeRedis,
    enqueue: AsyncMock,
    flag_enabled: bool = True,
):
    return (
        patch.object(subsession_wake, "get_chat_session_metadata", new=lookup),
        patch.object(
            subsession_wake, "get_redis_async", new=AsyncMock(return_value=redis)
        ),
        patch.object(
            subsession_wake,
            "is_feature_enabled",
            new=AsyncMock(return_value=flag_enabled),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.run_copilot_turn_via_queue",
            new=enqueue,
        ),
    )


@pytest.mark.asyncio
async def test_terminal_status_enqueues_exactly_one_parent_turn():
    lookup = _lookup(
        {
            _SUB: _session_info(delegated_by=_PARENT),
            _PARENT: _session_info(),
        }
    )
    enqueue = AsyncMock(return_value=("running", MagicMock()))
    p1, p2, p3, p4 = _patched(lookup=lookup, redis=_FakeRedis(), enqueue=enqueue)

    with p1, p2, p3, p4:
        await _wake_parent(_SUB, "completed")

    enqueue.assert_awaited_once()
    kwargs = enqueue.await_args.kwargs
    assert kwargs["session_id"] == _PARENT
    assert kwargs["user_id"] == _USER
    # timeout=0 hands the in-flight/idle decision to run_copilot_turn_via_queue
    # instead of occupying the completion worker with a wait.
    assert kwargs["timeout"] == 0
    assert f'sub_session_id="{_SUB}"' in kwargs["message"]
    assert 'status="completed"' in kwargs["message"]
    assert "get_sub_session_result" in kwargs["message"]
    # System-framed, because the pending buffer has no author field.
    assert kwargs["message"].startswith("[System notice")


@pytest.mark.asyncio
async def test_failed_status_is_reported_too():
    lookup = _lookup(
        {
            _SUB: _session_info(delegated_by=_PARENT),
            _PARENT: _session_info(),
        }
    )
    enqueue = AsyncMock(return_value=("running", MagicMock()))
    p1, p2, p3, p4 = _patched(lookup=lookup, redis=_FakeRedis(), enqueue=enqueue)

    with p1, p2, p3, p4:
        await _wake_parent(_SUB, "failed")

    assert 'status="failed"' in enqueue.await_args.kwargs["message"]


@pytest.mark.asyncio
async def test_repeated_completion_does_not_double_post():
    """A retry or a pod restart replaying the same completion must not
    produce a second parent turn — the Redis claim is the dedupe primitive."""
    lookup = _lookup(
        {
            _SUB: _session_info(delegated_by=_PARENT),
            _PARENT: _session_info(),
        }
    )
    enqueue = AsyncMock(return_value=("running", MagicMock()))
    redis = _FakeRedis()
    p1, p2, p3, p4 = _patched(lookup=lookup, redis=redis, enqueue=enqueue)

    with p1, p2, p3, p4:
        await _wake_parent(_SUB, "completed")
        await _wake_parent(_SUB, "completed")

    enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_parent_with_turn_in_flight_goes_to_pending_buffer():
    """End-to-end through the real ``run_copilot_turn_via_queue``: a busy
    parent must receive the wake on its pending buffer, never as a parallel
    turn racing the in-flight one on the cluster lock."""
    lookup = _lookup(
        {
            _SUB: _session_info(delegated_by=_PARENT),
            _PARENT: _session_info(),
        }
    )
    queue_message = AsyncMock(return_value=MagicMock(buffer_length=1))
    schedule = AsyncMock()
    parent_session = MagicMock()
    parent_session.metadata.llm_auth_provider = "platform"
    parent_session.metadata.llm_credential_id = None

    with (
        patch.object(subsession_wake, "get_chat_session_metadata", new=lookup),
        patch.object(
            subsession_wake,
            "get_redis_async",
            new=AsyncMock(return_value=_FakeRedis()),
        ),
        patch.object(
            subsession_wake, "is_feature_enabled", new=AsyncMock(return_value=True)
        ),
        patch(
            "backend.copilot.sdk.session_waiter.get_chat_session",
            new=AsyncMock(return_value=parent_session),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.queue_user_message",
            new=queue_message,
        ),
        patch("backend.copilot.sdk.session_waiter.schedule_turn", new=schedule),
    ):
        await _wake_parent(_SUB, "completed")

    queue_message.assert_awaited_once()
    assert queue_message.await_args.kwargs["session_id"] == _PARENT
    assert f'sub_session_id="{_SUB}"' in queue_message.await_args.kwargs["message"]
    schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_parent_linkage_is_a_noop():
    lookup = _lookup({_SUB: _session_info(delegated_by=None)})
    enqueue = AsyncMock()
    p1, p2, p3, p4 = _patched(lookup=lookup, redis=_FakeRedis(), enqueue=enqueue)

    with p1, p2, p3, p4:
        await _wake_parent(_SUB, "completed")

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_handoff_is_a_noop():
    """A handoff transfers ownership: the receiving expert reports to the
    user directly, and the handing-off session cannot even poll the sub."""
    lookup = _lookup(
        {
            _SUB: _session_info(delegated_by=_PARENT, handed_off_from="expert-9"),
            _PARENT: _session_info(),
        }
    )
    enqueue = AsyncMock()
    p1, p2, p3, p4 = _patched(lookup=lookup, redis=_FakeRedis(), enqueue=enqueue)

    with p1, p2, p3, p4:
        await _wake_parent(_SUB, "completed")

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_off_is_a_noop():
    lookup = _lookup(
        {
            _SUB: _session_info(delegated_by=_PARENT),
            _PARENT: _session_info(),
        }
    )
    enqueue = AsyncMock()
    flag = AsyncMock(return_value=False)

    with (
        patch.object(subsession_wake, "get_chat_session_metadata", new=lookup),
        patch.object(
            subsession_wake,
            "get_redis_async",
            new=AsyncMock(return_value=_FakeRedis()),
        ),
        patch.object(subsession_wake, "is_feature_enabled", new=flag),
        patch(
            "backend.copilot.sdk.session_waiter.run_copilot_turn_via_queue",
            new=enqueue,
        ),
    ):
        await _wake_parent(_SUB, "completed")

    enqueue.assert_not_awaited()
    assert flag.await_args.args[0].value == "copilot-subsession-wake"
    assert flag.await_args.args[1] == _USER


@pytest.mark.asyncio
async def test_missing_parent_session_is_a_noop():
    """Deleted parent (hard delete leaves no row) — nothing to wake."""
    lookup = _lookup({_SUB: _session_info(delegated_by=_PARENT)})
    enqueue = AsyncMock()
    p1, p2, p3, p4 = _patched(lookup=lookup, redis=_FakeRedis(), enqueue=enqueue)

    with p1, p2, p3, p4:
        await _wake_parent(_SUB, "completed")

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_sub_session_is_a_noop():
    lookup = _lookup()
    enqueue = AsyncMock()
    p1, p2, p3, p4 = _patched(lookup=lookup, redis=_FakeRedis(), enqueue=enqueue)

    with p1, p2, p3, p4:
        await _wake_parent(_SUB, "completed")

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_inline_waiter_suppresses_the_wake():
    """The spawning turn is still blocked in ``wait_for_session_result`` for
    this sub, so it will surface the result through its own tool call —
    waking as well would report the same outcome twice."""
    lookup = _lookup(
        {
            _SUB: _session_info(delegated_by=_PARENT),
            _PARENT: _session_info(),
        }
    )
    enqueue = AsyncMock()
    redis = _FakeRedis({subsession_wake.inline_wait_key(_SUB): "1"})
    p1, p2, p3, p4 = _patched(lookup=lookup, redis=redis, enqueue=enqueue)

    with p1, p2, p3, p4:
        await _wake_parent(_SUB, "completed")

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleared_inline_lease_no_longer_suppresses():
    redis = _FakeRedis()
    with patch.object(
        subsession_wake, "get_redis_async", new=AsyncMock(return_value=redis)
    ):
        await subsession_wake.mark_awaited_inline(_SUB, 30)
        assert await subsession_wake._is_awaited_inline(_SUB) is True
        await subsession_wake.clear_awaited_inline(_SUB)
        assert await subsession_wake._is_awaited_inline(_SUB) is False


@pytest.mark.asyncio
async def test_enqueue_failure_is_swallowed():
    """The sub-session's own completion must never break because the wake
    could not be enqueued."""
    lookup = _lookup(
        {
            _SUB: _session_info(delegated_by=_PARENT),
            _PARENT: _session_info(),
        }
    )
    enqueue = AsyncMock(side_effect=RuntimeError("rabbit down"))
    p1, p2, p3, p4 = _patched(lookup=lookup, redis=_FakeRedis(), enqueue=enqueue)

    with p1, p2, p3, p4:
        await _wake_parent(_SUB, "completed")

    enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_lookup_failure_is_swallowed():
    lookup = AsyncMock(side_effect=RuntimeError("db down"))
    enqueue = AsyncMock()
    p1, p2, p3, p4 = _patched(lookup=lookup, redis=_FakeRedis(), enqueue=enqueue)

    with p1, p2, p3, p4:
        await _wake_parent(_SUB, "completed")

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_schedule_parent_wake_runs_detached():
    wake = AsyncMock()
    with patch.object(subsession_wake, "_wake_parent", new=wake):
        schedule_parent_wake(_SUB, "completed")
        await asyncio.sleep(0)

    wake.assert_awaited_once_with(_SUB, "completed")


class _CompletionRedis:
    """Just enough of the Redis surface for ``mark_session_completed``."""

    def __init__(self, meta: dict[str, str]):
        self._meta = dict(meta)
        self.delete = AsyncMock(return_value=1)

    async def hgetall(self, _key: str):
        return dict(self._meta)


@pytest.mark.asyncio
async def test_mark_session_completed_schedules_the_wake_once_per_swap():
    """The wake hangs off the branch that actually swapped the status, so a
    second (idempotent) completion call cannot schedule a second wake."""
    redis = _CompletionRedis({"status": "running", "turn_id": "turn-1"})
    swapped = AsyncMock(side_effect=[True, False])
    schedule = MagicMock()

    with (
        patch.object(
            stream_registry, "get_redis_async", new=AsyncMock(return_value=redis)
        ),
        patch.object(stream_registry, "hash_compare_and_set", new=swapped),
        patch.object(stream_registry, "publish_chunk", new=AsyncMock()),
        patch.object(stream_registry, "schedule_parent_wake", new=schedule),
        patch.object(
            stream_registry.chat_db(),
            "set_turn_duration",
            new=AsyncMock(),
            create=True,
        ),
    ):
        assert await stream_registry.mark_session_completed("sess-1") is True
        assert await stream_registry.mark_session_completed("sess-1") is False

    schedule.assert_called_once_with("sess-1", "completed")
