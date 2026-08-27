from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.watchers.deliver import _DAILY_WATCHER_CAP, deliver_run_failed
from backend.copilot.watchers.events import WATCHER_METADATA_KIND, WatcherEvent

_MODULE = "backend.copilot.watchers.deliver"


def _redis(count: int = 1):
    pipe = MagicMock()
    pipe.incr = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock(return_value=[count, True])
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=None)
    redis = MagicMock()
    redis.pipeline.return_value = pipe
    redis.decr = AsyncMock()
    return redis


def _wire(monkeypatch, *, enabled=True, count=1, posted="session-1"):
    chat = MagicMock()
    chat.append_expert_run_message = AsyncMock(return_value=posted)
    library = MagicMock()
    library.get_library_agent_refs_by_graph_ids = AsyncMock(
        return_value=[SimpleNamespace(id="library-1", name="Lead Research")]
    )
    monkeypatch.setattr(
        f"{_MODULE}.is_feature_enabled", AsyncMock(return_value=enabled)
    )
    monkeypatch.setattr(
        f"{_MODULE}.get_redis_async", AsyncMock(return_value=_redis(count))
    )
    monkeypatch.setattr(f"{_MODULE}.chat_db", chat)
    monkeypatch.setattr(f"{_MODULE}.library_db", lambda: library)
    return chat


async def _deliver():
    return await deliver_run_failed(
        user_id="user-1",
        expert_id="expert-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        trigger_source="cron",
        error="Missing credentials",
    )


@pytest.mark.asyncio
async def test_flag_off_does_not_post(monkeypatch):
    chat = _wire(monkeypatch, enabled=False)

    assert await _deliver() is False
    chat.append_expert_run_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_failure_posts_one_semantic_card_with_exact_link(monkeypatch):
    chat = _wire(monkeypatch)

    assert await _deliver() is True
    kwargs = chat.append_expert_run_message.await_args.kwargs
    assert kwargs["metadata"] == {
        "kind": WATCHER_METADATA_KIND,
        "event": WatcherEvent.RUN_FAILED.value,
        "title": "Lead Research needs attention",
        "description": "Workflow run failed",
        "action_label": "Open run",
        "action_href": "/library/agents/library-1?activeTab=runs&activeItem=exec-1",
        "status": "failed",
    }
    assert "exec-1" not in kwargs["content"]
    assert "graph-1" not in kwargs["content"]


@pytest.mark.asyncio
async def test_replay_uses_same_message_id(monkeypatch):
    chat = _wire(monkeypatch)

    await _deliver()
    first = chat.append_expert_run_message.await_args.kwargs["message_id"]
    await _deliver()
    second = chat.append_expert_run_message.await_args.kwargs["message_id"]

    assert first == second


@pytest.mark.asyncio
async def test_rate_cap_posts_one_semantic_overflow_card(monkeypatch):
    chat = _wire(monkeypatch, count=_DAILY_WATCHER_CAP + 1)

    assert await _deliver() is True
    kwargs = chat.append_expert_run_message.await_args.kwargs
    assert kwargs["metadata"]["event"] == WatcherEvent.OVERFLOW.value
    assert kwargs["metadata"]["action_href"] == "/home"
    assert "exec-1" not in kwargs["content"]
