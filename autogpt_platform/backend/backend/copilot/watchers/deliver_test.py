"""Unit tests for proactive-watcher delivery — no DB or Redis required."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import TriggerSource

from backend.copilot.watchers.deliver import (
    _DAILY_WATCHER_CAP,
    deliver_expert_paused,
    deliver_review_waiting,
    deliver_run_failed,
)
from backend.copilot.watchers.events import WATCHER_METADATA_KIND, WatcherEvent

_MODULE = "backend.copilot.watchers.deliver"


def _redis(count: int = 1) -> MagicMock:
    client = MagicMock()
    client.incr = AsyncMock(return_value=count)
    client.expire = AsyncMock()
    client.decr = AsyncMock()
    return client


def _library(name: str = "Morning Brief", agent_id: str = "lib-1") -> MagicMock:
    refs = [SimpleNamespace(id=agent_id, graph_id="graph-1", name=name)]
    db = MagicMock()
    db.get_library_agent_refs_by_graph_ids = AsyncMock(return_value=refs)
    return db


def _chat_db(
    session_id: str | None = "sess-expert",
    posted: str | None = "sess-expert",
) -> MagicMock:
    chat_db = MagicMock()
    chat_db.get_expert_post_session_id = AsyncMock(return_value=session_id)
    chat_db.append_expert_run_message = AsyncMock(return_value=posted)
    return chat_db


def _wire(
    stack: ExitStack,
    *,
    enabled: bool = True,
    redis: MagicMock | None = None,
    chat_db: MagicMock | None = None,
    library: MagicMock | None = None,
) -> MagicMock:
    chat_db = chat_db or _chat_db()
    stack.enter_context(
        patch(f"{_MODULE}.is_feature_enabled", new=AsyncMock(return_value=enabled))
    )
    stack.enter_context(
        patch(
            f"{_MODULE}.get_redis_async",
            new=AsyncMock(return_value=redis or _redis()),
        )
    )
    stack.enter_context(patch(f"{_MODULE}.chat_db", new=chat_db))
    stack.enter_context(
        patch(f"{_MODULE}.library_db", return_value=library or _library())
    )
    return chat_db


async def _run_failed(**overrides) -> bool:
    kwargs = {
        "user_id": "user-1",
        "expert_id": "expert-1",
        "graph_exec_id": "exec-1",
        "graph_id": "graph-1",
        "trigger_source": TriggerSource.cron,
        "error": "missing Gmail credentials",
    }
    kwargs.update(overrides)
    return await deliver_run_failed(**kwargs)


@pytest.mark.asyncio
async def test_run_failed_posts_exactly_one_card_to_the_experts_thread():
    with ExitStack() as stack:
        chat_db = _wire(stack)
        owned = await _run_failed()

    assert owned is True
    chat_db.get_expert_post_session_id.assert_awaited_once_with("user-1", "expert-1")
    chat_db.append_expert_run_message.assert_awaited_once()
    kwargs = chat_db.append_expert_run_message.call_args.kwargs
    assert kwargs["session_id"] == "sess-expert"
    assert kwargs["expert_id"] == "expert-1"
    assert "Morning Brief" in kwargs["content"]
    assert "while running on its schedule" in kwargs["content"]
    assert kwargs["metadata"]["kind"] == WATCHER_METADATA_KIND
    assert kwargs["metadata"]["event"] == WatcherEvent.RUN_FAILED.value
    assert kwargs["metadata"]["execution_id"] == "exec-1"
    assert kwargs["metadata"]["trigger_source"] == TriggerSource.cron


@pytest.mark.asyncio
async def test_run_failed_stays_silent_when_the_flag_is_off():
    with ExitStack() as stack:
        chat_db = _wire(stack, enabled=False)
        owned = await _run_failed()

    assert owned is False
    chat_db.append_expert_run_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_failed_still_owns_the_event_when_it_chooses_not_to_post():
    """A deduped or capped watcher must not hand the event back — the legacy
    completion post would then tell the user about the same failure."""
    with ExitStack() as stack:
        _wire(stack, chat_db=_chat_db(posted=None))
        assert await _run_failed() is True


@pytest.mark.asyncio
async def test_replayed_event_derives_the_same_message_id():
    """Dedupe is the ChatMessage primary key, so a replay only has to produce
    the same deterministic id — no bookkeeping to get out of sync."""
    with ExitStack() as stack:
        chat_db = _wire(stack)
        await _run_failed()
        first = chat_db.append_expert_run_message.call_args.kwargs["message_id"]
        await _run_failed()
        second = chat_db.append_expert_run_message.call_args.kwargs["message_id"]

    assert first == second


@pytest.mark.asyncio
async def test_deduped_post_gives_its_rate_cap_slot_back():
    """Otherwise a run that keeps re-firing would silently eat the day's
    budget of unprompted messages without ever posting one."""
    redis = _redis()
    with ExitStack() as stack:
        _wire(stack, redis=redis, chat_db=_chat_db(posted=None))
        await _run_failed()

    redis.decr.assert_awaited_once()
    assert redis.incr.call_args.args[0] == redis.decr.call_args.args[0]


@pytest.mark.asyncio
async def test_rate_cap_stops_further_cards_for_the_day():
    with ExitStack() as stack:
        chat_db = _wire(stack, redis=_redis(count=_DAILY_WATCHER_CAP + 1))
        owned = await _run_failed()

    assert owned is True
    chat_db.append_expert_run_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_rate_cap_is_scoped_per_user_per_day():
    redis = _redis()
    with ExitStack() as stack:
        _wire(stack, redis=redis)
        await _run_failed()

    key = redis.incr.call_args.args[0]
    assert key.startswith("copilot-watchers:user-1:")


@pytest.mark.asyncio
async def test_delivery_failure_never_propagates_to_the_caller():
    """The caller is an executor completion hook — it must not lose a run
    because a chat message didn't land."""
    chat_db = _chat_db()
    chat_db.append_expert_run_message = AsyncMock(side_effect=RuntimeError("db down"))
    redis = _redis()
    with ExitStack() as stack:
        _wire(stack, redis=redis, chat_db=chat_db)
        owned = await _run_failed()

    assert owned is True
    redis.decr.assert_awaited_once()


@pytest.mark.asyncio
async def test_flag_lookup_failure_fails_closed():
    """These are unprompted messages: an unreachable flag service means
    silence, not a surprise post."""
    with ExitStack() as stack:
        chat_db = _chat_db()
        stack.enter_context(
            patch(
                f"{_MODULE}.is_feature_enabled",
                new=AsyncMock(side_effect=RuntimeError("LD down")),
            )
        )
        stack.enter_context(patch(f"{_MODULE}.chat_db", new=chat_db))
        owned = await _run_failed()

    assert owned is False
    chat_db.append_expert_run_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_lookup_failure_degrades_instead_of_dropping_the_card():
    library = MagicMock()
    library.get_library_agent_refs_by_graph_ids = AsyncMock(
        side_effect=RuntimeError("rpc down")
    )
    with ExitStack() as stack:
        chat_db = _wire(stack, library=library)
        assert await _run_failed() is True

    content = chat_db.append_expert_run_message.call_args.kwargs["content"]
    assert "failed while running on its schedule" in content


@pytest.mark.asyncio
async def test_expert_paused_posts_the_budget_card():
    with ExitStack() as stack:
        chat_db = _wire(stack)
        owned = await deliver_expert_paused(
            user_id="user-1", expert_id="expert-1", spent=500, budget=500
        )

    assert owned is True
    kwargs = chat_db.append_expert_run_message.call_args.kwargs
    assert "500 of my 500" in kwargs["content"]
    assert kwargs["metadata"]["event"] == WatcherEvent.EXPERT_PAUSED.value
    assert kwargs["metadata"]["budget"] == 500


@pytest.mark.asyncio
async def test_expert_paused_leaves_the_event_alone_when_the_flag_is_off():
    with ExitStack() as stack:
        chat_db = _wire(stack, enabled=False)
        owned = await deliver_expert_paused(
            user_id="user-1", expert_id="expert-1", spent=500, budget=500
        )

    assert owned is False
    chat_db.append_expert_run_message.assert_not_awaited()


def _execution(expert_id: str | None = "expert-1") -> SimpleNamespace:
    return SimpleNamespace(
        id="exec-1",
        userId="user-1",
        expertId=expert_id,
        agentGraphId="graph-1",
        triggerSource=TriggerSource.webhook,
    )


def _patch_execution(stack: ExitStack, execution) -> None:
    stack.enter_context(
        patch(
            f"{_MODULE}.AgentGraphExecution",
            prisma=MagicMock(
                return_value=MagicMock(find_first=AsyncMock(return_value=execution))
            ),
        )
    )


@pytest.mark.asyncio
async def test_review_waiting_posts_with_the_runs_provenance():
    with ExitStack() as stack:
        chat_db = _wire(stack)
        _patch_execution(stack, _execution())
        await deliver_review_waiting(
            user_id="user-1",
            graph_exec_id="exec-1",
            node_exec_id="node-1",
            instructions="Send $4,000 to Acme?",
        )

    kwargs = chat_db.append_expert_run_message.call_args.kwargs
    assert kwargs["expert_id"] == "expert-1"
    assert "from one of your triggers" in kwargs["content"]
    assert "> Send $4,000 to Acme?" in kwargs["content"]
    assert kwargs["metadata"]["event"] == WatcherEvent.REVIEW_WAITING.value
    assert kwargs["metadata"]["node_exec_id"] == "node-1"


@pytest.mark.asyncio
async def test_review_waiting_skips_runs_with_no_expert_to_speak_for_them():
    with ExitStack() as stack:
        chat_db = _wire(stack)
        _patch_execution(stack, _execution(expert_id=None))
        await deliver_review_waiting(
            user_id="user-1", graph_exec_id="exec-1", node_exec_id="node-1"
        )

    chat_db.append_expert_run_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_waiting_skips_an_execution_it_cannot_read():
    with ExitStack() as stack:
        chat_db = _wire(stack)
        _patch_execution(stack, None)
        await deliver_review_waiting(
            user_id="user-1", graph_exec_id="exec-1", node_exec_id="node-1"
        )

    chat_db.append_expert_run_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_waiting_dedupes_on_the_node_execution():
    with ExitStack() as stack:
        chat_db = _wire(stack)
        _patch_execution(stack, _execution())
        await deliver_review_waiting(
            user_id="user-1", graph_exec_id="exec-1", node_exec_id="node-1"
        )
        first = chat_db.append_expert_run_message.call_args.kwargs["message_id"]
        await deliver_review_waiting(
            user_id="user-1", graph_exec_id="exec-1", node_exec_id="node-1"
        )
        second = chat_db.append_expert_run_message.call_args.kwargs["message_id"]

    assert first == second
