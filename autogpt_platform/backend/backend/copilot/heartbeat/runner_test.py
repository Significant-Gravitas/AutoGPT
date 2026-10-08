"""A beat end to end with a stubbed engine: every skip, suppression, dedupe
and an alert that reaches the user."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.heartbeat import runner, state
from backend.copilot.heartbeat.config import HeartbeatConfig
from backend.copilot.heartbeat.delivery import DeliveryReport
from backend.copilot.heartbeat.runner import HeartbeatTurn, TurnResult, run_heartbeat

USER = "user-1"
# 10:00 in Europe/Berlin, inside the default 08:00-22:00 window.
NOON = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
_ALERT = "Your invoice sync agent failed three times this morning."


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.data[key] = value

    async def set(self, key: str, value: str, nx: bool = False, ex: int = 0):
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        return sum(1 for key in keys if self.data.pop(key, None) is not None)


class StubEngine:
    def __init__(self, result: TurnResult) -> None:
        self.result = result
        self.turns: list[HeartbeatTurn] = []

    async def __call__(self, turn: HeartbeatTurn) -> TurnResult:
        self.turns.append(turn)
        return self.result


def _config(**overrides) -> HeartbeatConfig:
    base = {
        "enabled": True,
        "checklist": "- Tell me if any agent run failed",
        "timezone": "Europe/Berlin",
    }
    return HeartbeatConfig.model_validate({**base, **overrides})


def _alert_call(text: str = _ALERT) -> dict:
    return {
        "tool_name": "run_capability",
        "input": {
            "id": "tool:heartbeat_respond",
            "input": {"notify": True, "notification_text": text},
        },
    }


@pytest.fixture
def redis():
    store = FakeRedis()
    with patch.object(state, "get_redis_async", AsyncMock(return_value=store)):
        yield store


@pytest.fixture
def env(redis):
    """A user with the heartbeat on, no running turn and something new."""
    deliver = AsyncMock(return_value=DeliveryReport(thread_session_id="main"))
    executions = AsyncMock()
    executions.get_graph_executions.return_value = [SimpleNamespace(id="run")]
    chats = AsyncMock()
    chats.get_user_chat_sessions.return_value = []
    with (
        patch.object(runner, "load_config", AsyncMock(return_value=_config())),
        patch.object(runner, "count_running_turns", AsyncMock(return_value=0)),
        patch.object(runner, "execution_db", return_value=executions),
        patch.object(runner, "chat_db", return_value=chats),
        patch.object(
            runner,
            "resolve_default_chat_route",
            AsyncMock(return_value=("platform", None)),
        ),
        patch.object(runner, "_open_session", AsyncMock(return_value="hb-session")),
        patch.object(runner, "deliver_alert", deliver),
    ):
        yield SimpleNamespace(
            redis=redis, deliver=deliver, executions=executions, chats=chats
        )


async def test_a_switched_off_heartbeat_makes_no_model_call(env):
    engine = StubEngine(TurnResult(outcome="completed", response_text="NO_REPLY"))
    with patch.object(
        runner, "load_config", AsyncMock(return_value=_config(enabled=False))
    ):
        result = await run_heartbeat(USER, engine=engine, now=NOON)
    assert (result.status, result.reason) == ("skipped", "disabled")
    assert engine.turns == []


@pytest.mark.parametrize("force", [False, True])
async def test_an_empty_checklist_skips_even_a_manual_run(env, force):
    engine = StubEngine(TurnResult(outcome="completed"))
    empty = _config(checklist="# Heartbeat\n<!-- add checks -->\n- ")
    with patch.object(runner, "load_config", AsyncMock(return_value=empty)):
        result = await run_heartbeat(USER, engine=engine, now=NOON, force=force)
    assert (result.status, result.reason) == ("skipped", "empty_checklist")
    assert engine.turns == []


async def test_outside_active_hours_nothing_runs(env):
    engine = StubEngine(TurnResult(outcome="completed"))
    # 23:30 in Berlin.
    late = datetime(2026, 10, 8, 21, 30, tzinfo=UTC)
    result = await run_heartbeat(USER, engine=engine, now=late)
    assert (result.status, result.reason) == ("skipped", "outside_active_hours")
    assert engine.turns == []


async def test_a_manual_run_ignores_the_window(env):
    engine = StubEngine(TurnResult(outcome="completed", response_text="NO_REPLY"))
    late = datetime(2026, 10, 8, 21, 30, tzinfo=UTC)
    result = await run_heartbeat(USER, engine=engine, now=late, force=True)
    assert result.status == "silent"
    assert len(engine.turns) == 1


async def test_a_running_turn_skips_the_beat(env):
    engine = StubEngine(TurnResult(outcome="completed"))
    with patch.object(runner, "count_running_turns", AsyncMock(return_value=1)):
        result = await run_heartbeat(USER, engine=engine, now=NOON)
    assert (result.status, result.reason) == ("skipped", "turn_running")
    assert engine.turns == []


async def test_nothing_new_since_the_last_beat_skips_the_model(env):
    engine = StubEngine(TurnResult(outcome="completed"))
    await state.set_last_run(USER, NOON - timedelta(minutes=30))
    env.executions.get_graph_executions.return_value = []
    env.chats.get_user_chat_sessions.return_value = [
        SimpleNamespace(started_at=NOON - timedelta(days=2))
    ]
    result = await run_heartbeat(USER, engine=engine, now=NOON)
    assert (result.status, result.reason) == ("skipped", "no_changes")
    assert engine.turns == []
    since = env.executions.get_graph_executions.await_args.kwargs["created_time_gte"]
    assert since == NOON - timedelta(minutes=30)


async def test_a_new_chat_since_the_last_beat_is_a_change(env):
    await state.set_last_run(USER, NOON - timedelta(minutes=30))
    env.executions.get_graph_executions.return_value = []
    env.chats.get_user_chat_sessions.return_value = [
        SimpleNamespace(started_at=NOON - timedelta(minutes=5))
    ]
    assert await runner.changed_since_last_run(USER)


async def test_the_first_beat_always_runs(env):
    env.executions.get_graph_executions.return_value = []
    assert await runner.changed_since_last_run(USER)
    env.executions.get_graph_executions.assert_not_awaited()


async def test_no_reply_is_suppressed(env):
    engine = StubEngine(TurnResult(outcome="completed", response_text="NO_REPLY"))
    result = await run_heartbeat(USER, engine=engine, now=NOON)
    assert (result.status, result.reason) == ("silent", "silent")
    env.deliver.assert_not_awaited()
    # The beat ran, so the next one compares against it.
    assert await state.get_last_run(USER) == NOON


async def test_the_turn_runs_isolated_on_the_cheap_tier_with_read_tools(env):
    engine = StubEngine(TurnResult(outcome="completed", response_text="NO_REPLY"))
    await run_heartbeat(USER, engine=engine, now=NOON)
    (turn,) = engine.turns
    assert turn.session_id == "hb-session"
    assert turn.model_tier == "standard"
    assert "<heartbeat_checklist>" in turn.prompt
    assert "Tell me if any agent run failed" in turn.prompt
    assert "NO_REPLY" in turn.prompt and "memory_search" in turn.prompt
    allowed = turn.permissions.effective_allowed_tools(frozenset(runner.ALL_TOOL_NAMES))
    assert {"heartbeat_respond", "memory_search", "run_capability"} <= allowed
    assert allowed.isdisjoint(
        {"bash_exec", "post_to_chat_platform", "run_agent", "ask_question"}
    )


async def test_an_explicit_alert_is_delivered_once_then_deduped(env):
    engine = StubEngine(
        TurnResult(outcome="completed", response_text="", tool_calls=[_alert_call()])
    )
    first = await run_heartbeat(USER, engine=engine, now=NOON)
    assert (first.status, first.reason) == ("delivered", "explicit_alert")
    assert first.text == _ALERT
    env.deliver.assert_awaited_once()
    user_id, session_id, text, targets = env.deliver.await_args.args
    assert (user_id, session_id, text) == (USER, "hb-session", _ALERT)
    assert targets == _config().delivery

    # The same alert, reworded only in case and spacing, within a day.
    engine.result = TurnResult(
        outcome="completed",
        tool_calls=[
            _alert_call("  your invoice SYNC agent failed three times this morning ")
        ],
    )
    second = await run_heartbeat(USER, engine=engine, now=NOON, force=True)
    assert (second.status, second.reason) == ("silent", "duplicate")
    env.deliver.assert_awaited_once()

    engine.result = TurnResult(
        outcome="completed", tool_calls=[_alert_call("A review is waiting for you.")]
    )
    third = await run_heartbeat(USER, engine=engine, now=NOON, force=True)
    assert third.status == "delivered"


async def test_the_answer_recorded_by_the_tool_outranks_the_stream(env):
    await state.record_response("hb-session", True, _ALERT)
    engine = StubEngine(TurnResult(outcome="completed", response_text="NO_REPLY"))
    result = await run_heartbeat(USER, engine=engine, now=NOON)
    assert result.status == "delivered"
    assert result.text == _ALERT


async def test_a_short_unprompted_reply_is_dropped(env):
    engine = StubEngine(
        TurnResult(outcome="completed", response_text="All quiet, nothing failed.")
    )
    result = await run_heartbeat(USER, engine=engine, now=NOON)
    assert (result.status, result.reason) == ("silent", "short_reply")
    env.deliver.assert_not_awaited()


async def test_a_failed_turn_delivers_nothing(env):
    engine = StubEngine(TurnResult(outcome="failed"))
    result = await run_heartbeat(USER, engine=engine, now=NOON)
    assert (result.status, result.reason) == ("failed", "failed")
    env.deliver.assert_not_awaited()
