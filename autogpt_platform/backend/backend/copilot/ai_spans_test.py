"""Tests for ``copilot.ai_spans``: the PostHog trace of an AutoPilot reply,
its tool calls and its expert subtasks."""

import asyncio
from unittest.mock import MagicMock

import pytest
import pytest_mock

from backend.copilot import ai_spans
from backend.copilot.ai_spans import (
    ToolSpanTimer,
    capture_turn,
    current_span_id,
    mark_tool_failed,
    mark_tool_raised,
    set_turn_span,
    stamp_turn_trace_id,
    tool_span,
    trace_id_for,
)
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.llm_attribution import set_turn_impersonated, set_turn_trace_id
from backend.copilot.model import ChatMessage
from backend.copilot.tree import TurnEnvelope
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

USER_ID = "user-1"
SESSION_ID = "sess-1"
TURN_ID = "turn-1"


@pytest.fixture(autouse=True)
def _outside_a_turn():
    set_turn_trace_id(None)
    set_turn_span(None)
    set_turn_impersonated(False)
    yield
    set_turn_trace_id(None)
    set_turn_span(None)
    set_turn_impersonated(False)


@pytest.fixture
def capture(mocker: pytest_mock.MockerFixture) -> MagicMock:
    return mocker.patch.object(ai_spans.posthog_client, "capture")


def _entry(**overrides) -> CoPilotExecutionEntry:
    fields = {
        "session_id": SESSION_ID,
        "turn_id": TURN_ID,
        "user_id": USER_ID,
        "message": "hi",
        "envelope": TurnEnvelope(tree_id=TURN_ID),
        **overrides,
    }
    return CoPilotExecutionEntry(**fields)


def _in_a_turn(turn_id: str = TURN_ID, trace_id: str = TURN_ID) -> None:
    set_turn_trace_id(trace_id)
    set_turn_span(turn_id)


def _sent(capture: MagicMock) -> list[tuple[PostHogEvent, dict]]:
    return [(call.args[1], call.args[2]) for call in capture.call_args_list]


# -- trace_id_for -------------------------------------------------------------


def test_a_reply_is_its_own_trace():
    assert trace_id_for(_entry(envelope=None)) == TURN_ID
    assert trace_id_for(_entry()) == TURN_ID


def test_a_subtask_joins_the_trace_of_the_turn_that_spawned_it():
    entry = _entry(
        turn_id="child-turn",
        envelope=TurnEnvelope(tree_id="root-turn", depth=1, ai_trace_id="root-turn"),
    )

    assert trace_id_for(entry) == "root-turn"


def test_a_subtask_of_a_woken_reply_joins_that_reply_not_its_tree_root():
    """An approval wake is a reply of its own (B) under the tree it was held
    in (A), so its subtasks belong to B's trace."""
    entry = _entry(
        turn_id="child-turn",
        envelope=TurnEnvelope(
            tree_id="held-turn-a", depth=1, ai_trace_id="woken-turn-b"
        ),
    )

    assert trace_id_for(entry) == "woken-turn-b"


def test_a_woken_subtask_stays_in_the_trace_of_the_turn_that_held_the_call():
    """Reply A is woken as reply B, B delegates to C, and C holds a call. The
    wake C' starts under C's envelope, so it joins B's trace, not A's."""
    held_by_c = TurnEnvelope(
        tree_id="reply-a",
        depth=1,
        ai_trace_id="reply-b",
        ai_parent_span_id="b-delegate-span",
    )

    assert trace_id_for(_entry(turn_id="woken-c", envelope=held_by_c)) == "reply-b"


def test_a_subtask_from_before_the_trace_was_recorded_falls_back_to_its_tree_root():
    """An envelope stored before it kept the trace (a queued child, a held
    call), or one rebuilt from its tree id alone."""
    entry = _entry(
        turn_id="child-turn", envelope=TurnEnvelope(tree_id="root-turn", depth=1)
    )

    assert trace_id_for(entry) == "root-turn"


def test_a_root_re_admitted_under_an_older_tree_is_still_its_own_trace():
    """An approval wake re-admits the envelope its call was held under, so
    a depth-0 tree id can name an earlier turn."""
    assert trace_id_for(_entry(envelope=TurnEnvelope(tree_id="earlier"))) == TURN_ID


# -- tool_span ----------------------------------------------------------------


def test_a_tool_call_is_a_span_under_its_turn(capture: MagicMock):
    _in_a_turn()

    with tool_span(
        "run_block", user_id=USER_ID, session_id=SESSION_ID, tool_call_id="call-1"
    ):
        pass

    capture.assert_called_once()
    distinct_id, event, properties = capture.call_args.args
    assert (distinct_id, event) == (USER_ID, PostHogEvent.AI_SPAN)
    assert capture.call_args.kwargs == {"source": "chat_copilot"}
    span_id = properties.pop("$ai_span_id")
    assert span_id and span_id != TURN_ID
    assert properties.pop("$ai_latency") >= 0
    assert properties == {
        "$ai_trace_id": TURN_ID,
        "$ai_parent_id": TURN_ID,
        "$ai_span_name": "run_block",
        "$ai_is_error": False,
        "$ai_session_id": SESSION_ID,
        "chat_session_id": SESSION_ID,
        "tool_call_id": "call-1",
        "cancelled": False,
        "impersonated": False,
    }


def test_a_span_never_carries_the_tools_input_or_output(capture: MagicMock):
    """Gmail and Drive content must stay out (Google Limited Use)."""
    _in_a_turn()

    with tool_span("gmail_read", user_id=USER_ID, session_id=SESSION_ID) as outcome:
        outcome.is_error = True

    properties = capture.call_args.args[2]
    assert "$ai_input_state" not in properties
    assert "$ai_output_state" not in properties
    assert "$ai_error" not in properties
    assert properties["$ai_is_error"] is True


def test_a_tool_that_answers_with_an_error_is_marked_failed(capture: MagicMock):
    _in_a_turn()

    with (
        tool_span("run_capability", user_id=USER_ID, session_id=SESSION_ID),
        tool_span("run_block", user_id=USER_ID, session_id=SESSION_ID),
    ):
        mark_tool_failed()

    (_, inner), (_, outer) = _sent(capture)
    assert inner["$ai_is_error"] is True
    assert outer["$ai_is_error"] is False


def test_marking_a_failure_outside_a_tool_does_nothing(capture: MagicMock):
    mark_tool_failed()

    capture.assert_not_called()


def test_a_tool_that_raises_is_an_error_named_by_its_type_only(capture: MagicMock):
    _in_a_turn()

    with (
        pytest.raises(ValueError),
        tool_span("web_fetch", user_id=USER_ID, session_id=SESSION_ID),
    ):
        raise ValueError("the secret contents of an email")

    properties = capture.call_args.args[2]
    assert properties["$ai_is_error"] is True
    assert properties["$ai_error"] == "ValueError"
    assert "secret" not in repr(properties)


def test_a_tool_that_caught_what_it_raised_is_named_by_its_type_only(
    capture: MagicMock,
):
    """``BaseTool`` answers an exception with an error instead of letting it
    reach the span, so it names the exception itself."""
    _in_a_turn()

    with tool_span("web_fetch", user_id=USER_ID, session_id=SESSION_ID):
        try:
            raise ValueError("the secret contents of an email")
        except ValueError as error:
            mark_tool_raised(error)

    properties = capture.call_args.args[2]
    assert properties["$ai_is_error"] is True
    assert properties["$ai_error"] == "ValueError"
    assert "secret" not in repr(properties)


def test_marking_a_raise_outside_a_tool_does_nothing(capture: MagicMock):
    mark_tool_raised(ValueError("x"))

    capture.assert_not_called()


def test_a_cancelled_tool_is_not_a_failure(capture: MagicMock):
    _in_a_turn()

    with (
        pytest.raises(asyncio.CancelledError),
        tool_span("bash_exec", user_id=USER_ID, session_id=SESSION_ID),
    ):
        raise asyncio.CancelledError()

    properties = capture.call_args.args[2]
    assert properties["$ai_is_error"] is False
    assert properties["cancelled"] is True


def test_a_tool_called_by_a_tool_nests_under_it(capture: MagicMock):
    _in_a_turn()

    with tool_span("run_capability", user_id=USER_ID, session_id=SESSION_ID):
        outer = current_span_id()
        with tool_span("run_block", user_id=USER_ID, session_id=SESSION_ID):
            pass

    (_, inner), (_, outer_props) = _sent(capture)
    assert inner["$ai_parent_id"] == outer == outer_props["$ai_span_id"]
    assert outer_props["$ai_parent_id"] == TURN_ID
    assert current_span_id() == TURN_ID


def test_a_subtasks_tool_calls_nest_under_the_subtask(capture: MagicMock):
    _in_a_turn(turn_id="child-turn", trace_id="root-turn")

    with tool_span("run_block", user_id=USER_ID, session_id=SESSION_ID):
        pass

    properties = capture.call_args.args[2]
    assert properties["$ai_trace_id"] == "root-turn"
    assert properties["$ai_parent_id"] == "child-turn"


def test_an_impersonated_turns_spans_are_marked(capture: MagicMock):
    _in_a_turn()
    set_turn_impersonated(True)

    with tool_span("run_block", user_id=USER_ID, session_id=SESSION_ID):
        pass

    assert capture.call_args.args[2]["impersonated"] is True


def test_outside_a_turn_no_span_is_sent(capture: MagicMock):
    with tool_span("run_block", user_id=USER_ID, session_id=SESSION_ID):
        pass

    capture.assert_not_called()


@pytest.mark.asyncio
async def test_spans_of_concurrent_tool_calls_do_not_cross(capture: MagicMock):
    _in_a_turn()

    async def call(name: str) -> str | None:
        with tool_span(name, user_id=USER_ID, session_id=SESSION_ID):
            await asyncio.sleep(0)
            return current_span_id()

    first, second = await asyncio.gather(
        asyncio.create_task(call("a")), asyncio.create_task(call("b"))
    )

    by_name = {props["$ai_span_name"]: props for _, props in _sent(capture)}
    assert by_name["a"]["$ai_span_id"] == first
    assert by_name["b"]["$ai_span_id"] == second
    assert by_name["a"]["$ai_parent_id"] == by_name["b"]["$ai_parent_id"] == TURN_ID


def test_a_posthog_failure_never_reaches_the_tool(mocker: pytest_mock.MockerFixture):
    """The real ``capture`` runs against a client that raises."""
    client = MagicMock()
    client.capture.side_effect = RuntimeError("posthog is down")
    mocker.patch.object(posthog_client, "get_posthog_client", return_value=client)
    _in_a_turn()

    with tool_span("run_block", user_id=USER_ID, session_id=SESSION_ID):
        result = "ran"

    assert result == "ran"
    client.capture.assert_called_once()


def test_a_broken_property_never_reaches_the_tool(capture: MagicMock):
    capture.side_effect = RuntimeError("unexpected")
    _in_a_turn()

    with tool_span("run_block", user_id=USER_ID, session_id=SESSION_ID):
        pass


# -- ToolSpanTimer (SDK built-in tools) ---------------------------------------


def test_a_builtin_tool_is_timed_from_its_hooks(capture: MagicMock):
    _in_a_turn()
    timer = ToolSpanTimer()

    timer.start("tu-1", "WebSearch")
    timer.finish(
        "tu-1", "WebSearch", user_id=USER_ID, session_id=SESSION_ID, is_error=False
    )

    properties = capture.call_args.args[2]
    assert properties["$ai_span_name"] == "WebSearch"
    assert properties["$ai_trace_id"] == TURN_ID
    assert properties["$ai_parent_id"] == TURN_ID
    assert properties["$ai_is_error"] is False
    assert properties["builtin"] is True
    assert properties["tool_call_id"] == "tu-1"


def test_a_failed_builtin_tool_is_an_error(capture: MagicMock):
    _in_a_turn()
    timer = ToolSpanTimer()

    timer.start("tu-1", "Bash")
    timer.finish("tu-1", "Bash", user_id=USER_ID, session_id=SESSION_ID, is_error=True)

    assert capture.call_args.args[2]["$ai_is_error"] is True


def test_a_builtin_tool_that_never_started_sends_nothing(capture: MagicMock):
    _in_a_turn()
    timer = ToolSpanTimer()

    timer.finish("tu-1", "Bash", user_id=USER_ID, session_id=SESSION_ID, is_error=True)

    capture.assert_not_called()


def test_a_builtin_tool_outside_a_turn_sends_nothing(capture: MagicMock):
    timer = ToolSpanTimer()

    timer.start("tu-1", "Bash")
    timer.finish("tu-1", "Bash", user_id=USER_ID, session_id=SESSION_ID, is_error=False)

    capture.assert_not_called()


@pytest.mark.parametrize("tool_name", ["TodoWrite", "Read", "Glob", "Grep", "LS"])
def test_a_high_volume_builtin_tool_sends_no_span(capture: MagicMock, tool_name: str):
    """The CLI calls these tens of times a turn, sub-agents included, and a
    span of a file read or a checklist update says little."""
    _in_a_turn()
    timer = ToolSpanTimer()

    timer.start("tu-1", tool_name)
    timer.finish(
        "tu-1", tool_name, user_id=USER_ID, session_id=SESSION_ID, is_error=False
    )

    capture.assert_not_called()


@pytest.mark.parametrize("tool_name", ["Task", "Agent", "Bash", "Edit", "Write"])
def test_a_builtin_tool_that_acts_keeps_its_span(capture: MagicMock, tool_name: str):
    _in_a_turn()
    timer = ToolSpanTimer()

    timer.start("tu-1", tool_name)
    timer.finish(
        "tu-1", tool_name, user_id=USER_ID, session_id=SESSION_ID, is_error=False
    )

    assert capture.call_args.args[2]["$ai_span_name"] == tool_name


def test_builtin_tool_spans_can_be_turned_off(capture: MagicMock):
    _in_a_turn()
    timer = ToolSpanTimer(enabled=False)

    timer.start("tu-1", "Task")
    timer.finish("tu-1", "Task", user_id=USER_ID, session_id=SESSION_ID, is_error=False)

    capture.assert_not_called()


# -- capture_turn -------------------------------------------------------------


def _end(entry: CoPilotExecutionEntry, **overrides) -> None:
    kwargs = {
        "latency_seconds": 12.5,
        "outcome": "completed",
        "error_code": None,
        "engine": "sdk",
        "origin": "chat",
        "tool_call_count": 3,
        **overrides,
    }
    capture_turn(entry, **kwargs)


def test_a_reply_ends_with_its_trace_root(capture: MagicMock):
    _end(_entry())

    capture.assert_called_once()
    distinct_id, event, properties = capture.call_args.args
    assert (distinct_id, event) == (USER_ID, PostHogEvent.AI_TRACE)
    assert properties == {
        "$ai_trace_id": TURN_ID,
        "$ai_session_id": SESSION_ID,
        "$ai_span_name": "copilot-turn",
        "$ai_latency": 12.5,
        "$ai_is_error": False,
        "chat_session_id": SESSION_ID,
        "turn_id": TURN_ID,
        "outcome": "completed",
        "engine": "sdk",
        "turn_origin": "chat",
        "tool_call_count": 3,
        "impersonated": False,
    }


def test_a_failed_reply_names_its_error_code(capture: MagicMock):
    _end(_entry(), outcome="failed", error_code="baseline_error")

    properties = capture.call_args.args[2]
    assert properties["$ai_is_error"] is True
    assert properties["$ai_error"] == "baseline_error"


def test_a_cancelled_reply_is_not_an_error(capture: MagicMock):
    _end(_entry(), outcome="cancelled")

    assert capture.call_args.args[2]["$ai_is_error"] is False


def test_an_impersonated_reply_is_marked(capture: MagicMock):
    _end(_entry(impersonated=True))

    assert capture.call_args.args[2]["impersonated"] is True


def test_a_subtask_is_a_span_under_the_tool_call_that_started_it(capture: MagicMock):
    entry = _entry(
        turn_id="child-turn",
        session_id="expert-sess",
        envelope=TurnEnvelope(
            tree_id="root-turn",
            depth=1,
            ai_trace_id="root-turn",
            ai_parent_span_id="spawn-tool-span",
        ),
    )

    _end(entry)

    _, event, properties = capture.call_args.args
    assert event == PostHogEvent.AI_SPAN
    assert properties["$ai_trace_id"] == "root-turn"
    assert properties["$ai_span_id"] == "child-turn"
    assert properties["$ai_parent_id"] == "spawn-tool-span"
    assert properties["$ai_span_name"] == "copilot-subtask"
    assert properties["$ai_session_id"] == "expert-sess"
    assert properties["depth"] == 1


def test_a_subtask_of_unknown_origin_hangs_off_the_trace_root(capture: MagicMock):
    """An envelope stored before it kept the spawning tool's span."""
    entry = _entry(
        turn_id="child-turn", envelope=TurnEnvelope(tree_id="root-turn", depth=1)
    )

    _end(entry)

    assert capture.call_args.args[2]["$ai_parent_id"] == "root-turn"


def test_a_turn_end_failure_is_swallowed(capture: MagicMock):
    capture.side_effect = RuntimeError("unexpected")

    _end(_entry())


def _redelivered_twice(
    entry: CoPilotExecutionEntry, mocker: pytest_mock.MockerFixture
) -> list[dict]:
    """End the same turn twice, as a redelivered turn would, and return what
    reached the PostHog client."""
    client = MagicMock()
    mocker.patch.object(posthog_client, "get_posthog_client", return_value=client)
    _end(entry)
    _end(entry)
    return [call.kwargs for call in client.capture.call_args_list]


def test_a_redelivered_reply_sends_one_trace_root(
    mocker: pytest_mock.MockerFixture,
):
    first, second = _redelivered_twice(_entry(), mocker)

    assert first["event"] == PostHogEvent.AI_TRACE.value
    assert first["uuid"] is not None
    assert first["uuid"] == second["uuid"]
    assert first["properties"]["$insert_id"] == TURN_ID


def test_a_redelivered_subtask_sends_one_span(mocker: pytest_mock.MockerFixture):
    entry = _entry(
        turn_id="child-turn",
        envelope=TurnEnvelope(tree_id="root-turn", depth=1, ai_trace_id="root-turn"),
    )

    first, second = _redelivered_twice(entry, mocker)

    assert first["event"] == PostHogEvent.AI_SPAN.value
    assert first["uuid"] is not None
    assert first["uuid"] == second["uuid"]


def test_different_turns_are_not_deduplicated(mocker: pytest_mock.MockerFixture):
    client = MagicMock()
    mocker.patch.object(posthog_client, "get_posthog_client", return_value=client)

    _end(_entry(turn_id="turn-a"))
    _end(_entry(turn_id="turn-b"))

    first, second = (call.kwargs for call in client.capture.call_args_list)
    assert first["uuid"] != second["uuid"]


# -- stamp_turn_trace_id ------------------------------------------------------


def _messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="assistant", content="old", sequence=0),
        ChatMessage(role="user", content="q", sequence=1),
        ChatMessage(role="assistant", content="working", sequence=2),
        ChatMessage(role="tool", content="{}", tool_call_id="t1"),
        ChatMessage(role="assistant", content="done"),
        ChatMessage(role="reasoning", content="checking the totals"),
    ]


def test_stamps_the_turns_replies_with_its_trace():
    set_turn_trace_id(TURN_ID)
    messages = _messages()

    stamp_turn_trace_id(messages, start_index=1)

    assert [m.trace_id for m in messages] == [
        None,  # an earlier turn's reply
        None,  # user row
        TURN_ID,
        None,  # tool row
        TURN_ID,
        TURN_ID,  # a reply bubble can be named after its reasoning row
    ]
    assert messages[2].stamps_pending_save is True  # flushed mid-turn
    assert messages[4].stamps_pending_save is False  # inserted by this save


def test_never_overwrites_a_recorded_trace():
    set_turn_trace_id(TURN_ID)
    messages = _messages()
    messages[2].trace_id = "earlier-turn"

    stamp_turn_trace_id(messages, start_index=0)

    assert messages[2].trace_id == "earlier-turn"
    assert messages[2].stamps_pending_save is False


def test_outside_a_turn_stamps_nothing():
    messages = _messages()

    stamp_turn_trace_id(messages, start_index=0)

    assert all(m.trace_id is None for m in messages)
    assert not any(m.stamps_pending_save for m in messages)
