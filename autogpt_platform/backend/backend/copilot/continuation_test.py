from typing import Any

import pytest

from backend.copilot.continuation import (
    KEEP_GOING_NOTE,
    RESUMED_AT_RESET_NOTE,
    send_trailing_note_as_user,
)

_TASK = {"role": "user", "content": "Build the report"}
_TOOL_CALL = {
    "id": "call-1",
    "type": "function",
    "function": {"name": "run_agent", "arguments": "{}"},
}


@pytest.mark.parametrize(
    ("note", "instruction"),
    [
        (KEEP_GOING_NOTE, "Keep going: pick the task up from the checkpoint above."),
        (
            RESUMED_AT_RESET_NOTE,
            "Resumed after the budget reset: pick the task up from the checkpoint "
            "above.",
        ),
    ],
)
def test_a_trailing_note_is_sent_as_the_user_turn(note: str, instruction: str):
    messages = [_TASK, {"role": "assistant", "content": note}]

    send_trailing_note_as_user(messages)

    assert messages == [_TASK, {"role": "user", "content": instruction}]


@pytest.mark.parametrize(
    "last",
    [
        pytest.param({"role": "assistant", "content": "Two done."}, id="plain-text"),
        pytest.param(
            {
                "role": "assistant",
                "content": KEEP_GOING_NOTE,
                "tool_calls": [_TOOL_CALL],
            },
            id="tool-calls",
        ),
        pytest.param({"role": "assistant", "tool_calls": [_TOOL_CALL]}, id="no-text"),
        pytest.param(_TASK, id="user"),
    ],
)
def test_any_other_last_entry_is_left_as_it_is(last: dict[str, Any]):
    messages = [_TASK, {**last}]

    send_trailing_note_as_user(messages)

    assert messages == [_TASK, last]


def test_an_empty_request_is_left_empty():
    messages: list[dict[str, Any]] = []

    send_trailing_note_as_user(messages)

    assert messages == []
