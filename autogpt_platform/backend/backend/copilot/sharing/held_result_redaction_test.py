"""A held call's late result in a shared chat, when its row isn't laid out
exactly as the approval gate writes it.

The gate stores the result as a user row: an opening tag, the tool's output,
a closing tag.  A row known as a late result by its opening tag or its
metadata, but laid out any other way, can't be split into the wrapper and the
output, so its whole body is withheld rather than shown as the user's words.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from backend.copilot.gate.held import _WAKE_KEY, HeldCall, _result_row, _wake_message
from backend.copilot.model import ChatMessage
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sharing.models import sanitize_chat_message

SECRET = "held-canary-4e7a"
WITHHELD = "<held_call_result>\n[redacted]\n</held_call_result>"


def _message(**fields: Any) -> ChatMessage:
    return ChatMessage(
        id="m1", sequence=0, created_at=datetime(2026, 11, 9, tzinfo=UTC), **fields
    )


def _late_result() -> PendingMessage:
    call = HeldCall(
        review_id="copilot-node-gate-run_block:3e0a034f",
        tool_name="run_block",
        tool_call_id="call-1",
        args={},
    )
    output = json.dumps({"type": "block_output", "access_token": SECRET, "n": 2})
    return _result_row(call, output, "approved")


@pytest.mark.parametrize("known_by", ["metadata", "opening tag"])
@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda content: content + "\n", id="trailing newline"),
        pytest.param(lambda content: "\n" + content, id="leading newline"),
        pytest.param(lambda content: content + "\n\nthanks", id="text after it"),
        pytest.param(
            lambda content: content.replace(">\n", " outcome=approved>\n", 1),
            id="unquoted attribute",
        ),
        pytest.param(
            lambda content: content.replace(">\n", "> ", 1),
            id="output on the tag's line",
        ),
    ],
)
def test_a_late_result_laid_out_differently_is_withheld(
    edit: Callable[[str], str], known_by: str
):
    late = _late_result()
    content = edit(late.content)
    assert SECRET in content
    metadata = late.metadata if known_by == "metadata" else None

    shown = sanitize_chat_message(
        _message(role="user", content=content, metadata=metadata)
    )

    assert shown.content == WITHHELD


def test_a_late_result_known_only_by_its_metadata_is_withheld():
    # The wrapper renamed: only the row's metadata still says what it is.
    late = _late_result()
    content = late.content.replace("held_call_result", "approved_call_result")

    shown = sanitize_chat_message(
        _message(role="user", content=content, metadata=late.metadata)
    )

    assert shown.content == WITHHELD


def test_a_late_result_with_a_bracket_in_an_attribute_is_still_redacted():
    # Laid out as the gate writes it: the '>' is inside a quoted value.
    late = _late_result()
    content = late.content.replace('tool="run_block"', 'tool="run>block"')

    shown = sanitize_chat_message(
        _message(role="user", content=content, metadata=late.metadata)
    ).content

    assert shown is not None
    opening, output = shown.split(">\n", 1)
    assert opening == content.split(">\n", 1)[0]
    output = output.removesuffix("\n</held_call_result>")
    assert json.loads(output) == {
        "type": "block_output",
        "access_token": "[redacted]",
        "n": 2,
    }


@pytest.mark.parametrize("waiting", [0, 2])
def test_the_gates_wake_row_is_shown_as_written(waiting: int):
    text = _wake_message(waiting)

    shown = sanitize_chat_message(
        _message(role="user", content=text, metadata={_WAKE_KEY: True})
    )

    assert shown.content == text
