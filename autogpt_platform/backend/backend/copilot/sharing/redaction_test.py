"""The stored shapes the shared-chat sanitizer has to read.

Both engines store a tool call's arguments as a JSON string, and a tool result
too large to keep inline as a preview, or cut in the middle when its full copy
can't be saved; the approval gate stores an approved call's late result in a
user row.  Each shape is built by the code that stores it and passed through
:func:`sanitize_chat_message`, the projection the share endpoint returns.
"""

import json
import re
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.gate.held import HeldCall, _result_row
from backend.copilot.model import ChatMessage
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.sharing.models import sanitize_chat_message
from backend.copilot.tools.base import _persist_and_summarize

HEADER_SECRET = "hdr-canary-5d1e"
KEY_SECRET = "key-canary-08b2"
ACCESS_SECRET = "access-canary-c94a"
REFRESH_SECRET = "refresh-canary-71f0"
SECRETS = (HEADER_SECRET, KEY_SECRET, ACCESS_SECRET, REFRESH_SECRET)

# An HTTP call carrying keys the model copied from an earlier result.
ARGUMENTS = {
    "block_id": "http-request",
    "input_data": {
        "url": "https://api.example.com/v1/items",
        "headers": {
            "Authorization": f"Bearer {HEADER_SECRET}",
            "X-API-Key": KEY_SECRET,
            "Accept": "application/json",
        },
    },
}
REDACTED_ARGUMENTS = {
    "block_id": "http-request",
    "input_data": {
        "url": "https://api.example.com/v1/items",
        "headers": {
            "Authorization": "[redacted]",
            "X-API-Key": "[redacted]",
            "Accept": "application/json",
        },
    },
}


def _message(**fields: Any) -> ChatMessage:
    return ChatMessage(
        id="m1",
        sequence=0,
        created_at=datetime(2026, 11, 9, tzinfo=UTC),
        **fields,
    )


def _call(arguments: str) -> dict[str, Any]:
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": "run_block", "arguments": arguments},
    }


def _assert_hidden(message: ChatMessage) -> None:
    shown = sanitize_chat_message(message).model_dump_json()
    for secret in SECRETS:
        assert secret not in shown


@pytest.mark.parametrize(
    "arguments",
    [
        # copilot/sdk/service.py stores json.dumps of the call's input.
        pytest.param(json.dumps(ARGUMENTS), id="sdk"),
        # The baseline engine stores the provider's own string.
        pytest.param(json.dumps(ARGUMENTS, separators=(",", ":")), id="baseline"),
    ],
)
def test_secrets_in_string_arguments_are_redacted(arguments: str):
    message = _message(role="assistant", tool_calls=[_call(arguments)])

    sanitized = sanitize_chat_message(message)

    assert sanitized.tool_calls is not None
    shown = sanitized.tool_calls[0]["function"]["arguments"]
    assert isinstance(shown, str)
    assert json.loads(shown) == REDACTED_ARGUMENTS
    _assert_hidden(message)


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            _message(
                role="assistant",
                function_call={"name": "run_block", "arguments": json.dumps(ARGUMENTS)},
            ),
            id="legacy function_call",
        ),
        pytest.param(
            _message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "run_block",
                        "arguments": json.dumps(ARGUMENTS),
                    }
                ],
            ),
            id="arguments at the top of the call",
        ),
        pytest.param(
            # get_sub_session_result quotes a sub-session's calls, arguments and all.
            _message(
                role="tool",
                content=json.dumps(
                    {
                        "type": "sub_session_result",
                        "tool_calls": [
                            {"tool_name": "run_block", "input": json.dumps(ARGUMENTS)}
                        ],
                    }
                ),
            ),
            id="arguments quoted in a result",
        ),
    ],
)
def test_secrets_in_json_carried_by_a_string_are_redacted(message: ChatMessage):
    _assert_hidden(message)


def test_arguments_that_do_not_parse_are_withheld():
    stored = json.dumps(ARGUMENTS)
    cut_off = stored[: stored.index(KEY_SECRET) + len(KEY_SECRET)]
    message = _message(role="assistant", tool_calls=[_call(cut_off)])

    sanitized = sanitize_chat_message(message)

    assert sanitized.tool_calls is not None
    assert sanitized.tool_calls[0]["function"]["arguments"] == "[redacted]"


def test_a_repeated_key_cannot_carry_a_value_past_the_redaction():
    # Parsing keeps only the last copy of a repeated key; the stored text
    # still holds the first.
    arguments = (
        '{"headers": {"Authorization": "Bearer ' + HEADER_SECRET + '"}, "headers": {}}'
    )

    _assert_hidden(_message(role="assistant", tool_calls=[_call(arguments)]))


def _large_result(items: int) -> dict[str, Any]:
    """A block's output with a token at each end of a long list."""
    return {
        "type": "block_output",
        "outputs": {
            "response": [
                {
                    "access_token": ACCESS_SECRET,
                    "items": [f"item {n:05d} " + "x" * 90 for n in range(items)],
                    "refresh_token": REFRESH_SECRET,
                }
            ]
        },
    }


async def _stored_preview(output: dict[str, Any], *, digest: bool) -> str:
    """What copilot/tools/base.py stores for a result too large to keep inline."""
    db = AsyncMock()
    db.get_or_create_workspace = AsyncMock(return_value=MagicMock(id="ws-1"))
    with (
        patch("backend.copilot.tools.base.workspace_db", return_value=db),
        patch("backend.copilot.tools.base.WorkspaceManager", return_value=AsyncMock()),
    ):
        return await _persist_and_summarize(
            json.dumps(output), "user-1", "session-1", "call-1", digest
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("digest", [False, True], ids=["middle-out", "outline"])
async def test_the_preview_of_a_truncated_result_is_withheld(digest: bool):
    stored = await _stored_preview(_large_result(1_000), digest=digest)
    assert ACCESS_SECRET in stored and REFRESH_SECRET in stored
    total_chars = re.match(r"<tool-output-truncated total_chars=(\d+)", stored)
    assert total_chars is not None

    sanitized = sanitize_chat_message(
        _message(role="tool", content=stored, tool_call_id="call-1")
    )

    assert sanitized.content == (
        f"<tool-output-truncated total_chars={total_chars[1]}>\n"
        "[redacted]\n"
        "</tool-output-truncated>"
    )


def _cut_to_size(output: str) -> str:
    """What is stored for a result too large to keep whole when its full copy
    can't be saved: the output cut in the middle."""
    stored = StreamToolOutputAvailable(
        toolCallId="call-1", toolName="run_block", output=output
    ).output
    assert isinstance(stored, str)
    return stored


def test_a_result_that_starts_like_json_but_does_not_parse_is_withheld():
    stored = json.dumps(_large_result(3))[:-4]

    sanitized = sanitize_chat_message(
        _message(role="tool", content=stored, tool_call_id="call-1")
    )

    assert ACCESS_SECRET in stored
    assert sanitized.content == "[redacted]"


def test_a_result_cut_to_size_is_withheld():
    stored = _cut_to_size(json.dumps(_large_result(2_000)))
    assert ACCESS_SECRET in stored and REFRESH_SECRET in stored

    sanitized = sanitize_chat_message(
        _message(role="tool", content=stored, tool_call_id="call-1")
    )

    assert sanitized.content == "[redacted]"


def test_a_cut_that_still_parses_is_withheld():
    # The text kept after the cut can start inside a secret whose key was cut
    # away, and then reads as the end of the value the kept start ended in.
    total = 200_000
    probe = _cut_to_size("x" * total)
    kept_after_cut = len(probe) - probe.index(")…") - 2
    before_secret = '", "api_key": "sk-'
    after_secret = KEY_SECRET + '", "more": "' + "b" * 1_000
    after_secret += "b" * (kept_after_cut - len(after_secret) - 2) + '"}'
    head = '{"note": "' + "a" * (total - 10 - len(before_secret) - len(after_secret))
    stored = _cut_to_size(head + before_secret + after_secret)
    assert KEY_SECRET in json.loads(stored)["note"]

    _assert_hidden(_message(role="tool", content=stored, tool_call_id="call-1"))


@pytest.mark.parametrize(
    "output",
    [
        pytest.param(json.dumps({"type": "block_output", **ARGUMENTS}), id="json"),
        pytest.param(json.dumps(_large_result(2_000)), id="cut to size"),
    ],
)
def test_the_output_in_a_held_calls_late_result_is_redacted(output: str):
    # gate/held.py stores it as a user row, which the share shows as written.
    call = HeldCall(
        review_id="copilot-node-gate-run_block:3e0a034f",
        tool_name="run_block",
        tool_call_id="call-1",
        args={},
    )
    late = _result_row(call, _cut_to_size(output), "approved")
    message = _message(role="user", content=late.content, metadata=late.metadata)

    sanitized = sanitize_chat_message(message)

    _assert_hidden(message)
    assert sanitized.content is not None
    assert sanitized.content.startswith(
        '<held_call_result tool="run_block" tool_call_id="call-1" '
    )
    assert sanitized.content.endswith("\n</held_call_result>")


@pytest.mark.parametrize(
    "words",
    [
        pytest.param('{"api_key": "' + KEY_SECRET + '"}', id="pasted json"),
        pytest.param('Use this: {"api_key": "' + KEY_SECRET + '"}', id="text"),
    ],
)
def test_a_users_own_words_are_not_redacted(words: str):
    # The share shows what the user typed or pasted, as written.
    assert sanitize_chat_message(_message(role="user", content=words)).content == words


def _deeply_nested(depth: int) -> str:
    """JSON *depth* objects deep with a secret at the bottom, written as text
    because json.dumps can't nest that far."""
    leaf = json.dumps({"api_key": KEY_SECRET})
    return '{"next": ' * depth + leaf + "}" * depth


@pytest.mark.parametrize("depth", [300, 2_000, 100_000])
@pytest.mark.parametrize("where", ["result", "arguments", "string in a result"])
def test_deeply_nested_json_is_withheld_without_failing(depth: int, where: str):
    nested = _deeply_nested(depth)
    message = {
        "result": _message(role="tool", content=nested),
        "arguments": _message(role="assistant", tool_calls=[_call(nested)]),
        "string in a result": _message(
            role="tool", content=json.dumps({"stdout": nested})
        ),
    }[where]

    _assert_hidden(message)


@pytest.mark.parametrize("where", ["result", "arguments"])
def test_json_holding_a_number_too_long_to_read_is_withheld(where: str):
    # Python refuses to convert an integer of more than 4,300 digits.
    text = '{"api_key": "' + KEY_SECRET + '", "count": ' + "1" * 5_000 + "}"
    message = {
        "result": _message(role="tool", content=text),
        "arguments": _message(role="assistant", tool_calls=[_call(text)]),
    }[where]

    _assert_hidden(message)


def test_large_arguments_are_redacted_rather_than_withheld():
    rows = [f"row {n:05d} " + "y" * 100 for n in range(20_000)]
    arguments = json.dumps({**ARGUMENTS, "rows": rows})
    message = _message(role="assistant", tool_calls=[_call(arguments)])

    sanitized = sanitize_chat_message(message)

    assert sanitized.tool_calls is not None
    shown = json.loads(sanitized.tool_calls[0]["function"]["arguments"])
    assert shown == {**REDACTED_ARGUMENTS, "rows": rows}


ORDINARY_CALLS = [
    {
        "id": "call-1",
        "type": "function",
        "display_name": "[Draft] Weekly report",
        "function": {
            "name": "web_search",
            "arguments": json.dumps({"query": "café opening hours", "limit": 5}),
        },
    },
    {
        "id": "call-2",
        "type": "function",
        "function": {
            "name": "run_block",
            "arguments": json.dumps(
                {"input_data": {"text": "[1/3] {name}", "page": json.dumps([2])}},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    },
    {"id": "call-3", "type": "function", "function": {"name": "ls", "arguments": ""}},
]
ORDINARY_RESULTS = [
    json.dumps(
        {
            "type": "bash_exec",
            "stdout": json.dumps({"ok": True, "name": "José"}, indent=2) + "\n",
            "exit_code": 0,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ),
    json.dumps(
        {
            "type": "block_output",
            "outputs": {"result": ["naïve", "{draft}", "[1/3] queued"]},
            "usage": {"total_tokens": 12},
            "author": "Ada Lovelace",
        }
    ),
    "completed without output",
    "<tool_use_error>File does not exist.</tool_use_error>",
]


def test_ordinary_content_is_returned_exactly_as_stored():
    sanitized = sanitize_chat_message(
        _message(
            role="assistant", content="Looking that up.", tool_calls=ORDINARY_CALLS
        )
    )

    assert sanitized.content == "Looking that up."
    assert sanitized.tool_calls == ORDINARY_CALLS
    for result in ORDINARY_RESULTS:
        shown = sanitize_chat_message(_message(role="tool", content=result))
        assert shown.content == result
