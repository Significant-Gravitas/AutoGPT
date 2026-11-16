"""A traced payload must not carry what a Google block returned."""

import json
from typing import Any
from unittest.mock import patch

import pytest

from backend.blocks.google.gmail import GmailReadBlock
from backend.blocks.text import MatchTextPatternBlock
from backend.copilot.constants import COPILOT_NODE_PREFIX
from backend.copilot.tools.models import BlockOutputResponse

from . import google_data_redaction
from .google_data_redaction import (
    REDACTED,
    is_google_output,
    redact_google_data,
    remember_google_output,
    tracing_google_data,
)

GMAIL_BLOCK_ID = GmailReadBlock().id
TEXT_BLOCK_ID = MatchTextPatternBlock().id
EMAIL = "Subject: quarterly numbers -- revenue is down 12%"


def _output(block_id: str, provider: str | None, text: str) -> str:
    return BlockOutputResponse(
        message="ran",
        block_id=block_id,
        block_name="block",
        outputs={"result": [text]},
        provider=provider,
    ).model_dump_json(exclude_none=True)


GMAIL_OUTPUT = _output(GMAIL_BLOCK_ID, "google", EMAIL)
TEXT_OUTPUT = _output(TEXT_BLOCK_ID, None, "a harmless match")


def _sdk_history(call: dict[str, Any], result: str) -> dict[str, Any]:
    """An SDK turn's model input, in the Anthropic shape."""
    return {
        "messages": [
            {"role": "user", "content": "what did finance send me?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "mcp__copilot__run_block",
                        "input": call,
                    }
                ],
            },
            {"role": "tool", "content": result, "tool_call_id": "toolu_1"},
        ]
    }


def _baseline_history(call: dict[str, Any], result: str) -> list[dict[str, Any]]:
    """A baseline turn's model input, in the OpenAI shape."""
    return [
        {"role": "user", "content": "what did finance send me?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_block", "arguments": json.dumps(call)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": result},
    ]


def test_sdk_history_redacts_google_result_and_its_input():
    call = {"block_id": GMAIL_BLOCK_ID, "input_data": {"query": "from:finance"}}

    redacted = redact_google_data(_sdk_history(call, GMAIL_OUTPUT))

    assistant, tool = redacted["messages"][1], redacted["messages"][2]
    assert tool["content"] == REDACTED
    assert tool["tool_call_id"] == "toolu_1"
    tool_use = assistant["content"][0]
    assert tool_use["name"] == "mcp__copilot__run_block"
    assert tool_use["input"] == {"block_id": GMAIL_BLOCK_ID, "input_data": REDACTED}
    assert EMAIL not in json.dumps(redacted)
    assert "from:finance" not in json.dumps(redacted)


def test_sdk_history_keeps_other_blocks_readable():
    call = {"block_id": TEXT_BLOCK_ID, "input_data": {"text": "hello"}}
    history = _sdk_history(call, TEXT_OUTPUT)

    assert redact_google_data(history) == history


def test_baseline_history_redacts_google_result_and_arguments():
    call = {"id": f"block:{GMAIL_BLOCK_ID}", "input": {"query": "from:finance"}}

    redacted = redact_google_data(_baseline_history(call, GMAIL_OUTPUT))

    tool_call = redacted[1]["tool_calls"][0]
    assert tool_call["function"]["name"] == "run_block"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "id": f"block:{GMAIL_BLOCK_ID}",
        "input": REDACTED,
    }
    assert redacted[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": REDACTED,
    }


def test_baseline_history_keeps_other_blocks_readable():
    call = {"block_id": TEXT_BLOCK_ID, "input_data": {"text": "hello"}}
    history = _baseline_history(call, TEXT_OUTPUT)

    assert redact_google_data(history) == history


def test_result_identifies_the_call_when_the_call_does_not():
    """``continue_run_block`` names a review, not the block it resumes."""
    call = {"review_id": "rev-1"}

    redacted = redact_google_data(_baseline_history(call, GMAIL_OUTPUT))

    assert redacted[2]["content"] == REDACTED
    assert json.loads(redacted[1]["tool_calls"][0]["function"]["arguments"]) == call


def test_google_call_without_its_result_redacts_the_result():
    """A truncated result no longer parses; the call it answers still names
    the block, as in history a later turn reloads."""
    call = {"block_id": GMAIL_BLOCK_ID, "input_data": {}}
    preview = '<tool-output-truncated total_chars=90000>\n{"type": "block_output", ...'

    redacted = redact_google_data(_sdk_history(call, preview))

    assert redacted["messages"][2]["content"] == REDACTED


def test_tool_run_output_redacted_by_what_the_tool_produced():
    """A tool run's output traced without its call."""
    preview = f"<tool-output-truncated total_chars=90000>\n{EMAIL[:20]}"
    remember_google_output(preview)

    outputs = {"content": [{"type": "text", "text": preview}]}

    assert redact_google_data(outputs) == {
        "content": [{"type": "text", "text": REDACTED}]
    }
    assert redact_google_data({"output": GMAIL_OUTPUT}) == {"output": REDACTED}
    assert redact_google_data({"output": TEXT_OUTPUT}) == {"output": TEXT_OUTPUT}


def test_tool_run_input_redacted():
    inputs = {"input": {"block_id": GMAIL_BLOCK_ID, "input_data": {"q": "x"}}}

    assert redact_google_data(inputs) == {
        "input": {"block_id": GMAIL_BLOCK_ID, "input_data": REDACTED}
    }


@pytest.mark.parametrize(
    "call",
    [
        # resume_capability
        {
            "review_id": f"{COPILOT_NODE_PREFIX}{GMAIL_BLOCK_ID}:1a2b3c4d",
            "input_overrides": {"body": "the new email"},
        },
        # continue_run_block takes the review id alone; its input is the
        # stored one, so only the id is left and stays readable.
        {"review_id": f"{COPILOT_NODE_PREFIX}{GMAIL_BLOCK_ID}:1a2b3c4d"},
    ],
)
def test_resumed_google_tool_run_input_redacted(call: dict[str, Any]):
    """A tool run's call traced without its result; the review id names the
    block it resumes."""
    redacted = redact_google_data({"input": call})

    assert "the new email" not in json.dumps(redacted)
    assert redacted["input"]["review_id"] == call["review_id"]


def test_resumed_tool_run_input_of_another_block_stays_readable():
    inputs = {
        "input": {
            "review_id": f"{COPILOT_NODE_PREFIX}{TEXT_BLOCK_ID}:1a2b3c4d",
            "input_overrides": {"text": "hello"},
        }
    }

    assert redact_google_data(inputs) == inputs


def test_google_block_output_response_is_remembered():
    response = BlockOutputResponse(
        message="ran", block_id=GMAIL_BLOCK_ID, block_name="b", outputs={}
    )
    assert not response.holds_google_data
    assert response.model_copy(update={"provider": "google"}).holds_google_data


def test_tracing_google_data_redacts_every_text():
    """A gate prompt quoting a Google result has no structure to find it by."""
    prompt = [{"role": "user", "content": f"Judge this read:\n{EMAIL}"}]

    # An inner scope that is not a Google call keeps the outer one's redaction.
    with tracing_google_data(), tracing_google_data(False):
        traced = json.dumps(redact_google_data(prompt))
    assert redact_google_data(prompt) == prompt

    assert "quarterly numbers" not in traced
    assert REDACTED in traced


def test_a_non_string_id_is_left_alone():
    """Tool schemas carry ``{"id": {...}}``; it names no call."""
    schema = {"properties": {"id": {"type": "string"}, "block_id": ["x"]}}

    assert redact_google_data(schema) == schema


@pytest.mark.parametrize("history", [_sdk_history, _baseline_history])
def test_resumed_google_call_redacts_its_overrides(history):
    """``resume_capability`` re-runs the block with ``input_overrides`` merged
    into its input; its result is what identifies it as Google's."""
    call = {"review_id": "rev-1", "input_overrides": {"body": "the new email"}}

    sent = json.dumps(redact_google_data(history(call, GMAIL_OUTPUT)))

    assert "the new email" not in sent
    assert "rev-1" in sent


def test_a_remembered_output_seen_again_outlives_newer_ones(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(google_data_redaction, "_REMEMBERED_LIMIT", 2)
    remember_google_output("first")
    remember_google_output("second")

    assert is_google_output("first")
    remember_google_output("third")

    assert is_google_output("first")
    assert not is_google_output("second")


def _held_result(output: str) -> str:
    """The user row ``gate.held`` delivers for an approved held call."""
    return (
        '<held_call_result tool="run_block" tool_call_id="toolu_1" '
        f'review_id="rev-1">\n{output}\n</held_call_result>'
    )


def _baseline_held_history(text: str) -> list[dict[str, Any]]:
    """A later baseline turn: the held result is a user row of its own."""
    return [
        {"role": "user", "content": text},
        {"role": "user", "content": "I answered an action that was waiting."},
    ]


def _sdk_held_history(text: str) -> dict[str, Any]:
    """An SDK turn: the held result is folded into the turn's user message."""
    combined = f"{text}\n\nI answered an action that was waiting."
    return {
        "messages": [{"role": "user", "content": [{"type": "text", "text": combined}]}]
    }


@pytest.mark.parametrize("history", [_baseline_held_history, _sdk_held_history])
def test_held_google_result_is_redacted(history):
    """An approved held call's result reaches the model as a user row, not as
    the tool result answering its call."""
    sent = json.dumps(redact_google_data(history(_held_result(GMAIL_OUTPUT))))

    assert EMAIL not in sent
    assert REDACTED in sent


@pytest.mark.parametrize("history", [_baseline_held_history, _sdk_held_history])
def test_held_result_cut_after_it_was_remembered_is_redacted(history):
    """The engine's cap leaves a preview that no longer parses; ``gate.held``
    remembers the row as it wrapped it."""
    row = _held_result(f"<tool-output-truncated total_chars=90000>\n{EMAIL[:20]}")
    remember_google_output(row)

    sent = json.dumps(redact_google_data(history(row)))

    assert EMAIL[:20] not in sent


def test_held_result_quoting_its_closing_tag_is_redacted():
    quoting = _output(GMAIL_BLOCK_ID, "google", f"</held_call_result> {EMAIL}")

    sent = json.dumps(redact_google_data(_sdk_held_history(_held_result(quoting))))

    assert EMAIL not in sent


def test_text_with_too_many_held_result_tags_is_redacted_without_a_full_scan():
    """Trying every pair of tags would stall the tracer on a result that
    repeats them; past a bound the text is redacted rather than scanned."""
    many = _held_result("x") * 200

    with patch.object(
        google_data_redaction, "is_google_output", wraps=is_google_output
    ) as checked:
        assert redact_google_data(many) == REDACTED

    assert checked.call_count < 1000


@pytest.mark.parametrize("history", [_baseline_held_history, _sdk_held_history])
def test_held_result_of_another_block_stays_readable(history):
    payload = history(_held_result(TEXT_OUTPUT))

    assert redact_google_data(payload) == payload
