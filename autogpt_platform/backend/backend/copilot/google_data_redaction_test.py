"""What reaches Langfuse must not carry what a Google block returned."""

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import langsmith
import pytest
from langfuse import Langfuse, get_client
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from backend.blocks.google.gmail import GmailReadBlock
from backend.blocks.text import MatchTextPatternBlock
from backend.copilot.constants import COPILOT_NODE_PREFIX
from backend.copilot.tools.models import BlockOutputResponse

from . import google_data_redaction
from .google_data_redaction import (
    REDACTED,
    install_langfuse_mask,
    install_trace_redaction,
    is_google_output,
    mask_langfuse_data,
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
    """An SDK turn's model input, as LangSmith records it."""
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
    """A baseline turn's model input, as the Langfuse OpenAI wrapper records it."""
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
    """LangSmith records a tool run's output without its call."""
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
    """A LangSmith tool run records the call without its result; the review
    id names the block it resumes."""
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


@pytest.fixture
def langfuse_spans() -> Iterator[InMemorySpanExporter]:
    """A Langfuse client built the way production builds it, exporting to
    memory instead of to Langfuse."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    LangfuseResourceManager.reset()
    Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        tracer_provider=provider,
        mask=mask_langfuse_data,
    )
    try:
        yield exporter
    finally:
        LangfuseResourceManager.reset()


def test_langfuse_generation_input_is_masked(langfuse_spans: InMemorySpanExporter):
    call = {"block_id": GMAIL_BLOCK_ID, "input_data": {"query": "from:finance"}}
    history = _baseline_history(call, GMAIL_OUTPUT)
    client = get_client()

    generation = client.start_observation(
        name="OpenAI-generation", as_type="generation", input=history
    )
    generation.update(output={"role": "assistant", "content": "Done."})
    generation.end()

    sent = json.dumps(
        [dict(span.attributes or {}) for span in langfuse_spans.get_finished_spans()]
    )
    assert EMAIL not in sent
    assert "from:finance" not in sent
    assert REDACTED in sent
    assert "run_block" in sent


def _build_langfuse(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")
    Langfuse(tracer_provider=TracerProvider(), **kwargs)


def test_install_langfuse_mask_refuses_a_client_built_without_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """The SDK hands out the first client built; a mask passed later is
    dropped, so installing it must fail rather than trace unmasked."""
    LangfuseResourceManager.reset()
    try:
        _build_langfuse(monkeypatch)

        with pytest.raises(RuntimeError):
            install_langfuse_mask()
    finally:
        LangfuseResourceManager.reset()


def test_install_langfuse_mask_accepts_a_client_built_with_it(
    monkeypatch: pytest.MonkeyPatch,
):
    LangfuseResourceManager.reset()
    try:
        _build_langfuse(monkeypatch, mask=mask_langfuse_data)

        install_langfuse_mask()
    finally:
        LangfuseResourceManager.reset()


def test_install_langfuse_mask_uses_the_configured_credentials():
    """Credentials read from ``.env`` reach ``Settings`` but not the process
    environment the SDK falls back to; without them the first client, which
    every later ``get_client()`` reuses, would be disabled."""
    secrets = MagicMock(
        langfuse_public_key="pk-from-settings",
        langfuse_secret_key="sk-from-settings",
        langfuse_host="https://langfuse.example.com",
    )
    with (
        patch.object(google_data_redaction, "Settings") as settings,
        patch.object(google_data_redaction, "Langfuse") as langfuse,
    ):
        settings.return_value.secrets = secrets
        langfuse.return_value._resources.mask = mask_langfuse_data
        install_langfuse_mask()

    langfuse.assert_called_once_with(
        public_key="pk-from-settings",
        secret_key="sk-from-settings",
        host="https://langfuse.example.com",
        mask=mask_langfuse_data,
    )


def test_install_trace_redaction_masks_both_tracers():
    with (
        patch.object(google_data_redaction, "Langfuse") as langfuse,
        patch.object(google_data_redaction.langsmith, "configure") as configure,
    ):
        langfuse.return_value._resources.mask = mask_langfuse_data
        install_trace_redaction()

    langfuse.assert_called_once()
    assert langfuse.call_args.kwargs["mask"] is mask_langfuse_data
    client = configure.call_args.kwargs["client"]
    assert isinstance(client, langsmith.Client)
    inputs = {"input": {"block_id": GMAIL_BLOCK_ID, "input_data": {"q": "x"}}}
    assert client._hide_run_inputs(inputs)["input"]["input_data"] == REDACTED
    assert client._hide_run_outputs({"output": GMAIL_OUTPUT}) == {"output": REDACTED}


def test_tracing_google_data_redacts_every_text(langfuse_spans: InMemorySpanExporter):
    """A gate prompt quoting a Google result has no structure to find it by."""
    prompt = [{"role": "user", "content": f"Judge this read:\n{EMAIL}"}]
    client = get_client()

    # An inner scope that is not a Google call keeps the outer one's redaction.
    with tracing_google_data(), tracing_google_data(False):
        generation = client.start_observation(
            name="OpenAI-generation", as_type="generation", input=prompt
        )
        generation.update(output={"role": "assistant", "content": EMAIL})
        generation.end()
    assert redact_google_data(prompt) == prompt

    sent = json.dumps(
        [dict(span.attributes or {}) for span in langfuse_spans.get_finished_spans()]
    )
    assert "quarterly numbers" not in sent
    assert REDACTED in sent


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
