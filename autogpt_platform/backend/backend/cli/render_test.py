import json
from datetime import datetime
from types import SimpleNamespace

import click
import pytest

from backend.cli.chat import _parse_seq_range, _render_message, _render_session


def _msg(**kwargs):
    base = {
        "sequence": 0,
        "role": "user",
        "content": None,
        "name": None,
        "toolCalls": None,
        "durationMs": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_render_user_strips_context_blocks():
    msg = _msg(
        sequence=0,
        role="user",
        content=(
            "<available_skills>lots of stuff</available_skills>\n"
            "<user_context>name=Reinier</user_context>\n"
            "I want to set up my notifier agent"
        ),
    )
    rendered = "\n".join(_render_message(msg, full=False))
    assert "[0] USER" in rendered
    assert "I want to set up my notifier agent" in rendered
    assert "lots of stuff" not in rendered
    assert "name=Reinier" not in rendered
    assert "[available_skills omitted]" in rendered


def test_render_assistant_with_tool_calls():
    msg = _msg(
        sequence=2,
        role="assistant",
        content="Let me find that agent.",
        toolCalls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "find_library_agent",
                    "arguments": json.dumps({"query": "GitHub PR Gmail Notifier"}),
                },
            }
        ],
    )
    rendered = "\n".join(_render_message(msg, full=False))
    assert "Let me find that agent." in rendered
    assert '-> find_library_agent(query="GitHub PR Gmail Notifier")' in rendered


def test_render_tool_summarizes_type():
    msg = _msg(
        sequence=3,
        role="tool",
        content=json.dumps(
            {"type": "no_results", "message": "Nothing found", "name": "no_results"}
        ),
    )
    rendered = "\n".join(_render_message(msg, full=False))
    assert "<no_results>" in rendered
    assert "Nothing found" in rendered


def test_render_reasoning_is_prefixed():
    msg = _msg(sequence=1, role="reasoning", content="Thinking about the request.")
    rendered = "\n".join(_render_message(msg, full=False))
    assert "[1] REASONING" in rendered
    assert "| Thinking about the request." in rendered


def test_truncation_and_full_flag():
    long_text = "x" * 800
    msg = _msg(sequence=0, role="user", content=long_text)

    truncated = "\n".join(_render_message(msg, full=False))
    assert "[truncated]" in truncated
    assert len(truncated) < 800

    full = "\n".join(_render_message(msg, full=True))
    assert "[truncated]" not in full
    assert "x" * 800 in full


def test_parse_seq_range_forms():
    assert _parse_seq_range("7") == (7, 7)
    assert _parse_seq_range("6-10") == (6, 10)
    assert _parse_seq_range("6-") == (6, None)
    assert _parse_seq_range("-10") == (None, 10)
    assert _parse_seq_range(" 3 - 5 ") == (3, 5)


def test_parse_seq_range_invalid_raises():
    with pytest.raises(click.BadParameter):
        _parse_seq_range("abc")
    with pytest.raises(click.BadParameter):
        _parse_seq_range("10-6")  # start after end


def test_render_session_header_and_messages():
    session = SimpleNamespace(
        id="abc-123",
        title="GitHub PR to Gmail Notifier Setup",
        chatStatus="idle",
        userId="user-1",
        createdAt=datetime(2026, 6, 5, 13, 16, 54),
        updatedAt=datetime(2026, 6, 5, 13, 18, 14),
        totalPromptTokens=42,
        totalCompletionTokens=7,
    )
    messages = [
        _msg(sequence=0, role="user", content="Set up my agent"),
        _msg(sequence=1, role="assistant", content="Done."),
    ]
    rendered = _render_session(session, messages, full=False)
    assert "ChatSession abc-123" in rendered
    assert "title:    GitHub PR to Gmail Notifier Setup" in rendered
    assert "tokens:   prompt=42 completion=7" in rendered
    assert "messages: 2" in rendered
    assert "[0] USER" in rendered
    assert "[1] ASSISTANT" in rendered
