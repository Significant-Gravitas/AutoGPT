"""compact_transcript: the preserved final turn is trimmed, not abandoned.

The last assistant turn is kept verbatim so its thinking-block signatures
survive.  When that turn is what exceeds the budget, the compaction used to
return None and the caller threw the whole transcript away.  Its text,
tool_use inputs and tool_result strings can be shortened; the thinking
blocks cannot.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.transcript import (
    _tail_token_count,
    _truncate_tail_lines,
    compact_transcript,
)
from backend.util import json
from backend.util.prompt import CompressResult

MODEL = "gpt-4o"
_FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do "


def _text(tokens: int) -> str:
    return (_FILLER * (tokens // 12 + 1))[: tokens * 4]


def _jsonl(*entries: dict) -> str:
    return "\n".join(json.dumps(e) for e in entries) + "\n"


def _user(uuid: str, parent: str, text: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": "user", "content": text},
    }


def _assistant(uuid: str, parent: str, blocks: list[dict], msg_id: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {
            "role": "assistant",
            "id": msg_id,
            "model": "anthropic/claude-4.7-opus-20260416",
            "content": blocks,
        },
    }


THINKING = {"type": "thinking", "thinking": "reasoning", "signature": "REAL_SIG"}


def _history() -> list[dict]:
    entries = [_user("u0", "", "start")]
    parent = "u0"
    for i in range(10):
        entries.append(_user(f"u{i + 1}", parent, f"q{i} " + _text(60)))
        entries.append(
            _assistant(
                f"a{i + 1}",
                f"u{i + 1}",
                [{"type": "text", "text": f"a{i} " + _text(60)}],
                f"msg_{i}",
            )
        )
        parent = f"a{i + 1}"
    entries.append(_user("u99", parent, "make it long"))
    return entries


def _big_final(parent: str) -> dict:
    return _assistant(
        "a99",
        parent,
        [
            THINKING,
            {"type": "text", "text": "BIGFINAL " + _text(20_000)},
            {
                "type": "tool_use",
                "id": "toolu_big",
                "name": "mcp__copilot__create_agent",
                "input": {
                    "nodes": [{"id": f"n{i}", "block": _text(200)} for i in range(40)]
                },
            },
        ],
        "msg_final",
    )


def _compacted_prefix() -> CompressResult:
    return CompressResult(
        messages=[{"role": "user", "content": "summary of the prefix"}],
        token_count=100,
        was_compacted=True,
        original_token_count=5_000,
    )


@pytest.mark.asyncio
async def test_final_turn_over_budget_is_trimmed_not_abandoned():
    entries = _history()
    entries.append(_big_final("u99"))
    content = _jsonl(*entries)

    with patch(
        "backend.copilot.transcript._run_compression",
        new_callable=AsyncMock,
        return_value=_compacted_prefix(),
    ):
        result = await compact_transcript(content, model=MODEL, target_tokens=8_000)

    assert result is not None
    lines = [line for line in result.strip().split("\n") if line.strip()]
    last = json.loads(lines[-1])
    blocks = last["message"]["content"]
    thinking = next(b for b in blocks if b["type"] == "thinking")
    assert thinking == THINKING, "signed thinking must be value-identical"
    text = next(b for b in blocks if b["type"] == "text")
    assert len(text["text"]) < len("BIGFINAL " + _text(20_000))
    assert text["text"].startswith("BIGFINAL")
    tool = next(b for b in blocks if b["type"] == "tool_use")
    assert isinstance(tool["input"], dict) and "_truncated" in tool["input"]
    assert _tail_token_count([lines[-1]], MODEL) <= 8_000


@pytest.mark.asyncio
async def test_prefix_that_already_fits_keeps_its_lines_verbatim():
    """When only the final turn is over budget the compressor reports the
    prefix as already fitting; that used to be treated as failure."""
    entries = _history()
    entries.append(_big_final("u99"))
    content = _jsonl(*entries)

    with patch(
        "backend.copilot.transcript._run_compression",
        new_callable=AsyncMock,
        return_value=CompressResult(
            messages=[], token_count=0, was_compacted=False, original_token_count=800
        ),
    ):
        result = await compact_transcript(content, model=MODEL, target_tokens=8_000)

    assert result is not None
    lines = [line for line in result.strip().split("\n") if line.strip()]
    assert json.loads(lines[0]) == entries[0]  # prefix untouched, uuids intact
    assert json.loads(lines[-1])["message"]["id"] == "msg_final"
    assert json.loads(lines[-1])["parentUuid"] == "u99"


@pytest.mark.asyncio
async def test_everything_fits_and_compressor_declined_still_signals_failure():
    """The 'within budget but the SDK rejected it' contract is unchanged."""
    entries = _history()
    entries.append(
        _assistant("a99", "u99", [{"type": "text", "text": "short"}], "msg_final")
    )
    content = _jsonl(*entries)

    with patch(
        "backend.copilot.transcript._run_compression",
        new_callable=AsyncMock,
        return_value=CompressResult(
            messages=[], token_count=0, was_compacted=False, original_token_count=800
        ),
    ):
        result = await compact_transcript(content, model=MODEL, target_tokens=8_000)

    assert result is None


def test_truncate_tail_lines_leaves_a_thinking_only_turn_alone():
    entry = _assistant("a1", "u1", [THINKING], "msg_x")
    assert _truncate_tail_lines([json.dumps(entry)], 10, MODEL) is None


def test_truncate_tail_lines_shares_the_budget_across_blocks():
    entry = _assistant(
        "a1",
        "u1",
        [
            THINKING,
            {"type": "text", "text": "T " + _text(4_000)},
            {"type": "tool_result", "tool_use_id": "x", "content": "R " + _text(4_000)},
        ],
        "msg_x",
    )
    trimmed = _truncate_tail_lines([json.dumps(entry)], 2_000, MODEL)
    assert trimmed is not None
    blocks = json.loads(trimmed[0])["message"]["content"]
    assert blocks[0] == THINKING
    assert blocks[1]["text"].startswith("T ") and len(blocks[1]["text"]) < len(
        "T " + _text(4_000)
    )
    assert blocks[2]["content"].startswith("R ") and len(blocks[2]["content"]) < len(
        "R " + _text(4_000)
    )
    assert _tail_token_count(trimmed, MODEL) <= 2_400  # thinking + two shares
