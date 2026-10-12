"""compress_context: the behaviours adversarial probes found missing.

Each test here corresponds to a defect measured against the previous
implementation (see the PR): a count-based tail, a summariser fed only the
first 100K characters, a fixed 8K cap that over-truncated, a deletion
cascade around a large last message, untouchable tool-call arguments,
uncounted image blocks, and a trimmed system prompt.
"""

import json

import pytest
from tiktoken import encoding_for_model

from backend.util.prompt import SUMMARY_CHUNK_CHARS, _msg_tokens, compress_context

MODEL = "gpt-4o"  # tokenizer factor 1.0, so counts below are real tokens
_FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do "


def _text(tokens: int) -> str:
    return (_FILLER * (tokens // 12 + 1))[: tokens * 4]


def _u(text: str) -> dict:
    return {"role": "user", "content": text}


def _a(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _pair(i: int, out_tokens: int, arg_tokens: int = 20) -> list[dict]:
    call_id = f"call_{i}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "edit_workspace_file",
                        "arguments": json.dumps(
                            {"path": f"/w/f{i}", "pad": _text(arg_tokens)}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"[f{i}] " + _text(out_tokens),
        },
    ]


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class FakeSummariser:
    """Stands in for the OpenAI client: records what it was shown."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen = ""

    def with_options(self, **_):
        return self

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    async def create(self, model, messages, max_tokens, temperature):
        self.calls += 1
        self.seen += messages[-1]["content"]
        return _Response("SUMMARY " + _text(300))


def _out(result) -> str:
    return json.dumps(result.messages)


@pytest.mark.asyncio
async def test_tail_is_sized_in_tokens_and_kept_verbatim():
    msgs = [_u("start")]
    for i in range(30):
        msgs += [_u(f"q{i} " + _text(400)), _a(f"a{i} " + _text(400))]
    client = FakeSummariser()

    result = await compress_context(
        msgs,
        target_tokens=12_000,
        model=MODEL,
        client=client,
        keep_recent_tokens=4_000,
        reserve=0,
    )

    out = _out(result)
    assert all(m["content"] in out for m in msgs[-4:])
    assert result.messages_summarized > 0
    assert result.summary_coverage == 1.0
    assert result.tail_tokens >= 4_000
    assert result.token_count <= 12_000
    assert result.error is None


@pytest.mark.asyncio
async def test_summariser_is_shown_the_whole_old_history():
    """Older history beyond one chunk is chunked and merged, not cut off."""
    msgs = [_u(f"m{i} NDL{i:04d} " + _text(250)) for i in range(260)] + [_u("end")]
    assert sum(len(m["content"]) for m in msgs) > 2 * SUMMARY_CHUNK_CHARS
    client = FakeSummariser()

    result = await compress_context(
        msgs,
        target_tokens=20_000,
        model=MODEL,
        client=client,
        keep_recent_tokens=2_000,
        reserve=0,
    )

    assert client.calls >= 3  # at least two chunks plus a merge
    # Every message either reached the summariser or survives verbatim in
    # the tail — nothing falls in the gap the old 100K-character cap left.
    out = _out(result)
    assert all(f"NDL{i:04d}" in client.seen or f"NDL{i:04d}" in out for i in range(260))
    assert result.summary_coverage == 1.0


@pytest.mark.asyncio
async def test_huge_recent_response_is_truncated_to_fit_not_summarised_away():
    msgs = [_u("start")]
    for i in range(10):
        msgs += [_u(_text(100)), _a(_text(100))]
    msgs += [_a("BIG " + _text(30_000)), _u("next?")]

    result = await compress_context(
        msgs,
        target_tokens=20_000,
        model=MODEL,
        client=FakeSummariser(),
        keep_recent_tokens=2_000,
        reserve=0,
    )

    big = next(
        m for m in result.messages if str(m.get("content", "")).startswith("BIG")
    )
    kept = len(encoding_for_model(MODEL).encode(big["content"]))
    assert kept >= 10_000, "more than half the budget, not a fixed 8K cap"
    assert 16_000 <= result.token_count <= 20_000
    assert result.error is None


@pytest.mark.asyncio
async def test_huge_last_message_does_not_delete_the_rest():
    msgs = [_u("start")]
    for i in range(10):
        msgs += [_u(f"q{i} " + _text(100)), _a(f"a{i} " + _text(100))]
    msgs += [_a("BIGLAST " + _text(30_000))]

    result = await compress_context(
        msgs,
        target_tokens=20_000,
        model=MODEL,
        client=FakeSummariser(),
        keep_recent_tokens=2_000,
        reserve=0,
    )

    assert result.messages_dropped == 0
    assert result.token_count <= 20_000
    assert result.error is None


@pytest.mark.asyncio
async def test_tool_arguments_are_cut_only_when_the_caller_allows():
    msgs = [_u("start")] + _pair(1, 50, arg_tokens=30_000) + [_u("now")]
    original = msgs[1]["tool_calls"][0]["function"]["arguments"]

    kept = await compress_context(
        msgs, target_tokens=10_000, model=MODEL, client=None, reserve=0
    )
    assert kept.messages[1]["tool_calls"][0]["function"]["arguments"] == original
    assert kept.error is not None  # nothing else was there to cut

    cut = await compress_context(
        msgs,
        target_tokens=10_000,
        model=MODEL,
        client=None,
        reserve=0,
        truncate_tool_arguments=True,
    )
    assert len(cut.messages[1]["tool_calls"][0]["function"]["arguments"]) < len(
        original
    )
    assert cut.token_count <= 10_000
    assert cut.error is None


def test_image_blocks_are_counted():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "A" * 400_000},
            },
        ],
    }
    assert _msg_tokens(message, encoding_for_model(MODEL)) > 50_000


@pytest.mark.asyncio
async def test_system_prompt_is_never_truncated():
    system_text = "SYS " + _text(3_000)
    msgs = [{"role": "system", "content": system_text}]
    for i in range(20):
        msgs += [_u(_text(500)), _a(_text(500))]

    result = await compress_context(
        msgs, target_tokens=8_000, model=MODEL, client=None, reserve=0
    )

    assert result.messages[0]["content"] == system_text
    assert result.token_count <= 8_000


@pytest.mark.asyncio
async def test_without_a_summariser_old_history_gives_way_before_the_tail():
    msgs = [_u(f"old{i} " + _text(500)) for i in range(40)]
    msgs += [_u("RECENT1 " + _text(500)), _a("RECENT2 " + _text(500))]

    result = await compress_context(
        msgs,
        target_tokens=6_000,
        model=MODEL,
        client=None,
        keep_recent_tokens=2_000,
        reserve=0,
    )

    out = _out(result)
    assert msgs[-1]["content"] in out and msgs[-2]["content"] in out
    assert result.summarizer_available is False
    assert result.token_count <= 6_000


@pytest.mark.asyncio
async def test_tool_pairs_stay_intact_across_the_tail_boundary():
    msgs = [_u("start")]
    for i in range(12):
        msgs += _pair(i, 800)
    msgs += [_u("now")]

    result = await compress_context(
        msgs,
        target_tokens=6_000,
        model=MODEL,
        client=FakeSummariser(),
        keep_recent_tokens=2_500,
        reserve=0,
    )

    call_ids = {tc["id"] for m in result.messages for tc in m.get("tool_calls") or []}
    for m in result.messages:
        if m.get("role") == "tool":
            assert m["tool_call_id"] in call_ids
    assert result.token_count <= 6_000
