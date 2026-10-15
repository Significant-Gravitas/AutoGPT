import json
import os
from pathlib import Path

import pytest

from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path
from backend.copilot.sdk.model_gateway.cli_gateway_test import _gateway
from backend.copilot.sdk.model_gateway.testing.canned_upstream import (
    CannedReply,
    CannedUpstream,
)
from backend.copilot.sdk.model_gateway.testing.cli_session import CliSession
from backend.copilot.sdk.model_gateway.testing.regressions import (
    REFUSAL,
    chat_reply,
    chunk,
    fragments,
    gateway_for,
    refusal_events,
)
from backend.copilot.sdk.model_gateway.testing.script import Script
from backend.copilot.sdk.model_gateway.testing.scripted_upstream import ScriptedUpstream


@pytest.mark.parametrize("adapter", ["openai_chat", "codex_responses"])
@pytest.mark.parametrize("window", [24_000, 272_000])
async def test_real_cli_compacts_fully_cached_history_at_the_real_window(
    adapter, window, cli, tmp_path
):
    async with ScriptedUpstream(Script(fully_cached=True)) as upstream, _gateway(
        adapter, upstream
    ) as gateway:
        session = CliSession(
            cli_path=cli,
            gateway=gateway,
            root=tmp_path,
            window_tokens=window,
            output_tokens=2_000,
            page_chars=24_000,
        )
        (first,) = await session.run(
            [
                "Read handbook pages 1 and 2 with read_doc, one page per call, then tell me each page's code word."
            ]
        )
        (second,) = await session.run(
            ["Without calling any tools: which code words did you find?"],
            resume=first.session_id,
        )
    assert not first.is_error and not second.is_error
    assert (
        (first.compactions >= 1 and bool(upstream.compactions))
        if window == 24_000
        else first.compactions == 0
    )
    assert "amber-7" in second.result and "cobalt-14" in second.result
    usages = [message["usage"] for message in saved_assistants(tmp_path)]
    assert usages and all(usage["input_tokens"] == 0 for usage in usages)
    assert gateway.usage.input_tokens == 0


async def test_codex_prompt_cache_key_stays_the_same_across_real_cli_resume(
    cli, tmp_path
):
    async with ScriptedUpstream(Script()) as upstream, _gateway(
        "codex_responses", upstream
    ) as gateway:
        session = CliSession(
            cli_path=cli,
            gateway=gateway,
            root=tmp_path,
            window_tokens=272_000,
            output_tokens=8_000,
        )
        (first,) = await session.run(["Remember this fact: fact10 is number 70."])
        resume_start = len(upstream.seen)
        (second,) = await session.run(
            ["Without calling any tools, recall fact10."], resume=first.session_id
        )
    assert not first.is_error and not second.is_error
    assert first.session_id == second.session_id
    assert 0 < resume_start < len(upstream.seen)
    assert {seen.body["prompt_cache_key"] for seen in upstream.seen} == {
        first.session_id
    }


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> Path:
    path = _resolve_cli_path()
    assert path is not None, "The bundled or configured Claude Code CLI is required"
    for key in [k for k in os.environ if k.startswith(("CLAUDE", "ANTHROPIC"))]:
        monkeypatch.delenv(key)
    return path


def saved_assistants(root: Path) -> list[dict]:
    records = [
        json.loads(line)
        for path in (root / "claude-config" / "projects").rglob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return [
        record["message"] for record in records if record.get("type") == "assistant"
    ]


@pytest.mark.parametrize("adapter", ["openai_chat", "codex_responses"])
@pytest.mark.parametrize("filtered", [False, True])
async def test_real_cli_displays_refusal_text_once(adapter, filtered, cli, tmp_path):
    reply = (
        chat_reply(
            [
                chunk({"refusal": REFUSAL}),
                chunk({}, "content_filter" if filtered else "stop"),
            ]
        )
        if adapter == "openai_chat"
        else CannedReply.of_sse(refusal_events())
    )
    if filtered and adapter == "codex_responses":
        events = refusal_events()
        events[-1] = {
            "type": "response.incomplete",
            "response": {"incomplete_details": {"reason": "content_filter"}},
        }
        reply = CannedReply.of_sse(events)
    async with CannedUpstream(*[reply] * 12) as upstream, gateway_for(
        adapter, upstream
    ) as gateway:
        session = CliSession(
            cli_path=cli,
            gateway=gateway,
            root=tmp_path,
            window_tokens=272_000,
            output_tokens=8_000,
        )
        (turn,) = await session.run(["Please answer this short question."])
    assert not turn.is_error and turn.result == REFUSAL
    assert len(upstream.calls) == 1
    assert saved_assistants(tmp_path)[-1]["stop_reason"] == "end_turn"


async def test_real_cli_executes_a_tool_with_fragmented_name_and_id(cli, tmp_path):
    answer = chat_reply([chunk({"content": "amber-7"}, "stop")])
    async with CannedUpstream(chat_reply(fragments()), answer) as upstream, gateway_for(
        "openai_chat", upstream
    ) as gateway:
        session = CliSession(
            cli_path=cli,
            gateway=gateway,
            root=tmp_path,
            window_tokens=272_000,
            output_tokens=8_000,
        )
        (turn,) = await session.run(["Read handbook page 1 with read_doc."])
    assert not turn.is_error and turn.tool_uses == ["mcp__handbook__read_doc"]
    results = [m for m in upstream.calls[-1].body["messages"] if m["role"] == "tool"]
    assert results[0]["tool_call_id"] == "call_review"
    assert "HANDBOOK PAGE 1" in results[0]["content"]
    assert "No such tool" not in results[0]["content"]


@pytest.mark.parametrize("adapter", ["openai_chat", "codex_responses"])
async def test_real_cli_saves_a_fully_cached_prompt_with_zero_input(
    adapter, cli, tmp_path
):
    if adapter == "openai_chat":
        reply = chat_reply(
            [
                chunk({"content": "answer"}, "stop"),
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "prompt_tokens_details": {"cached_tokens": 100},
                        "completion_tokens": 2,
                    },
                },
            ]
        )
    else:
        reply = CannedReply.of_sse(
            [
                {
                    "type": "response.output_text.delta",
                    "item_id": "m",
                    "delta": "answer",
                },
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {
                            "input_tokens": 100,
                            "input_tokens_details": {"cached_tokens": 100},
                            "output_tokens": 2,
                        }
                    },
                },
            ]
        )
    async with CannedUpstream(reply) as upstream, gateway_for(
        adapter, upstream
    ) as gateway:
        session = CliSession(
            cli_path=cli,
            gateway=gateway,
            root=tmp_path,
            window_tokens=24_000,
            output_tokens=2_000,
        )
        (turn,) = await session.run(["Please answer this short question."])
    assert not turn.is_error and turn.result == "answer"
    usage = saved_assistants(tmp_path)[-1]["usage"]
    assert usage["input_tokens"] == 0 and usage["cache_read_input_tokens"] == 100
    assert gateway.usage.prompt_tokens == 100
