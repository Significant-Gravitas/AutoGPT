"""The real CLI, through the gateway, against each adapter's kind of upstream.

Each adapter is driven by the CLI that ships with ``claude-agent-sdk`` (or the
one ``CHAT_CLAUDE_AGENT_CLI_PATH``/``CLAUDE_AGENT_CLI_PATH`` names) against a
scripted model on loopback: llama.cpp-style ``/v1/messages`` for the
pass-through, Chat Completions for ``openai_chat``, a Responses backend with
encrypted reasoning for Codex. The window is set the way a route sets it,
through ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` and ``CLAUDE_CODE_MAX_OUTPUT_TOKENS``:
24,000 and 2,000 put the CLI's compaction trigger at 9,000 tokens.

Each test shows, for every adapter, one of the behaviours the CLI needs from
any route: a tool round trip with a compaction that lands mid-turn and a
resume after it, and an upstream overflow the CLI compacts past and retries.
"""

import json
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path
from backend.copilot.sdk.model_gateway.adapters.codex_responses import CodexResponses
from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.route import AdapterKind, GatewayRoute
from backend.copilot.sdk.model_gateway.testing.cli_session import HOOK_NOTE, CliSession
from backend.copilot.sdk.model_gateway.testing.codex_lease import (
    codex_route,
    fake_backend_client,
)
from backend.copilot.sdk.model_gateway.testing.script import (
    COMPACTION_MARKERS,
    Script,
    fact_sheet,
)
from backend.copilot.sdk.model_gateway.testing.scripted_upstream import (
    ScriptedUpstream,
    SeenRequest,
)

_WINDOW, _OUTPUT = 24_000, 2_000
_ADAPTERS: list[AdapterKind] = [
    "anthropic_passthrough",
    "openai_chat",
    "codex_responses",
]
_READ = "Read handbook pages 1 and 2 with read_doc, one page per call, then tell me each page's code word."
_RECALL = "Without calling any tools: which code words did you find?"


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> Path:
    path = _resolve_cli_path()
    assert path is not None, "The bundled or configured Claude Code CLI is required"
    # A run inside Claude Code would hand the CLI its own settings and keys.
    for key in [k for k in os.environ if k.startswith(("CLAUDE", "ANTHROPIC"))]:
        monkeypatch.delenv(key)
    return path


@asynccontextmanager
async def _gateway(
    adapter: AdapterKind, upstream: ScriptedUpstream
) -> AsyncIterator[ModelGateway]:
    async with AsyncExitStack() as stack:
        if adapter == "codex_responses":
            route = codex_route("gpt-codex-test", context_window=_WINDOW)
            client = await stack.enter_async_context(
                fake_backend_client(f"{upstream.url}/codex")
            )
            yield await stack.enter_async_context(
                ModelGateway(route, upstream=CodexResponses(route, client=client))
            )
            return
        base = (
            f"{upstream.url}/anthropic"
            if adapter == "anthropic_passthrough"
            else f"{upstream.url}/openai/v1"
        )
        route = GatewayRoute(
            adapter=adapter,
            model="local-test-model",
            base_url=base,
            context_window=_WINDOW,
        )
        yield await stack.enter_async_context(ModelGateway(route))


@pytest.mark.parametrize("adapter", _ADAPTERS)
async def test_a_tool_round_trip_compacts_mid_turn_and_resumes(adapter, cli, tmp_path):
    async with ScriptedUpstream(Script(pages=2)) as upstream, _gateway(
        adapter, upstream
    ) as gateway:
        session = CliSession(
            cli_path=cli,
            gateway=gateway,
            root=tmp_path,
            window_tokens=_WINDOW,
            output_tokens=_OUTPUT,
            page_chars=24_000,
        )
        (first,) = await session.run([_READ])
        (second,) = await session.run([_RECALL], resume=first.session_id)

    assert upstream.unexpected == []
    assert first.tool_uses == ["mcp__handbook__read_doc"] * 2, session.stderr
    assert (
        not first.is_error and "amber-7" in first.result and "cobalt-14" in first.result
    )
    assert first.compactions >= 1 and upstream.compactions
    compaction = upstream.compactions[0]
    _assert_compaction_request_whole(adapter, compaction.body)
    # The CLI unwraps the summary into its continuation message.
    following = json.dumps(upstream.seen[upstream.seen.index(compaction) + 1].body)
    assert "This session is being continued from a previous conversation" in following
    assert "HANDBOOK PAGE 1. The code word is amber-7." in following
    assert second.session_id == first.session_id
    assert (
        not second.is_error
        and "amber-7" in second.result
        and "cobalt-14" in second.result
    )
    assert gateway.requests == len(upstream.seen)
    if adapter == "codex_responses":
        _assert_reasoning_came_back_intact(upstream)
        assert {seen.body["prompt_cache_key"] for seen in upstream.seen} == {
            first.session_id
        }


@pytest.mark.parametrize("adapter", _ADAPTERS)
async def test_an_upstream_overflow_makes_the_cli_compact_and_retry(
    adapter, cli, tmp_path
):
    # Usage is under-reported, so the CLI never compacts on its own and the
    # upstream's refusal is the only thing that can make it.
    script = Script(overflow_chars=30_000, usage_scale=0.02)
    async with ScriptedUpstream(script) as upstream, _gateway(
        adapter, upstream
    ) as gateway:
        session = CliSession(
            cli_path=cli,
            gateway=gateway,
            root=tmp_path,
            window_tokens=_WINDOW,
            output_tokens=_OUTPUT,
        )
        _, asked = await session.run(
            [
                "Remember these facts: " + fact_sheet(1500),
                "second question: what is fact10?",
            ]
        )

    kinds = [seen.decision.kind for seen in upstream.seen]
    assert kinds[:2] == ["text", "overflow"], kinds
    assert upstream.seen[2].decision.compaction, kinds
    assert kinds[-1] == "text" and not upstream.seen[-1].decision.compaction
    assert asked.compactions == 1
    assert not asked.is_error and "fact10 is number 70." in asked.result, session.stderr


def _assert_compaction_request_whole(
    adapter: AdapterKind, body: dict[str, Any]
) -> None:
    """The tool results, the summary instruction and the hook note all reached
    the upstream, in that order, in its own protocol."""
    instruction = COMPACTION_MARKERS[0]
    if adapter == "anthropic_passthrough":
        *_, last_user, note = body["messages"]
        kinds = [block["type"] for block in last_user["content"]]
        assert "tool_result" in kinds and instruction in json.dumps(last_user)
        assert note["role"] == "system" and HOOK_NOTE in json.dumps(note)
        assert "x-anthropic-billing-header" not in json.dumps(body["system"])
    elif adapter == "openai_chat":
        roles = [message["role"] for message in body["messages"]]
        assert roles[-2:] == ["tool", "user"], roles
        last = body["messages"][-1]["content"]
        assert "<system-reminder>" in last
        assert last.index(instruction) < last.index(HOOK_NOTE)
    else:
        kinds = [item["type"] for item in body["input"]]
        assert kinds[-2:] == ["function_call_output", "message"], kinds
        texts = [part["text"] for part in body["input"][-1]["content"]]
        assert instruction in texts[0] and any(HOOK_NOTE in text for text in texts[1:])
        assert body["store"] is False


def _assert_reasoning_came_back_intact(upstream: ScriptedUpstream) -> None:
    """Every reasoning item sent back is one the backend issued, byte for byte,
    and the tool round's second call carried the first call's reasoning."""
    issued = set(upstream.issued_reasoning)
    returned = [_reasoning_in(seen) for seen in upstream.seen]
    assert all(set(items) <= issued for items in returned)
    assert upstream.issued_reasoning[0] in returned[1]
    assert all(len(item) > 4096 for items in returned for item in items)
    for seen in upstream.seen:
        assert seen.body["store"] is False
        assert seen.body["include"] == ["reasoning.encrypted_content"]


def _reasoning_in(seen: SeenRequest) -> list[str]:
    return [
        item["encrypted_content"]
        for item in seen.body["input"]
        if item.get("type") == "reasoning"
    ]
