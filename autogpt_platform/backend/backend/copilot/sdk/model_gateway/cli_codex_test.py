"""The real CLI on the Codex route: the billing suite's backend, and reasoning
that arrives with no summary text at all.

The first runs the CLI through the gateway against
``backend.copilot.billing_e2e``'s fake Responses endpoint, the one the billing
suite bills Codex turns from, and checks the gateway's usage matches what that
backend reported. The second makes the backend send reasoning items that are
encrypted content alone, so the thinking block the CLI keeps has an empty text
and only a signature, and checks the CLI still hands it back.
"""

import os
from pathlib import Path

import pytest

from backend.copilot.billing_e2e.provider import FakeProvider
from backend.copilot.billing_e2e.replies import AFTER_TOOL, GREETING
from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path
from backend.copilot.sdk.model_gateway.adapters.codex_responses import CodexResponses
from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.testing.cli_session import CliSession
from backend.copilot.sdk.model_gateway.testing.codex_lease import (
    codex_route,
    fake_backend_client,
)
from backend.copilot.sdk.model_gateway.testing.script import Script
from backend.copilot.sdk.model_gateway.testing.scripted_upstream import ScriptedUpstream
from backend.copilot.sdk.model_gateway.usage import GatewayUsage
from backend.data.llm_registry.llm_models import LLMModel

_MODEL = LLMModel.GPT5_6_TERRA.value
_WINDOW, _OUTPUT = 272_000, 8_000


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> Path:
    path = _resolve_cli_path()
    assert path is not None, "The bundled or configured Claude Code CLI is required"
    for key in [k for k in os.environ if k.startswith(("CLAUDE", "ANTHROPIC"))]:
        monkeypatch.delenv(key)
    return path


async def test_a_codex_turn_through_the_billing_backend_bills_what_it_reported(
    cli, tmp_path
):
    route = codex_route(_MODEL, context_window=_WINDOW)
    async with FakeProvider() as provider:
        async with fake_backend_client(f"{provider.url}/codex") as client:
            async with ModelGateway(
                route, upstream=CodexResponses(route, client=client)
            ) as gateway:
                session = CliSession(
                    cli_path=cli,
                    gateway=gateway,
                    root=tmp_path,
                    window_tokens=_WINDOW,
                    output_tokens=_OUTPUT,
                )
                (turn,) = await session.run(["Please [use-tool] and report back."])

    assert provider.unexpected == []
    assert not turn.is_error, session.stderr
    # The backend asked for a tool the CLI does not have and the CLI said so.
    # That backend answers only the last input item, and CLI 2.1.284 sends its
    # token-budget note after the tool result, so it may greet instead.
    assert turn.tool_uses == ["lookup_fact"]
    assert turn.result in (AFTER_TOOL, GREETING)
    billed = [call.usage for call in provider.calls]
    assert len(billed) == gateway.requests == 2
    assert gateway.usage == GatewayUsage(
        input_tokens=sum(u.input_tokens for u in billed),
        cache_read_tokens=sum(u.cache_read_tokens for u in billed),
        output_tokens=sum(u.output_tokens for u in billed),
    )


async def test_reasoning_without_a_summary_still_comes_back(cli, tmp_path):
    route = codex_route("gpt-codex-test", context_window=_WINDOW)
    script = Script(pages=2, reasoning_summary=False)
    async with ScriptedUpstream(script) as upstream:
        async with fake_backend_client(f"{upstream.url}/codex") as client:
            async with ModelGateway(
                route, upstream=CodexResponses(route, client=client)
            ) as gateway:
                session = CliSession(
                    cli_path=cli,
                    gateway=gateway,
                    root=tmp_path,
                    window_tokens=_WINDOW,
                    output_tokens=_OUTPUT,
                )
                (turn,) = await session.run(
                    [
                        "Read handbook pages 1 and 2 with read_doc, one page per"
                        " call, then tell me each page's code word."
                    ]
                )

    assert not turn.is_error and "cobalt-14" in turn.result, session.stderr
    assert turn.signatures and all(turn.signatures)
    sent_back = [
        item["encrypted_content"]
        for seen in upstream.seen[1:]
        for item in seen.body["input"]
        if item.get("type") == "reasoning"
    ]
    assert upstream.issued_reasoning[0] in sent_back
    assert set(sent_back) <= set(upstream.issued_reasoning)
