"""A direct-Anthropic turn that reports no usage is priced request by request."""

import json
from typing import Any

import pytest
from aiohttp import web

from backend.copilot.baseline import service as baseline
from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatMessage, ChatSession
from backend.util.prompt import estimate_token_count, estimate_token_count_str

from . import provider as provider_module
from . import wire
from .baseline_turns import BaselineTurns
from .invoice import ANTHROPIC_PRICES, Price, Route
from .ledger import PlatformLedger
from .provider import FakeProvider
from .replies import AFTER_TOOL, TOOL_MARKER

MODEL = "claude-haiku-5-5"
HAIKU = Price(
    input=0.1, output=0.5, cache_read=0.01, cache_write_5m=0.125, cache_write_1h=0.2
)
# Every request's estimated output, whether it answered or called a tool.
OUTPUT_TOKENS = 20_000
# A document the model writes through its tool call, as the call's arguments.
DOCUMENT = {
    "path": "notes.md",
    "content": " ".join(
        f"Note {i}: the receipts are in the blue folder." for i in range(2_000)
    ),
}


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("prompt_tokens", [100_000, 150_000])
@pytest.mark.parametrize("report_usage", [False, True])
@pytest.mark.parametrize("requests", [1, 2])
async def test_missing_usage_prices_each_request_at_its_own_tier(
    provider: FakeProvider,
    ledger: PlatformLedger,
    monkeypatch: pytest.MonkeyPatch,
    prompt_tokens: int,
    report_usage: bool,
    requests: int,
):
    monkeypatch.setitem(ANTHROPIC_PRICES, MODEL, HAIKU)
    if not report_usage:
        _drop_usage(monkeypatch)
    monkeypatch.setattr(
        baseline, "estimate_token_count", lambda *_, **__: prompt_tokens
    )
    monkeypatch.setattr(
        baseline,
        "estimate_token_count_str",
        lambda value, **_: 0 if isinstance(value, list) else OUTPUT_TOKENS,
    )
    run_turn = BaselineTurns(
        monkeypatch, _direct_config(), provider, path="/anthropic/v1", model=MODEL
    )
    session = ChatSession.new("haiku-recovery-test", dry_run=False)
    session.title = "Recovery regression"
    prompt = f"Look this up {TOOL_MARKER}" if requests == 2 else "Hello"
    session.messages.append(ChatMessage(role="user", content=prompt))
    await run_turn(session, prompt, new_pod=False)
    await ledger.settle()
    assert not provider.unexpected
    assert len(provider.calls) == requests
    if report_usage:
        expected = round(sum(call.invoice_usd for call in provider.calls) * 1_000_000)
    else:
        # The tool-call request bills its call as output, as the answering
        # one bills its answer, each at the price of its own prompt's tier.
        expected = requests * _request_microdollars(prompt_tokens, OUTPUT_TOKENS)
    assert [charge.microdollars for charge in ledger.spend] == [expected]
    assert [charge.microdollars for charge in ledger.tree] == [expected]


@pytest.mark.asyncio(loop_scope="function")
async def test_a_tool_call_bills_its_name_and_arguments_as_output(
    ledger: PlatformLedger, monkeypatch: pytest.MonkeyPatch
):
    # Real token estimates, so a call priced from its name alone would show.
    monkeypatch.setitem(ANTHROPIC_PRICES, MODEL, HAIKU)
    monkeypatch.setattr(wire, "TOOL_ARGUMENTS", DOCUMENT)
    _drop_usage(monkeypatch)
    async with RecordingProvider() as provider:
        run_turn = BaselineTurns(
            monkeypatch, _direct_config(), provider, path="/anthropic/v1", model=MODEL
        )
        session = ChatSession.new("haiku-tool-output-test", dry_run=False)
        session.title = "Tool output regression"
        prompt = f"Save my notes {TOOL_MARKER}"
        session.messages.append(ChatMessage(role="user", content=prompt))
        await run_turn(session, prompt, new_pod=False)
        await ledger.settle()

    assert not provider.unexpected
    prompts = [
        estimate_token_count(body["messages"], model=MODEL)
        + estimate_token_count_str(body["tools"], model=MODEL)
        for body in provider.bodies
    ]
    outputs = [
        estimate_token_count_str(wire.TOOL_NAME + json.dumps(DOCUMENT), model=MODEL),
        estimate_token_count_str(AFTER_TOOL, model=MODEL),
    ]
    assert len(prompts) == 2 and max(prompts) < 100_000
    assert outputs[0] > 10_000
    # Both requests are at the base tier: each prompt at the input price, and
    # what each wrote, the tool call or the answer, at the output price.
    expected = sum(prompts) * 0.1 + sum(outputs) * 0.5
    charged = [charge.microdollars for charge in ledger.spend]
    assert len(charged) == 1 and abs(charged[0] - expected) <= 1
    assert [charge.microdollars for charge in ledger.tree] == charged


class RecordingProvider(FakeProvider):
    """Keeps each chat request's body, to estimate the prompt it sent."""

    def __init__(self) -> None:
        super().__init__()
        self.bodies: list[dict[str, Any]] = []

    async def _chat(self, request: web.Request, route: Route) -> web.StreamResponse:
        self.bodies.append(await request.json())
        return await super()._chat(request, route)


def _direct_config() -> ChatConfig:
    return baseline.config.model_copy(
        update={
            "use_local": False,
            "use_openrouter": False,
            "use_claude_code_subscription": False,
            "use_e2b_sandbox": False,
            "baseline_prompt_cache_ttl": "1h",
        }
    )


def _drop_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stream without the usage chunk, as direct Anthropic does."""
    original_stream = provider_module._stream

    async def stream(request: web.Request, sse: str) -> web.StreamResponse:
        events = (
            event
            for event in sse.split("\n\n")
            if not event
            or event == "data: [DONE]"
            or "usage" not in json.loads(event.removeprefix("data: "))
        )
        return await original_stream(request, "\n\n".join(events))

    monkeypatch.setattr(provider_module, "_stream", stream)


def _request_microdollars(prompt_tokens: int, output_tokens: int) -> int:
    """Haiku 5.5's published prices, by one request's prompt length."""
    long = prompt_tokens > 100_000
    return round(
        prompt_tokens * (0.5 if long else 0.1) + output_tokens * (2.5 if long else 0.5)
    )
