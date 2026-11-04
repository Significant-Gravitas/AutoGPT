import json

import pytest
from aiohttp import web

from backend.copilot.baseline import service as baseline
from backend.copilot.model import ChatMessage, ChatSession

from . import provider as provider_module
from .baseline_turns import BaselineTurns
from .invoice import ANTHROPIC_PRICES, Price
from .ledger import PlatformLedger
from .provider import FakeProvider
from .replies import TOOL_MARKER


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize(
    "prompt_tokens,expected", [(100_000, 20_000), (150_000, 85_000)]
)
@pytest.mark.parametrize("report_usage", [False, True])
async def test_missing_usage_does_not_tier_the_whole_turn_output(
    provider: FakeProvider,
    ledger: PlatformLedger,
    monkeypatch: pytest.MonkeyPatch,
    prompt_tokens: int,
    expected: int,
    report_usage: bool,
):
    original_stream = provider_module._stream
    monkeypatch.setitem(
        ANTHROPIC_PRICES,
        "claude-haiku-5-5",
        Price(
            input=0.1,
            output=0.5,
            cache_read=0.01,
            cache_write_5m=0.125,
            cache_write_1h=0.2,
        ),
    )

    async def stream(request: web.Request, sse: str) -> web.StreamResponse:
        if not report_usage:
            sse = "\n\n".join(
                event
                for event in sse.split("\n\n")
                if not event
                or event == "data: [DONE]"
                or "usage" not in json.loads(event.removeprefix("data: "))
            )
        return await original_stream(request, sse)

    monkeypatch.setattr(provider_module, "_stream", stream)
    monkeypatch.setattr(
        baseline, "estimate_token_count", lambda *_, **__: prompt_tokens
    )
    monkeypatch.setattr(baseline, "estimate_token_count_str", lambda *_, **__: 20_000)
    config = baseline.config.model_copy(
        update={
            "use_local": False,
            "use_openrouter": False,
            "use_claude_code_subscription": False,
            "use_e2b_sandbox": False,
            "baseline_prompt_cache_ttl": "1h",
        }
    )
    run_turn = BaselineTurns(
        monkeypatch,
        config,
        base_url=f"{provider.url}/anthropic/v1",
        model="claude-haiku-5-5",
    )
    session = ChatSession.new("haiku-recovery-test", dry_run=False)
    session.title = "Recovery regression"
    prompt = f"Look this up {TOOL_MARKER}"
    session.messages.append(ChatMessage(role="user", content=prompt))
    try:
        await run_turn(session, prompt, new_pod=False)
        await ledger.settle()
    finally:
        await run_turn.aclose()
    assert run_turn.client.is_closed
    assert not provider.unexpected
    assert len(provider.calls) == 2
    if report_usage:
        expected = round(sum(call.invoice_usd for call in provider.calls) * 1_000_000)
    assert [charge.microdollars for charge in ledger.spend] == [expected]
    assert [charge.microdollars for charge in ledger.tree] == [expected]
