"""A real streaming client bills three turns from each request's token tier."""

import pytest

from backend.copilot.baseline import service as baseline
from backend.copilot.model import ChatMessage, ChatSession

from . import provider as provider_module
from .baseline_turns import BaselineTurns
from .invoice import ANTHROPIC_PRICES, Price, Usage
from .ledger import PlatformLedger
from .provider import FakeProvider
from .replies import TOOL_MARKER

MODEL = "claude-haiku-5-5"
BASE_PRICE = Price(
    input=0.1, output=0.5, cache_read=0.01, cache_write_5m=0.125, cache_write_1h=0.2
)
LONG_PRICE = Price(
    input=0.5, output=2.5, cache_read=0.05, cache_write_5m=0.625, cache_write_1h=1.0
)


@pytest.mark.asyncio(loop_scope="function")
async def test_three_turns_charge_each_request_at_its_prompt_tier(
    provider: FakeProvider,
    ledger: PlatformLedger,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = [
        (
            Usage(
                input_tokens=20_000,
                output_tokens=2_000,
                cache_read_tokens=70_000,
                cache_write_tokens=10_000,
            ),
            BASE_PRICE,
        ),
        (
            Usage(
                input_tokens=20_000,
                output_tokens=2_000,
                cache_read_tokens=120_000,
                cache_write_tokens=10_000,
            ),
            LONG_PRICE,
        ),
        (
            Usage(
                input_tokens=20_000,
                output_tokens=1_000,
                cache_read_tokens=20_000,
                cache_write_tokens=10_000,
            ),
            BASE_PRICE,
        ),
        (Usage(input_tokens=30_000, output_tokens=1_000), BASE_PRICE),
    ]

    def scripted_usage(index: int, *, cached: bool) -> Usage:
        assert cached
        usage, price = calls[index]
        monkeypatch.setitem(ANTHROPIC_PRICES, MODEL, price)
        return usage

    monkeypatch.setattr(provider_module, "scripted_usage", scripted_usage)
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
        monkeypatch, config, base_url=f"{provider.url}/anthropic/v1", model=MODEL
    )
    session = ChatSession.new("haiku-billing-test", dry_run=False)
    session.title = "Tier pricing test"
    prompts = ("Hello", f"Look this up {TOOL_MARKER}", "Thanks")
    try:
        for turn, prompt in enumerate(prompts):
            provider.turn = ledger.turn = turn
            session.messages.append(ChatMessage(role="user", content=prompt))
            await run_turn(session, prompt, new_pod=turn == 2)
            await ledger.settle()
    finally:
        await run_turn.aclose()

    assert run_turn.client.is_closed()
    assert not provider.unexpected
    assert [len(provider.calls_in(turn)) for turn in range(3)] == [1, 2, 1]
    assert [charge.microdollars for charge in ledger.spend] == [5_700, 35_700, 3_500]
    assert [charge.microdollars for charge in ledger.tree] == [5_700, 35_700, 3_500]
    assert sum(charge.microdollars for charge in ledger.spend) == 44_900
    assert [call.invoice_usd for call in provider.calls] == pytest.approx(
        [0.0057, 0.031, 0.0047, 0.0035]
    )
    assert len(ledger.cost_log) == 3
    assert [row.entry.cost_microdollars for row in ledger.cost_log] == [
        5_700,
        35_700,
        3_500,
    ]
