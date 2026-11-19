"""The live spend figure the governor lands turns on."""

import pytest
from claude_agent_sdk import AssistantMessage, StreamEvent, TextBlock

from backend.copilot.moonshot import rate_card_usd

from .usage_meter import (
    REFERENCE_MODEL,
    CallUsage,
    UsageMeter,
    cli_price_ratio,
    price_call_usd,
)

KIMI = "moonshotai/kimi-k2.6"


def _message(
    call_id: str | None,
    *,
    model: str = KIMI,
    output: int = 0,
    prompt: int = 0,
    parent: str | None = None,
    usage: bool = True,
) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text="x")],
        model=model,
        parent_tool_use_id=parent,
        usage={"input_tokens": prompt, "output_tokens": output} if usage else None,
        message_id=call_id,
    )


def _kimi_usd(prompt: int, output: int) -> float:
    rates = rate_card_usd(KIMI)
    assert rates is not None
    return (prompt * rates[0] + output * rates[1]) / 1_000_000


def test_a_call_reported_once_per_block_is_counted_once():
    # The CLI repeats a call's AssistantMessage per content block; the last
    # block carries the latest output count.
    meter = UsageMeter()
    meter.observe(_message("gen-1", prompt=10_000, output=200))
    meter.observe(_message("gen-1", prompt=10_000, output=900))
    assert meter.main_calls == 1
    assert meter.spent_usd == pytest.approx(_kimi_usd(10_000, 900))
    assert meter.last_main_call_usd == pytest.approx(_kimi_usd(10_000, 900))


def test_counts_never_go_backwards_within_a_call():
    meter = UsageMeter()
    meter.observe(_message("gen-1", prompt=10_000, output=900))
    meter.observe(_message("gen-1", prompt=10_000, output=100))
    assert meter.spent_usd == pytest.approx(_kimi_usd(10_000, 900))


def test_sub_agent_calls_cost_money_but_are_not_main_steps():
    meter = UsageMeter()
    meter.observe(_message("gen-main", prompt=1_000, output=100))
    meter.observe(_message("gen-sub", prompt=50_000, output=2_000, parent="tu-1"))
    assert meter.main_calls == 1
    assert meter.spent_usd == pytest.approx(
        _kimi_usd(1_000, 100) + _kimi_usd(50_000, 2_000)
    )
    assert meter.last_main_call_usd == pytest.approx(_kimi_usd(1_000, 100))


@pytest.mark.parametrize(
    "message", [_message(None, output=100), _message("gen-1", usage=False)]
)
def test_messages_without_an_id_or_usage_are_ignored(message):
    meter = UsageMeter()
    meter.observe(message)
    assert meter.main_calls == 0
    assert meter.spent_usd == 0.0


def test_anthropic_calls_use_the_rate_card_with_cache_discounts():
    cached = CallUsage(
        model="anthropic/claude-sonnet-5", input_tokens=1_000, cache_read_tokens=99_000
    )
    uncached = CallUsage(model="anthropic/claude-sonnet-5", input_tokens=100_000)
    assert 0 < price_call_usd(cached) < price_call_usd(uncached) / 5


def test_an_unknown_model_is_priced_like_the_reference():
    unknown = CallUsage(
        model="someone/new-model", input_tokens=10_000, output_tokens=500
    )
    reference = CallUsage(model=REFERENCE_MODEL, input_tokens=10_000, output_tokens=500)
    assert price_call_usd(unknown) == pytest.approx(price_call_usd(reference))


@pytest.mark.parametrize(
    "reported, alias",
    [
        # OpenRouter's slug, as the CLI reports a call on that route.
        ("anthropic/claude-sonnet-5-5", "claude-sonnet-5-5"),
        ("anthropic/claude-opus-4.6", "claude-opus-4-6"),
        # The API's dated snapshot of an alias the card keys undated.
        ("claude-sonnet-5-5-20260930", "claude-sonnet-5-5"),
    ],
)
def test_a_call_is_priced_at_its_models_rate_however_its_route_names_it(
    reported: str, alias: str, caplog: pytest.LogCaptureFixture
):
    # Missing the card priced these as Opus 4.1: about 7x a Sonnet 5.5 turn.
    usage = {"input_tokens": 2_600, "cache_read_tokens": 24_000, "output_tokens": 30}
    as_reported = price_call_usd(CallUsage(model=reported, **usage))
    assert as_reported == pytest.approx(price_call_usd(CallUsage(model=alias, **usage)))
    assert "no entry for model" not in caplog.text


def test_cli_price_ratio_is_one_for_anthropic_and_unknown_models():
    assert cli_price_ratio("anthropic/claude-sonnet-5") == 1.0
    assert cli_price_ratio("claude-sonnet-4-5") == 1.0
    assert cli_price_ratio("someone/new-model") == 1.0
    assert cli_price_ratio(None) == 1.0


def test_cli_price_ratio_covers_the_cli_pricing_kimi_like_sonnet():
    # Sonnet is $3/$15 to Kimi K2.x's $0.60/$2.80: the larger ratio, output,
    # is what keeps the backstop from firing before the real budget is spent.
    assert cli_price_ratio(KIMI) == pytest.approx(15 / 2.80, rel=0.05)


def _start(call_id: str, *, prompt: int = 0, parent: str | None = None) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id="s",
        event={
            "type": "message_start",
            "message": {
                "id": call_id,
                "model": KIMI,
                "usage": {"input_tokens": prompt, "output_tokens": 1},
            },
        },
        parent_tool_use_id=parent,
    )


def _delta(output: int, *, parent: str | None = None) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id="s",
        event={"type": "message_delta", "usage": {"output_tokens": output}},
        parent_tool_use_id=parent,
    )


def test_output_comes_from_the_stream_not_the_placeholder():
    # As the CLI streams it: message_start and every assistant message carry
    # the placeholder output count; message_delta carries the real one.
    meter = UsageMeter()
    meter.observe_stream(_start("m1", prompt=100))
    meter.observe(_message("m1", prompt=100, output=1))
    meter.observe(_message("m1", prompt=100, output=1))
    meter.observe_stream(_delta(10_000))
    assert meter.spent_usd == pytest.approx(_kimi_usd(100, 10_000))
    assert meter.main_calls == 1


def test_a_sub_agents_stream_is_kept_apart_from_the_main_agents():
    meter = UsageMeter()
    meter.observe_stream(_start("main-1", prompt=100))
    meter.observe_stream(_start("sub-1", prompt=50, parent="tu-1"))
    meter.observe_stream(_delta(2_000, parent="tu-1"))
    meter.observe_stream(_delta(500))
    assert meter.spent_usd == pytest.approx(_kimi_usd(100, 500) + _kimi_usd(50, 2_000))
    assert meter.main_calls == 1
    assert meter.last_main_call_usd == pytest.approx(_kimi_usd(100, 500))


def test_a_delta_with_no_call_started_on_its_stream_is_ignored():
    meter = UsageMeter()
    meter.observe_stream(_delta(10_000))
    meter.observe_stream(_start("main-1"))
    meter.observe_stream(_delta(10_000, parent="tu-unknown"))
    assert meter.spent_usd == pytest.approx(_kimi_usd(0, 1))


def test_the_next_response_on_a_stream_gets_its_own_delta():
    meter = UsageMeter()
    meter.observe_stream(_start("m1"))
    meter.observe_stream(_delta(300))
    meter.observe_stream(_start("m2"))
    meter.observe_stream(_delta(700))
    assert meter.spent_usd == pytest.approx(_kimi_usd(0, 300) + _kimi_usd(0, 700))
    assert meter.main_calls == 2
