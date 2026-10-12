from backend.copilot.sdk.model_gateway.stream import AnthropicStreamBuilder
from backend.copilot.sdk.model_gateway.usage import (
    GatewayUsage,
    estimate_tokens,
    merge_stream_usage,
    usage_from_anthropic,
)


def test_output_only_progress_updates_leave_input_unknown():
    usage = merge_stream_usage(
        GatewayUsage(output_tokens=1), GatewayUsage(output_tokens=2)
    )
    events = AnthropicStreamBuilder(model="m", input_estimate=100).finish(
        "end_turn", usage
    )
    assert events[-2]["usage"]["input_tokens"] == 100
    assert events[-2]["usage"]["output_tokens"] == 2


def test_the_estimate_counts_text_in_system_messages_and_tools():
    body = {
        "model": "ignored-by-the-estimate",
        "system": "x" * 400,
        "messages": [{"role": "user", "content": "y" * 400}],
        "tools": [{"name": "t", "description": "z" * 400}],
    }
    # String values only: the system text, the role and content, the tool's
    # name and description.
    assert estimate_tokens(body) == (400 + len("user") + 400 + len("t") + 400) // 4


def test_an_image_costs_a_flat_amount_whatever_its_size():
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "A" * 900_000},
    }
    body = {"messages": [{"role": "user", "content": [image]}]}
    assert estimate_tokens(body) < 2_000


def test_the_estimate_is_never_zero():
    assert estimate_tokens({}) == 1


def test_anthropic_usage_is_read_into_its_buckets():
    usage = usage_from_anthropic(
        {
            "input_tokens": 10,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 30,
            "output_tokens": 5,
        }
    )
    assert usage == GatewayUsage(
        input_tokens=10, cache_read_tokens=20, cache_write_tokens=30, output_tokens=5
    )
    assert usage.prompt_tokens == 60


def test_garbage_usage_reads_as_zero():
    assert usage_from_anthropic({"input_tokens": "lots", "output_tokens": -3}) == (
        GatewayUsage()
    )
    assert usage_from_anthropic(None) is None


def test_the_delta_replaces_the_start_only_where_it_reports_something():
    start = GatewayUsage(input_tokens=900, cache_read_tokens=50)
    delta = GatewayUsage(input_tokens=700, output_tokens=12)
    assert merge_stream_usage(start, delta) == GatewayUsage(
        input_tokens=700, cache_read_tokens=50, output_tokens=12
    )


def test_usage_adds_up_across_requests():
    total = GatewayUsage(input_tokens=1, output_tokens=2, reasoning_tokens=3).plus(
        GatewayUsage(input_tokens=10, cache_read_tokens=4, reasoning_tokens=1)
    )
    assert total == GatewayUsage(
        input_tokens=11, cache_read_tokens=4, output_tokens=2, reasoning_tokens=4
    )
