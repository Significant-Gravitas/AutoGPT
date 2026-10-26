"""Dream inference over real HTTP through the production usage ledger seam.

The provider speaks Anthropic's wire format. Ledger writes use the existing
billing_e2e capture, so these tests do not require a database or Redis.
"""

import pytest

from backend.copilot.dream import billing
from backend.copilot.dream.phase_jobs import phase_job
from backend.copilot.dream.schemas import ConsolidationOutput
from backend.copilot.inference import routing as inference_routing
from backend.copilot.inference.complete import structured_complete
from backend.copilot.inference.context import (
    InferenceContext,
    InferenceError,
    InferenceScope,
)
from backend.copilot.inference.routing import resolve_route
from backend.copilot.transport_routing import ProviderRoutingKwargs
from backend.data.block_cost_config import compute_token_credits
from backend.data.execution import NodeExecutionStats

from . import dream_provider as provider_fixtures
from .ledger import PlatformLedger


def _context(model: str) -> InferenceContext:
    scope = InferenceScope(user_id="dream-owner")
    job = phase_job("consolidate", "dream-pass", timeout_seconds=30, pinned_model=model)
    return InferenceContext(scope=scope, job=job, route=resolve_route(scope, job))


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("prose", [False, True])
@pytest.mark.parametrize(
    "model,expected_microdollars",
    [
        ("anthropic/claude-sonnet-5-5", 5605),
        ("anthropic/claude-opus-5-5", 11210),
        ("anthropic/claude-haiku-4-5", 2953),
        ("anthropic/claude-haiku-5-5", 295),
        ("anthropic/claude-sonnet-5", 5906),
        ("anthropic/claude-fable-5-1", 27274),
    ],
)
async def test_native_dream_http_usage_reaches_ledger(
    model: str,
    expected_microdollars: int,
    prose: bool,
    dream_provider: provider_fixtures.DreamProvider,
    ledger: PlatformLedger,
) -> None:
    dream_provider.prose = prose
    ctx = _context(model)
    completion = await structured_complete(
        ctx,
        messages=[{"role": "user", "content": "Consolidate facts."}],
        response_model=ConsolidationOutput,
    )
    assert completion.value.facts == []
    await billing.record_phase_cost(ctx, completion.usage)
    await ledger.settle()
    assert len(dream_provider.requests) == 1
    sent = dream_provider.requests[0]
    assert "facts" in sent["tools"][0]["input_schema"]["properties"]
    assert sent["model"] == model.removeprefix("anthropic/")
    if model.endswith(("sonnet-5-5", "opus-5-5", "fable-5-1")):
        assert sent["tool_choice"]["type"] == "auto"
    assert [(s.user_id, s.microdollars, s.skip_daily) for s in ledger.spend] == [
        ("dream-owner", expected_microdollars, True)
    ]
    (row,) = ledger.rows_in(0)
    assert row.cost_microdollars == expected_microdollars
    assert (
        row.input_tokens,
        row.output_tokens,
        row.cache_read_tokens,
        row.cache_creation_tokens,
    ) == (1001, 203, 3007, 509)
    assert row.graph_exec_id == "dream-pass"
    assert row.provider == "anthropic"
    assert row.metadata is not None
    assert row.metadata["discount_applied"] == 0.0


@pytest.mark.asyncio(loop_scope="function")
async def test_truncated_http_response_preserves_usage_but_rejects_valid_json(
    dream_provider: provider_fixtures.DreamProvider,
) -> None:
    dream_provider.stop_reason = "max_tokens"
    with pytest.raises(InferenceError, match="max_tokens") as caught:
        await structured_complete(
            _context("anthropic/claude-sonnet-5-5"),
            messages=[{"role": "user", "content": "Consolidate facts."}],
            response_model=ConsolidationOutput,
        )
    assert caught.value.usage is not None
    assert caught.value.usage.input_tokens == 1001
    assert caught.value.usage.output_tokens == 203
    assert caught.value.usage.cache_read_tokens == 3007
    assert caught.value.usage.cache_creation_tokens == 509


@pytest.mark.parametrize(
    "cache_tokens,expected", [(1, 1), (66666, 1), (66667, 2), (1000000, 15)]
)
def test_sonnet_cache_credit_rate_and_rounding(
    cache_tokens: int, expected: int
) -> None:
    stats = NodeExecutionStats(cache_read_token_count=cache_tokens)
    assert compute_token_credits({"model": "claude-sonnet-5-5"}, stats) == expected


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize(
    "reported_cost,expected_microdollars", [(None, 5605), (0.0042, 4200)]
)
async def test_openrouter_cache_buckets_and_reported_cost(
    reported_cost: float | None,
    expected_microdollars: int,
    dream_provider: provider_fixtures.DreamProvider,
    ledger: PlatformLedger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dream_provider.cost = reported_cost
    routing = ProviderRoutingKwargs(
        provider="open_router",
        api_key="sk-test",
        base_url=None,
        supports_flex=True,
        cost_log_provider="open_router",
    )
    monkeypatch.setattr(
        inference_routing, "routing_kwargs_for_chat_transport", lambda: routing
    )
    ctx = _context("anthropic/claude-sonnet-5-5")
    completion = await structured_complete(
        ctx,
        messages=[{"role": "user", "content": "Consolidate facts as JSON."}],
        response_model=ConsolidationOutput,
    )
    await billing.record_phase_cost(ctx, completion.usage)
    await ledger.settle()
    (row,) = ledger.rows_in(0)
    assert row.input_tokens == 1001
    assert row.cache_read_tokens == 3007
    assert row.cache_creation_tokens == 509
    assert row.cost_microdollars == expected_microdollars
    assert [s.microdollars for s in ledger.spend] == [expected_microdollars]


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("cache_read,cache_write", [(3007, 0), (0, 509), (3007, 509)])
async def test_unpriced_cache_only_http_response_keeps_one_token_row(
    cache_read: int,
    cache_write: int,
    dream_provider: provider_fixtures.DreamProvider,
    ledger: PlatformLedger,
) -> None:
    dream_provider.usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    }
    ctx = _context("anthropic/claude-unpriced-test")
    completion = await structured_complete(
        ctx,
        [{"role": "user", "content": "Consolidate facts."}],
        ConsolidationOutput,
    )
    await billing.record_phase_cost(ctx, completion.usage)
    await ledger.settle()
    (row,) = ledger.rows_in(0)
    assert (row.input_tokens, row.output_tokens) == (0, 0)
    assert (row.cache_read_tokens or 0, row.cache_creation_tokens or 0) == (
        cache_read,
        cache_write,
    )
    assert row.cost_microdollars is None
    assert row.tracking_type == "tokens"
    assert ledger.spend == []


dream_provider = provider_fixtures.dream_provider
