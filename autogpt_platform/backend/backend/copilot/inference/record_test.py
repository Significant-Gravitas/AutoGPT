"""``record`` is the one call from each migrated background job (dream phase,
briefing lede, consult) into the cost ledger."""

from unittest.mock import AsyncMock, patch

import pytest

from .context import (
    InferenceContext,
    InferenceJob,
    InferenceScope,
    InferenceUsage,
    RouteDecision,
)
from .record import price, record
from .routing import anthropic_batch_route

# Sonnet 5's catalog card: $2 in, $10 out, $0.20 cache read, $2.50 cache
# write per Mtok, every bucket billed on its own.
_SONNET_5_LIST_COST = 2.0 + 1.0 + 0.04 + 0.025


def _sync_route(
    *, provider="open_router", payer="platform_allowance", label="open_router"
) -> RouteDecision:
    return RouteDecision(
        engine="provider_sync",
        auth_provider="platform",
        provider=provider,
        model="anthropic/claude-sonnet-5",
        payer=payer,
        execution_path="sync_baseline",
        cost_log_provider=label,
        reason="test",
    )


def _ctx(
    kind="dream",
    phase="consolidate",
    *,
    route: RouteDecision | None = None,
    expert_id: str | None = "e1",
    trace_id: str | None = None,
) -> InferenceContext:
    return InferenceContext(
        scope=InferenceScope(user_id="u1", expert_id=expert_id),
        job=InferenceJob(
            kind=kind,
            phase=phase,
            correlation_id="corr-1",
            latency_class="deferred" if kind == "dream" else "bounded",
            tier="standard",
        ),
        route=route or _sync_route(),
        trace_id=trace_id,
    )


def _usage(cost_usd: float | None = None, model="anthropic/claude-sonnet-5", **tokens):
    return InferenceUsage(
        model=model,
        input_tokens=tokens.get("input_tokens", 1_000_000),
        output_tokens=tokens.get("output_tokens", 100_000),
        cache_read_tokens=tokens.get("cache_read_tokens", 200_000),
        cache_creation_tokens=tokens.get("cache_creation_tokens", 10_000),
        cost_usd=cost_usd,
        cost_source="none" if cost_usd is None else "provider",
        payer="platform_allowance",
    )


@pytest.fixture
def persist():
    with patch(
        "backend.copilot.inference.record.persist_and_record_usage", AsyncMock()
    ) as mock:
        yield mock


@pytest.mark.asyncio
async def test_row_carries_the_route_label_credential_and_expert(persist):
    await record(_ctx(), _usage(0.01), block_name="copilot:dream:consolidate")

    kwargs = persist.await_args.kwargs
    assert kwargs["provider"] == "open_router"
    # ``None`` is the platform's own key; the ledger writes it as the copilot
    # system credential.
    assert kwargs["credential_id_override"] is None
    assert kwargs["expert_id"] == "e1"
    assert kwargs["user_id"] == "u1"
    assert kwargs["block_name_override"] == "copilot:dream:consolidate"
    assert kwargs["model"] == "anthropic/claude-sonnet-5"
    assert kwargs["session"] is None


@pytest.mark.asyncio
async def test_row_metadata_names_the_job_the_path_and_who_pays(persist):
    await record(
        _ctx(),
        _usage(0.01),
        block_name="b",
        metadata={"dream_pass_id": "corr-1", "source": "caller-cannot-override"},
    )

    assert persist.await_args.kwargs["extra_metadata"] == {
        "dream_pass_id": "corr-1",
        "source": "dream_pass",
        "job_kind": "dream",
        "phase": "consolidate",
        "execution_path": "sync_baseline",
        "billing_mode": "platform",
        "discount_applied": 0.0,
        "expert_id": "e1",
    }


@pytest.mark.asyncio
async def test_row_links_the_langfuse_trace_when_there_is_one(persist):
    await record(_ctx(trace_id="t-123"), _usage(0.01), block_name="b")
    assert persist.await_args.kwargs["extra_metadata"]["langfuse_trace_id"] == "t-123"


@pytest.mark.asyncio
async def test_account_scope_leaves_the_expert_empty(persist):
    await record(_ctx(expert_id=None), _usage(0.01), block_name="b")
    assert persist.await_args.kwargs["expert_id"] is None
    assert persist.await_args.kwargs["extra_metadata"]["expert_id"] is None


@pytest.mark.parametrize(
    "kind,columns,skip_daily,source,path",
    [
        # A pass's rows join under its id, on the dream's own path label.
        ("dream", ("corr-1", None), True, "dream_pass", "sync_baseline"),
        # A consult belongs to the chat that asked, and to its daily budget.
        ("consult", (None, "corr-1"), False, "copilot", "sync"),
        # The admin view would read a briefing's graphExecId as a chat.
        ("briefing_narrative", (None, None), True, "morning_briefing", "sync"),
    ],
)
@pytest.mark.asyncio
async def test_each_kind_lands_where_the_cost_views_read_it(
    persist, kind, columns, skip_daily, source, path
):
    await record(_ctx(kind, None), _usage(0.01), block_name="b")

    kwargs = persist.await_args.kwargs
    assert (
        kwargs["graph_exec_id_override"],
        kwargs["chat_session_id_override"],
    ) == columns
    assert kwargs["skip_daily"] is skip_daily
    assert kwargs["extra_metadata"]["source"] == source
    assert kwargs["extra_metadata"]["execution_path"] == path


@pytest.mark.asyncio
async def test_a_consult_joins_its_chats_usage(persist):
    session = object()
    await record(_ctx("consult", None), _usage(0.01), block_name="b", session=session)
    assert persist.await_args.kwargs["session"] is session


@pytest.mark.asyncio
async def test_the_providers_cost_is_what_gets_charged(persist):
    priced = await record(_ctx(), _usage(0.0042), block_name="b")
    assert persist.await_args.kwargs["cost_usd"] == 0.0042
    assert (priced.cost_usd, priced.cost_source) == (0.0042, "provider")


@pytest.mark.asyncio
async def test_an_unpriced_call_is_priced_from_the_catalog(persist):
    priced = await record(_ctx(), _usage(None), block_name="b")

    assert persist.await_args.kwargs["cost_usd"] == pytest.approx(_SONNET_5_LIST_COST)
    assert priced.cost_source == "catalog"
    assert priced.cost_usd == pytest.approx(_SONNET_5_LIST_COST)


@pytest.mark.asyncio
async def test_the_batch_route_is_priced_at_half_and_says_so(persist):
    ctx = _ctx(route=anthropic_batch_route("claude-sonnet-5"))
    await record(ctx, _usage(None, model="claude-sonnet-5"), block_name="b")

    kwargs = persist.await_args.kwargs
    assert kwargs["cost_usd"] == pytest.approx(_SONNET_5_LIST_COST / 2)
    assert kwargs["provider"] == "anthropic"
    assert kwargs["extra_metadata"]["execution_path"] == "anthropic_batch"
    assert kwargs["extra_metadata"]["discount_applied"] == 0.5


@pytest.mark.asyncio
async def test_a_model_with_no_catalog_price_logs_tokens_without_a_charge(persist):
    """Unknown, never zero: the row keeps its tokens and charges nothing."""
    priced = await record(
        _ctx(), _usage(None, model="vendor/no-such-model"), block_name="b"
    )

    kwargs = persist.await_args.kwargs
    assert kwargs["cost_usd"] is None
    assert kwargs["prompt_tokens"] == 1_000_000
    assert (priced.cost_usd, priced.cost_source) == (None, "none")


@pytest.mark.asyncio
async def test_a_local_backend_is_never_priced_at_cloud_rates(persist):
    ctx = _ctx(route=_sync_route(provider="ollama", payer="local", label="ollama"))
    await record(ctx, _usage(None), block_name="b")

    kwargs = persist.await_args.kwargs
    assert kwargs["cost_usd"] is None
    assert kwargs["provider"] == "ollama"
    assert kwargs["extra_metadata"]["billing_mode"] == "local"


_NO_TOKENS = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_creation_tokens": 0,
}


@pytest.mark.parametrize(
    "kind", ["dream", "briefing_narrative", "consult", "eval_judge"]
)
@pytest.mark.asyncio
async def test_a_call_that_reported_no_usage_writes_no_row(persist, kind):
    """The provider sent no usage (OpenRouter can omit it): no tokens, no
    cost. The catalog would make a known $0 call of it, so nothing is written
    and the cost stays unknown, whatever the job."""
    usage = _usage(None, **_NO_TOKENS)

    recorded = await record(_ctx(kind, None), usage, block_name="b")

    persist.assert_not_awaited()
    assert recorded is usage
    assert (recorded.cost_usd, recorded.cost_source) == (None, "none")


@pytest.mark.asyncio
async def test_cache_only_usage_is_still_recorded_and_priced(persist):
    usage = _usage(None, **{**_NO_TOKENS, "cache_read_tokens": 1_000_000})

    recorded = await record(_ctx(), usage, block_name="b")

    # Sonnet 5 reads its cache at $0.20 per Mtok.
    assert persist.await_args.kwargs["cost_usd"] == pytest.approx(0.2)
    assert recorded.cost_source == "catalog"


@pytest.mark.asyncio
async def test_a_cost_the_provider_stated_is_recorded_even_at_zero(persist):
    """Only unreported usage is skipped: a figure the provider stated, even
    $0 with no tokens, is known and gets its row."""
    await record(_ctx(), _usage(0.0, **_NO_TOKENS), block_name="b")
    assert persist.await_args.kwargs["cost_usd"] == 0.0


def test_price_leaves_a_priced_usage_alone():
    usage = _usage(0.5)
    assert price(usage, _sync_route()) is usage
