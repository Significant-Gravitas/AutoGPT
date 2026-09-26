"""Dream-pass billing tests — pre-flight check + per-phase cost row.

Covers the billing seams the orchestrator and the batch callbacks use:
  * check_dream_budget — paywall, rate-limit cap, Redis brown-out
  * record_phase_cost — the dream's row shape through ``inference.record``:
    provider label from the route, dream block name and metadata,
    graph_exec_id correlation, expert attribution, catalog pricing at the
    path's discount, no row for an empty phase
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.inference.context import (
    InferenceContext,
    InferenceScope,
    InferenceUsage,
    RouteDecision,
)
from backend.copilot.inference.routing import anthropic_batch_route
from backend.copilot.rate_limit import RateLimitExceeded, RateLimitUnavailable

from . import billing as billing_mod
from .phase_jobs import phase_job

# ---------------------------------------------------------------------------
# check_dream_budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_dream_budget_allows_under_cap_user():
    with patch.object(
        billing_mod, "is_user_paywalled", new=AsyncMock(return_value=False)
    ), patch.object(
        billing_mod,
        "get_global_rate_limits",
        new=AsyncMock(return_value=(1_000_000, 5_000_000, None)),
    ), patch.object(
        billing_mod, "check_rate_limit", new=AsyncMock(return_value=None)
    ):
        ok, reason = await billing_mod.check_dream_budget("u1")
    assert (ok, reason) == (True, None)


@pytest.mark.asyncio
async def test_check_dream_budget_skips_paywalled_user_as_insufficient_credits():
    with patch.object(
        billing_mod, "is_user_paywalled", new=AsyncMock(return_value=True)
    ):
        ok, reason = await billing_mod.check_dream_budget("u1")
    assert (ok, reason) == (False, "insufficient_credits")


@pytest.mark.asyncio
async def test_check_dream_budget_skips_when_user_is_over_daily_cap():
    reset_at = datetime(2026, 5, 22, tzinfo=timezone.utc)
    with patch.object(
        billing_mod, "is_user_paywalled", new=AsyncMock(return_value=False)
    ), patch.object(
        billing_mod,
        "get_global_rate_limits",
        new=AsyncMock(return_value=(0, 0, None)),
    ), patch.object(
        billing_mod,
        "check_rate_limit",
        new=AsyncMock(side_effect=RateLimitExceeded("daily", reset_at)),
    ):
        ok, reason = await billing_mod.check_dream_budget("u1")
    assert (ok, reason) == (False, "insufficient_credits")


@pytest.mark.asyncio
async def test_check_dream_budget_fails_closed_on_redis_brownout():
    """Redis unreadable → orchestrator must NOT bill on it; fail closed."""
    with patch.object(
        billing_mod, "is_user_paywalled", new=AsyncMock(return_value=False)
    ), patch.object(
        billing_mod,
        "get_global_rate_limits",
        new=AsyncMock(return_value=(1_000_000, 5_000_000, None)),
    ), patch.object(
        billing_mod,
        "check_rate_limit",
        new=AsyncMock(side_effect=RateLimitUnavailable()),
    ):
        ok, reason = await billing_mod.check_dream_budget("u1")
    assert (ok, reason) == (False, "rate_limit_unavailable")


@pytest.mark.asyncio
async def test_check_dream_budget_fails_closed_when_paywall_lookup_raises():
    """Background callers can't rely on enforce_payment_paywall's HTTP-503;
    we treat lookup failure as 'cannot prove eligibility' → skip."""
    with patch.object(
        billing_mod,
        "is_user_paywalled",
        new=AsyncMock(side_effect=Exception("supabase blip")),
    ):
        ok, reason = await billing_mod.check_dream_budget("u1")
    assert (ok, reason) == (False, "rate_limit_unavailable")


# ---------------------------------------------------------------------------
# record_phase_cost
# ---------------------------------------------------------------------------

_PERSIST = "backend.copilot.inference.record.persist_and_record_usage"


def _sync_route(label: str = "open_router") -> RouteDecision:
    provider = {"open_router": "open_router", "ollama": "ollama"}.get(
        label, "anthropic"
    )
    return RouteDecision(
        engine="provider_sync",
        auth_provider="platform",
        provider=provider,
        model="anthropic/claude-sonnet-5",
        payer="local" if label == "ollama" else "platform_allowance",
        execution_path="sync_baseline",
        cost_log_provider=label,
        reason="test",
    )


def _ctx(
    phase="consolidate",
    *,
    pass_id="pass-uuid",
    route: RouteDecision | None = None,
    expert_id: str | None = None,
) -> InferenceContext:
    return InferenceContext(
        scope=InferenceScope(user_id="u1", expert_id=expert_id),
        job=phase_job(phase, pass_id, timeout_seconds=240),
        route=route or _sync_route(),
    )


def _usage(model="anthropic/claude-sonnet-5", cost_usd=None, **tokens):
    return InferenceUsage(
        model=model,
        cost_usd=cost_usd,
        cost_source="none" if cost_usd is None else "provider",
        payer="platform_allowance",
        **tokens,
    )


@pytest.mark.parametrize("label", ["open_router", "ollama", "anthropic"])
@pytest.mark.asyncio
async def test_record_phase_cost_labels_the_row_with_the_routes_provider(label):
    """The sync path follows the chat transport (resolved by the route), so a
    local install logs ``ollama``, subscription or direct Anthropic logs
    ``anthropic`` and the cloud default logs ``open_router``."""
    spy = AsyncMock()
    with patch(_PERSIST, new=spy):
        await billing_mod.record_phase_cost(
            _ctx(route=_sync_route(label)), _usage(input_tokens=10, cost_usd=0.001)
        )
    assert spy.await_args.kwargs["provider"] == label


@pytest.mark.asyncio
async def test_record_phase_cost_labels_the_batch_path_anthropic():
    spy = AsyncMock()
    with patch(_PERSIST, new=spy):
        await billing_mod.record_phase_cost(
            _ctx("recombine", route=anthropic_batch_route("claude-opus-5-5")),
            _usage("claude-opus-5-5", input_tokens=10),
        )
    assert spy.await_args.kwargs["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_record_phase_cost_uses_dream_block_name_and_passes_pass_id_as_graph_exec_id():
    """Block name must distinguish dream from chat, and pass_id is the
    correlation key on the row so all 3 phases of one pass join up."""
    spy = AsyncMock()
    with patch(_PERSIST, new=spy):
        await billing_mod.record_phase_cost(
            _ctx("sanitize", pass_id="pass-uuid-1"), _usage(input_tokens=10)
        )
    kwargs = spy.await_args.kwargs
    assert kwargs["block_name_override"] == "copilot:dream:sanitize"
    assert kwargs["graph_exec_id_override"] == "pass-uuid-1"
    assert kwargs["chat_session_id_override"] is None
    # Background work: the weekly cap, never the interactive daily budget.
    assert kwargs["skip_daily"] is True


@pytest.mark.parametrize(
    "route,path,discount",
    [
        (_sync_route(), "sync_baseline", 0.0),
        (anthropic_batch_route("claude-sonnet-5"), "anthropic_batch", 0.5),
    ],
)
@pytest.mark.asyncio
async def test_record_phase_cost_writes_dream_metadata(route, path, discount):
    """Downstream dashboards key off `source` and `dream_pass_id` —
    if those drift the cost rollups silently lose dream rows."""
    spy = AsyncMock()
    with patch(_PERSIST, new=spy):
        await billing_mod.record_phase_cost(
            _ctx(pass_id="pass-xyz", route=route, expert_id="e1"),
            _usage(input_tokens=1, cost_usd=0.001),
        )
    assert spy.await_args.kwargs["extra_metadata"] == {
        "source": "dream_pass",
        "dream_pass_id": "pass-xyz",
        "dream_phase": "consolidate",
        "execution_path": path,
        "discount_applied": discount,
        "job_kind": "dream",
        "phase": "consolidate",
        "billing_mode": "platform",
        "expert_id": "e1",
    }
    assert spy.await_args.kwargs["expert_id"] == "e1"


# Sonnet 5's catalog card: $2 in, $10 out, $0.20 cache read, $2.50 cache
# write per Mtok, every bucket billed on its own.
_SONNET_5_LIST_COST = 2.0 + 1.0 + 0.04 + 0.025
_SONNET_5_TOKENS = {
    "input_tokens": 1_000_000,
    "output_tokens": 100_000,
    "cache_read_tokens": 200_000,
    "cache_creation_tokens": 10_000,
}


@pytest.mark.parametrize(
    "route,cost",
    [
        (_sync_route("anthropic"), _SONNET_5_LIST_COST),
        (anthropic_batch_route("claude-sonnet-5"), _SONNET_5_LIST_COST / 2),
    ],
)
@pytest.mark.asyncio
async def test_record_phase_cost_prices_an_unpriced_phase_from_the_catalog(route, cost):
    """Native Anthropic reports no cost: the phase is charged at its model's
    list price, less the batch discount, and the priced usage comes back for
    the pass's aggregate."""
    spy = AsyncMock()
    with patch(_PERSIST, new=spy):
        priced = await billing_mod.record_phase_cost(
            _ctx(route=route), _usage("claude-sonnet-5", **_SONNET_5_TOKENS)
        )
    assert spy.await_args.kwargs["cost_usd"] == pytest.approx(cost)
    assert priced.cost_usd == pytest.approx(cost)
    assert priced.cost_source == "catalog"


@pytest.mark.asyncio
async def test_record_phase_cost_leaves_an_unpriced_model_unknown():
    """No catalog price means an unknown cost, never a zero one: the row
    logs tokens and charges nothing."""
    spy = AsyncMock()
    with patch(_PERSIST, new=spy):
        priced = await billing_mod.record_phase_cost(
            _ctx("recombine"), _usage("vendor/no-such-model", input_tokens=10)
        )
    assert spy.await_args.kwargs["cost_usd"] is None
    assert spy.await_args.kwargs["prompt_tokens"] == 10
    assert priced.cost_usd is None


@pytest.mark.parametrize(
    "usage,rows",
    [
        # Zero tokens and no cost (a skipped phase, or a response that carried
        # no usage): no row and no charge, even for a model the catalog
        # prices, which would otherwise log a $0 call.
        (_usage("claude-sonnet-5"), 0),
        # Tokens but no known cost: the row logs for analytics, and the
        # rate-limit counter is left alone.
        (_usage("vendor/no-such-model", input_tokens=10, output_tokens=5), 1),
    ],
)
@pytest.mark.asyncio
async def test_record_phase_cost_logs_a_row_only_for_a_phase_that_ran(usage, rows):
    log = MagicMock()
    charge = AsyncMock()
    with patch("backend.copilot.token_tracking._schedule_cost_log", new=log), patch(
        "backend.copilot.token_tracking.record_cost_usage", new=charge
    ):
        await billing_mod.record_phase_cost(_ctx(), usage)
        await asyncio.sleep(0)
    assert log.call_count == rows
    charge.assert_not_called()
