"""Dream-pass billing tests — pre-flight check + per-phase cost log.

Covers the two billing seams the orchestrator uses:
  * check_dream_budget — paywall, rate-limit cap, Redis brown-out
  * record_phase_cost — provider tag mapping, dream metadata,
    block_name, graph_exec_id correlation, no-op on empty phase
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.rate_limit import RateLimitExceeded, RateLimitUnavailable

from . import billing as billing_mod
from .schemas import PhaseUsage

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


def _routing_kwargs(cost_log_provider: str):
    """Build a ``ProviderRoutingKwargs`` whose ``cost_log_provider``
    is the only field ``_provider_for_execution_path`` reads on the
    sync_baseline path. The other fields are placeholders."""
    from backend.copilot.transport_routing import ProviderRoutingKwargs

    return ProviderRoutingKwargs(
        provider="open_router",
        api_key="",
        base_url=None,
        supports_flex=False,
        cost_log_provider=cost_log_provider,
    )


@pytest.mark.asyncio
async def test_record_phase_cost_tags_sync_baseline_provider_from_transport():
    """sync_baseline ``provider`` follows the active chat transport's
    ``cost_log_provider`` — the cloud OpenRouter case still labels
    rows as ``open_router`` so existing dashboards stay correct."""
    spy = AsyncMock()
    with patch.object(billing_mod, "persist_and_record_usage", new=spy), patch.object(
        billing_mod,
        "routing_kwargs_for_chat_transport",
        return_value=_routing_kwargs("open_router"),
    ):
        await billing_mod.record_phase_cost(
            user_id="u1",
            pass_id="pass-uuid",
            phase_usage=PhaseUsage(
                phase="consolidate",
                model="anthropic/claude-sonnet-4.6",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.0025,
            ),
            execution_path="sync_baseline",
        )
    kwargs = spy.await_args.kwargs
    assert kwargs["provider"] == "open_router"


@pytest.mark.asyncio
async def test_record_phase_cost_tags_sync_baseline_provider_ollama_under_local():
    """Local-transport dream rows must label as ``ollama`` so the
    admin platform-costs dashboard distinguishes them from OR cloud
    spend. Previously hard-coded to ``open_router``."""
    spy = AsyncMock()
    with patch.object(billing_mod, "persist_and_record_usage", new=spy), patch.object(
        billing_mod,
        "routing_kwargs_for_chat_transport",
        return_value=_routing_kwargs("ollama"),
    ):
        await billing_mod.record_phase_cost(
            user_id="u1",
            pass_id="p",
            phase_usage=PhaseUsage(phase="recombine", model="qwen3.5:8b", cost_usd=0.0),
            execution_path="sync_baseline",
        )
    assert spy.await_args.kwargs["provider"] == "ollama"


@pytest.mark.asyncio
async def test_record_phase_cost_tags_sync_baseline_provider_anthropic_under_subscription():
    """Subscription / direct_anthropic transports label sync_baseline
    rows as ``anthropic`` so dream + chat spend roll up together
    under one provider on the admin dashboard."""
    spy = AsyncMock()
    with patch.object(billing_mod, "persist_and_record_usage", new=spy), patch.object(
        billing_mod,
        "routing_kwargs_for_chat_transport",
        return_value=_routing_kwargs("anthropic"),
    ):
        await billing_mod.record_phase_cost(
            user_id="u1",
            pass_id="p",
            phase_usage=PhaseUsage(
                phase="sanitize",
                model="anthropic/claude-sonnet-4-6",
                cost_usd=0.0025,
            ),
            execution_path="sync_baseline",
        )
    assert spy.await_args.kwargs["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_record_phase_cost_tags_provider_anthropic_for_batch_path():
    spy = AsyncMock()
    with patch.object(billing_mod, "persist_and_record_usage", new=spy):
        await billing_mod.record_phase_cost(
            user_id="u1",
            pass_id="p",
            phase_usage=PhaseUsage(
                phase="recombine", model="anthropic/claude-opus-4.7", cost_usd=0.01
            ),
            execution_path="anthropic_batch",
        )
    assert spy.await_args.kwargs["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_record_phase_cost_tags_provider_openai_for_openai_batch_path():
    spy = AsyncMock()
    with patch.object(billing_mod, "persist_and_record_usage", new=spy):
        await billing_mod.record_phase_cost(
            user_id="u1",
            pass_id="p",
            phase_usage=PhaseUsage(phase="sanitize", model="gpt-5", cost_usd=0.005),
            execution_path="openai_batch",
        )
    assert spy.await_args.kwargs["provider"] == "openai"


@pytest.mark.asyncio
async def test_record_phase_cost_uses_dream_block_name_and_passes_pass_id_as_graph_exec_id():
    """Block name must distinguish dream from chat, and pass_id is the
    correlation key on the row so all 3 phases of one pass join up."""
    spy = AsyncMock()
    with patch.object(billing_mod, "persist_and_record_usage", new=spy):
        await billing_mod.record_phase_cost(
            user_id="u1",
            pass_id="pass-uuid-1",
            phase_usage=PhaseUsage(
                phase="sanitize", model="m", input_tokens=10, cost_usd=0.001
            ),
            execution_path="sync_baseline",
        )
    kwargs = spy.await_args.kwargs
    assert kwargs["block_name_override"] == "copilot:dream:sanitize"
    assert kwargs["graph_exec_id_override"] == "pass-uuid-1"


@pytest.mark.asyncio
async def test_record_phase_cost_writes_dream_metadata():
    """Downstream dashboards key off `source` and `dream_pass_id` —
    if those drift the cost rollups silently lose dream rows."""
    spy = AsyncMock()
    with patch.object(billing_mod, "persist_and_record_usage", new=spy):
        await billing_mod.record_phase_cost(
            user_id="u1",
            pass_id="pass-xyz",
            phase_usage=PhaseUsage(
                phase="consolidate", model="m", input_tokens=1, cost_usd=0.001
            ),
            execution_path="anthropic_batch",
        )
    metadata = spy.await_args.kwargs["extra_metadata"]
    assert metadata["source"] == "dream_pass"
    assert metadata["dream_pass_id"] == "pass-xyz"
    assert metadata["dream_phase"] == "consolidate"
    assert metadata["execution_path"] == "anthropic_batch"


@pytest.mark.asyncio
async def test_record_phase_cost_skips_completely_empty_phase():
    """Zero tokens AND no cost → no billing row, no rate-limit charge.
    Avoids logging junk rows when a phase short-circuited (e.g. no input
    to consolidate)."""
    spy = AsyncMock()
    with patch.object(billing_mod, "persist_and_record_usage", new=spy):
        await billing_mod.record_phase_cost(
            user_id="u1",
            pass_id="p",
            phase_usage=PhaseUsage(phase="consolidate", model="m"),
            execution_path="sync_baseline",
        )
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_record_phase_cost_logs_token_only_phase_even_without_cost():
    """Phase ran (real tokens) but provider didn't report cost — still
    log the row for analytics even though rate-limit won't charge."""
    spy = AsyncMock()
    with patch.object(billing_mod, "persist_and_record_usage", new=spy):
        await billing_mod.record_phase_cost(
            user_id="u1",
            pass_id="p",
            phase_usage=PhaseUsage(
                phase="recombine", model="m", input_tokens=10, output_tokens=5
            ),
            execution_path="sync_baseline",
        )
    spy.assert_called_once()
