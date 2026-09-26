import pytest
from pydantic import ValidationError

from .context import (
    InferenceContext,
    InferenceError,
    InferenceJob,
    InferenceScope,
    InferenceUsage,
    RouteDecision,
)


def _job(**overrides) -> InferenceJob:
    return InferenceJob(
        **{
            "kind": "dream",
            "phase": "consolidate",
            "correlation_id": "pass-1",
            "latency_class": "deferred",
            "tier": "standard",
            **overrides,
        }
    )


def _route() -> RouteDecision:
    return RouteDecision(
        engine="provider_sync",
        auth_provider="platform",
        provider="open_router",
        model="anthropic/claude-sonnet-5",
        payer="platform_allowance",
        execution_path="sync_baseline",
        cost_log_provider="open_router",
        reason="test",
    )


def test_scope_defaults_to_the_account_and_refuses_empty_ids():
    assert InferenceScope(user_id="u1").expert_id is None
    with pytest.raises(ValidationError):
        InferenceScope(user_id="")
    with pytest.raises(ValidationError):
        InferenceScope(user_id="u1", expert_id="")


def test_job_label_names_kind_and_phase():
    assert _job().label == "dream:consolidate"
    assert _job(kind="consult", phase=None).label == "consult"


def test_job_needs_a_correlation_id():
    with pytest.raises(ValidationError):
        _job(correlation_id="")


def test_models_are_frozen():
    ctx = InferenceContext(
        scope=InferenceScope(user_id="u1"), job=_job(), route=_route()
    )
    with pytest.raises(ValidationError):
        ctx.trace_id = "t"


def test_route_keeps_the_platform_credential_implicit():
    assert _route().credential_id is None


@pytest.mark.parametrize(
    "cost_usd,cost_source",
    [(0.01, "none"), (None, "provider"), (None, "catalog")],
)
def test_usage_cost_and_its_source_must_agree(cost_usd, cost_source):
    """An unknown cost says so, and a known one says where it came from."""
    with pytest.raises(ValidationError):
        InferenceUsage(
            model="m",
            cost_usd=cost_usd,
            cost_source=cost_source,
            payer="platform_allowance",
        )


def test_usage_with_a_zero_cost_is_known_not_unknown():
    usage = InferenceUsage(
        model="m", cost_usd=0.0, cost_source="provider", payer="local"
    )
    assert (usage.cost_usd, usage.cost_source) == (0.0, "provider")


def test_error_carries_the_billed_usage():
    usage = InferenceUsage(model="m", input_tokens=5, payer="platform_allowance")
    assert InferenceError("bad json", usage).usage == usage
    assert InferenceError("no key").usage is None
