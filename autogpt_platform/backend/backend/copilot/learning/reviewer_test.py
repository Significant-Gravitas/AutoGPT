"""The skill reviewer on the inference layer: the review is a skill-learning
job on the account's own scope, pinned to the reviewer model, and its spend is
recorded as the learning ledger wrote it (``copilot:learning:review``, joined
under the run, weekly only), priced from the catalog when the provider
reported no cost, without ever failing the review."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.inference.complete import StructuredCompletion
from backend.copilot.inference.context import InferenceUsage, RouteDecision

from . import reviewer
from .contract import EvidenceBundle
from .prompts import LearningProposal


def _route(model: str = "anthropic/claude-sonnet-5-5") -> RouteDecision:
    return RouteDecision(
        engine="provider_sync",
        auth_provider="platform",
        provider="open_router",
        model=model,
        payer="platform_allowance",
        execution_path="sync_baseline",
        cost_log_provider="open_router",
        reason="test",
    )


def _usage(cost_usd: float | None) -> InferenceUsage:
    return InferenceUsage(
        model="anthropic/claude-sonnet-5-5",
        input_tokens=1_000,
        output_tokens=200,
        cost_usd=cost_usd,
        cost_source="provider" if cost_usd is not None else "none",
        payer="platform_allowance",
    )


@pytest.fixture
def route():
    with patch.object(reviewer, "resolve_route", return_value=_route()) as mock:
        yield mock


@pytest.mark.asyncio
async def test_a_review_is_a_skill_learning_job_on_the_accounts_scope(route):
    proposal = LearningProposal.model_construct()
    completion = StructuredCompletion(value=proposal, usage=_usage(0.002))
    config = ChatConfig(learning_reviewer_model="anthropic/claude-opus-5-5")
    with (
        patch.object(
            reviewer, "structured_complete", AsyncMock(return_value=completion)
        ) as complete,
        patch.object(reviewer, "build_review_messages", return_value=[]),
    ):
        got = await reviewer.review_evidence(
            config, EvidenceBundle.model_construct(), [], None, "user-1", "run-1"
        )

    assert got is completion
    ctx = complete.await_args.args[0]
    assert ctx.scope.user_id == "user-1" and ctx.scope.expert_id is None
    assert ctx.job.kind == "skill_learning"
    assert ctx.job.correlation_id == "run-1"
    assert ctx.job.pinned_model == "anthropic/claude-opus-5-5"
    assert complete.await_args.kwargs["max_output_tokens"] == (
        reviewer.REVIEW_MAX_OUTPUT_TOKENS
    )


@pytest.mark.asyncio
async def test_a_review_without_a_provider_cost_is_priced_and_recorded(route):
    persist = AsyncMock()
    with patch("backend.copilot.inference.record.persist_and_record_usage", persist):
        micro = await reviewer.record_review_cost(
            user_id="user-1", run_id="run-1", usage=_usage(None)
        )

    row = persist.await_args.kwargs
    assert row["user_id"] == "user-1"
    assert row["block_name_override"] == "copilot:learning:review"
    assert row["graph_exec_id_override"] == "run-1"
    assert row["skip_daily"] is True
    assert row["extra_metadata"]["source"] == "skill_learning"
    assert row["extra_metadata"]["learning_run_id"] == "run-1"
    assert row["cost_usd"] is not None and row["cost_usd"] > 0
    assert micro == int(round(row["cost_usd"] * 1_000_000))


@pytest.mark.asyncio
async def test_a_review_that_reported_no_usage_writes_no_row(route):
    """No tokens and no provider cost: the cost is unknown, not a known $0,
    so nothing is charged or logged and the run's ledger keeps no figure."""
    persist = AsyncMock()
    usage = _usage(None).model_copy(update={"input_tokens": 0, "output_tokens": 0})
    with patch("backend.copilot.inference.record.persist_and_record_usage", persist):
        micro = await reviewer.record_review_cost(
            user_id="user-1", run_id="run-1", usage=usage
        )

    persist.assert_not_awaited()
    assert micro is None


@pytest.mark.asyncio
async def test_a_ledger_failure_never_fails_the_review(route):
    persist = AsyncMock(side_effect=RuntimeError("ledger down"))
    with patch("backend.copilot.inference.record.persist_and_record_usage", persist):
        micro = await reviewer.record_review_cost(
            user_id="user-1", run_id="run-1", usage=_usage(0.004)
        )

    assert micro == 4_000
