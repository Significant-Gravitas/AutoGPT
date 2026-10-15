"""The style judge runs through the inference seam on its pinned model."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.inference.complete import StructuredCompletion
from backend.copilot.inference.context import InferenceUsage, RouteDecision

from .assembly import load_rubric, roster_experts
from .models import DimensionJudgement, Judgement
from .scorer import EVAL_USER_ID, JUDGE_TIMEOUT_SECONDS, judge_response

_SCORER = "backend.copilot.eval.style.scorer"


def _route(scope, job, *, config=None) -> RouteDecision:
    return RouteDecision(
        engine="provider_sync",
        auth_provider="platform",
        provider="anthropic",
        model=job.pinned_model,
        payer="platform_allowance",
        execution_path="sync_baseline",
        cost_log_provider="anthropic",
        reason="test",
    )


@pytest.mark.asyncio
async def test_the_judge_runs_on_its_pinned_model_for_the_judged_expert():
    rubric = load_rubric()
    (expert,) = roster_experts(["Max"])
    judgement = Judgement(
        scores={
            d.key: DimensionJudgement(score=rubric.scale_max) for d in rubric.dimensions
        }
    )
    usage = InferenceUsage(
        model="claude-haiku-4-5",
        input_tokens=100,
        output_tokens=20,
        payer="platform_allowance",
    )
    complete = AsyncMock(
        return_value=StructuredCompletion(value=judgement, usage=usage)
    )
    with patch(f"{_SCORER}.resolve_route", side_effect=_route), patch(
        f"{_SCORER}.structured_complete", complete
    ):
        result, judged_usage = await judge_response(
            expert,
            rubric,
            kind="email",
            prompt="p",
            response="r",
            model="claude-haiku-4-5",
            run_id="run-1",
        )

    assert result == judgement
    ctx, _messages, response_model = complete.await_args.args
    assert response_model is Judgement
    assert complete.await_args.kwargs["temperature"] == 0.0
    assert (ctx.scope.user_id, ctx.scope.expert_id) == (EVAL_USER_ID, expert.id)
    assert (ctx.job.kind, ctx.job.correlation_id) == ("eval_judge", "run-1")
    assert ctx.job.timeout_seconds == JUDGE_TIMEOUT_SECONDS
    assert ctx.route.model == "claude-haiku-4-5"
    # The eval prices its own spend from the rate card; nothing is recorded.
    assert judged_usage.cost_usd is not None
