"""Reviewer model call and cost accounting for one learning review.

Goes through the inference layer every background call takes
(``copilot/inference``): the chat transport's provider and the platform's
key, on the reviewer's model, the call traced, and its spend recorded in the
usage ledger so learning appears next to chat spend under one provider label.
"""

from __future__ import annotations

import logging

from backend.copilot.config import ChatConfig
from backend.copilot.inference.complete import StructuredCompletion, structured_complete
from backend.copilot.inference.context import (
    InferenceContext,
    InferenceError,
    InferenceJob,
    InferenceScope,
    InferenceUsage,
)
from backend.copilot.inference.record import price, record
from backend.copilot.inference.routing import resolve_route
from backend.copilot.inference.trace import trace
from backend.copilot.tools.skills import ParsedSkill

from .contract import EvidenceBundle
from .packages import ReviewedPackage
from .prompts import LearningProposal, build_review_messages

logger = logging.getLogger(__name__)

REVIEW_TEMPERATURE = 0.1
REVIEW_MAX_OUTPUT_TOKENS = 12_288
REVIEW_TIMEOUT_SECONDS = 240

# The learning pass's name for a review that failed: the inference layer's
# error, carrying the usage a provider billed for an answer that did not parse.
DreamLLMError = InferenceError

__all__ = ["DreamLLMError", "review_evidence", "record_review_cost", "reviewer_model"]


def reviewer_model(config: ChatConfig) -> str:
    return config.learning_reviewer_model.strip() or config.fast_standard_model


async def review_evidence(
    config: ChatConfig,
    bundle: EvidenceBundle,
    existing_skills: list[ParsedSkill],
    packages: dict[str, ReviewedPackage] | None,
    user_id: str,
    run_id: str,
) -> StructuredCompletion[LearningProposal]:
    """One reviewer call for *user_id*'s learning run *run_id*. Raises
    ``DreamLLMError`` on provider/parse failure; its spend is the caller's to
    record (``record_review_cost``)."""
    ctx = _review_context(user_id, run_id, reviewer_model(config), config)
    async with trace(ctx) as call:
        completion = await structured_complete(
            call.ctx,
            build_review_messages(bundle, existing_skills, packages),
            LearningProposal,
            temperature=REVIEW_TEMPERATURE,
            max_output_tokens=REVIEW_MAX_OUTPUT_TOKENS,
        )
        call.usage = completion.usage
    return completion


async def record_review_cost(
    *, user_id: str, run_id: str, usage: InferenceUsage
) -> int | None:
    """Charge one review against the user's weekly window; returns microdollars.

    Never raises — an accounting hiccup must not fail the review whose tokens
    were already paid for; the ledger row still records the token counts.
    """
    priced = usage
    try:
        ctx = _review_context(user_id, run_id, usage.model, ChatConfig())
        priced = price(usage, ctx.route)
        await record(
            ctx,
            priced,
            block_name="copilot:learning:review",
            metadata={"learning_run_id": run_id},
        )
    except Exception:
        logger.warning(
            "Skill learning cost accounting failed for user %s run %s",
            user_id[:12],
            run_id[:12],
            exc_info=True,
        )
    cost = priced.cost_usd
    return int(round(cost * 1_000_000)) if cost is not None else None


def _review_context(
    user_id: str, run_id: str, model: str, config: ChatConfig
) -> InferenceContext:
    """A review on the account's own scope, pinned to *model*, its spend
    joined under the learning run."""
    scope = InferenceScope(user_id=user_id)
    job = InferenceJob(
        kind="skill_learning",
        correlation_id=run_id,
        latency_class="deferred",
        tier="standard",
        timeout_seconds=REVIEW_TIMEOUT_SECONDS,
        pinned_model=model,
    )
    return InferenceContext(
        scope=scope, job=job, route=resolve_route(scope, job, config=config)
    )
