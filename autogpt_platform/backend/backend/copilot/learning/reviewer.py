"""Reviewer model call and cost accounting for one learning review.

Routes through the same transport + JSON-mode wrapper the dream pass uses
(``dream/llm.structured_completion``) so every transport the platform
supports works unchanged, and charges the spend through the normal usage
ledger so learning appears next to chat spend under one provider label.
"""

from __future__ import annotations

import logging

from backend.copilot.config import ChatConfig
from backend.copilot.dream.llm import (
    CompletionUsage,
    DreamLLMError,
    StructuredCompletion,
    structured_completion,
)
from backend.copilot.dream.model_pricing import compute_cost_usd
from backend.copilot.token_tracking import persist_and_record_usage
from backend.copilot.tools.skills import ParsedSkill
from backend.copilot.transport_routing import routing_kwargs_for_chat_transport

from .contract import EvidenceBundle
from .prompts import LearningProposal, build_review_messages

logger = logging.getLogger(__name__)

REVIEW_TEMPERATURE = 0.1
REVIEW_MAX_OUTPUT_TOKENS = 6144
REVIEW_TIMEOUT_SECONDS = 240

__all__ = ["DreamLLMError", "review_evidence", "record_review_cost", "reviewer_model"]


def reviewer_model(config: ChatConfig) -> str:
    return config.learning_reviewer_model.strip() or config.fast_standard_model


async def review_evidence(
    config: ChatConfig,
    bundle: EvidenceBundle,
    existing_skills: list[ParsedSkill],
) -> StructuredCompletion[LearningProposal]:
    """One reviewer call. Raises ``DreamLLMError`` on provider/parse failure."""
    return await structured_completion(
        model=reviewer_model(config),
        messages=build_review_messages(bundle, existing_skills),
        response_model=LearningProposal,
        temperature=REVIEW_TEMPERATURE,
        max_output_tokens=REVIEW_MAX_OUTPUT_TOKENS,
        timeout_seconds=REVIEW_TIMEOUT_SECONDS,
    )


async def record_review_cost(
    *, user_id: str, run_id: str, usage: CompletionUsage
) -> int | None:
    """Charge one review against the user's weekly window; returns microdollars.

    Never raises — an accounting hiccup must not fail the review whose tokens
    were already paid for; the ledger row still records the token counts.
    """
    cost = usage.cost_usd
    if cost is None:
        cost = compute_cost_usd(
            model=usage.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
        )
    try:
        await persist_and_record_usage(
            session=None,
            user_id=user_id,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            log_prefix="[learning:review]",
            cost_usd=cost,
            model=usage.model,
            provider=routing_kwargs_for_chat_transport().cost_log_provider,
            block_name_override="copilot:learning:review",
            graph_exec_id_override=run_id,
            extra_metadata={"source": "skill_learning", "learning_run_id": run_id},
            skip_daily=True,
        )
    except Exception:
        logger.warning(
            "Skill learning cost accounting failed for user %s run %s",
            user_id[:12],
            run_id[:12],
            exc_info=True,
        )
    return int(round(cost * 1_000_000)) if cost is not None else None
