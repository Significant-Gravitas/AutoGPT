"""Ledger dispositions for one nightly pass.

Every source revision the pass touches ends in exactly one disposition
row; only durable dispositions advance the source cursor, so a provider
error, a paused Expert, or a clipped evidence bundle is retried on a later
night instead of being forgotten.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from backend.copilot.config import ChatConfig
from backend.data.db_accessors import skill_learning_db, skill_reviews_db
from backend.data.skill_publication import ReviewStamp

from .content_checks import safe_diagnostic
from .contract import EligibilityState, SourceRevision
from .fingerprint import behavior_fingerprint
from .memory_link import link_version_to_memory
from .publish import PublishOutcome, PublishRequest
from .reviewer import DreamLLMError, record_review_cost, reviewer_model

logger = logging.getLogger(__name__)

POLICY_VERSION = 1

# Dispositions that settle a revision: the cursor moves past it.
_TERMINAL_DISPOSITIONS = frozenset(
    {
        "applied",
        "skipped",
        "no_novel_procedure",
        "blocked_content",
        "suppressed",
        "needs_decision",
        "stale_eligibility",
        "inaccessible_evidence",
    }
)


class SkillLearningResult(BaseModel):
    user_id: str
    run_id: str
    trigger: str
    started_at: datetime
    completed_at: datetime | None = None
    elapsed_seconds: float | None = None
    reviewed: int = 0
    model_calls: int = 0
    applied: int = 0
    proposed: int = 0
    dispositions: dict[str, int] = Field(default_factory=dict)
    reconciled_writes: int = 0
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None
    per_source_errors: list[str] = Field(default_factory=list)


async def already_settled(
    user_id: str, source: SourceRevision, result: SkillLearningResult
) -> str | None:
    """Consult the ledger before paying for a review.

    A completed row for this exact revision means an earlier run finished
    the work but crashed before moving the cursor: the cursor is advanced
    and nothing is re-reviewed or overwritten. A row still ``applied_pending``
    belongs to ``reconcile_pending``, which ran at the start of this pass.
    """
    review = await skill_reviews_db().get_review_for_revision(
        user_id, source.source_id, source.revision, POLICY_VERSION
    )
    if review is None or review.disposition not in _TERMINAL_DISPOSITIONS:
        return None
    await skill_learning_db().advance_source_cursor(
        user_id, source.source_id, source.revision
    )
    result.reviewed += 1
    return review.disposition


async def settle_ineligible(
    user_id: str,
    source: SourceRevision,
    result: SkillLearningResult,
    state: EligibilityState,
    reason: str,
) -> str:
    if state in (EligibilityState.WAITING_APPROVAL, EligibilityState.PAUSED):
        return await record(user_id, source, result, "deferred", reason, advance=False)
    if state == EligibilityState.STALE:
        return await record(user_id, source, result, "deferred", reason, advance=False)
    if state == EligibilityState.INACCESSIBLE:
        return await settle(user_id, source, result, "inaccessible_evidence", reason)
    return await settle(user_id, source, result, "stale_eligibility", reason)


async def settle_outcome(
    user_id: str,
    source: SourceRevision,
    result: SkillLearningResult,
    outcome: PublishOutcome,
    stamp: ReviewStamp,
    request: PublishRequest,
) -> str:
    fingerprint = behavior_fingerprint(request.skill_name, request.body)
    reason = safe_diagnostic(outcome.reason)
    # Rejected metadata is never persisted: a blocked outcome records only
    # the safe diagnostic, not the model-written skill name.
    skill_name = None if outcome.status == "blocked_content" else request.skill_name
    if outcome.status == "applied":
        result.applied += 1
        if outcome.version is not None:
            await link_version_to_memory(user_id, outcome.version)
        # The ledger row was stamped inside the publication transaction and
        # completed with the workspace write; only the cursor remains.
        await skill_learning_db().advance_source_cursor(
            user_id, source.source_id, source.revision
        )
        return "applied"
    if outcome.status == "needs_decision":
        result.proposed += 1
    if outcome.status in ("conflict", "write_failed", "paused"):
        return await record(
            user_id,
            source,
            result,
            outcome.status,
            reason,
            advance=False,
            stamp=stamp,
            fingerprint=fingerprint,
            skill_name=skill_name,
            version_id=outcome.version.id if outcome.version else None,
        )
    return await settle(
        user_id,
        source,
        result,
        outcome.status,
        reason,
        stamp=stamp,
        fingerprint=fingerprint,
        skill_name=skill_name,
        version_id=outcome.version.id if outcome.version else None,
    )


async def record_provider_error(
    user_id: str,
    source: SourceRevision,
    result: SkillLearningResult,
    exc: DreamLLMError,
    config: ChatConfig,
) -> str:
    cost = None
    stamp = None
    if exc.usage is not None:
        result.model_calls += 1
        cost = await record_review_cost(
            user_id=user_id, run_id=result.run_id, usage=exc.usage
        )
        stamp = ReviewStamp(
            source=await skill_learning_db().require_owned_source(
                user_id, source.source_id
            ),
            source_revision=source.revision,
            policy_version=POLICY_VERSION,
            run_id=result.run_id,
            model=exc.usage.model,
            input_tokens=exc.usage.input_tokens,
            output_tokens=exc.usage.output_tokens,
            cost_microdollars=cost,
        )
    diagnostic = safe_diagnostic(str(exc))
    result.per_source_errors.append(f"{source.source_id[:12]}: {diagnostic[:120]}")
    return await record(
        user_id,
        source,
        result,
        "provider_error",
        diagnostic,
        advance=False,
        stamp=stamp,
        model=reviewer_model(config),
    )


async def settle(
    user_id: str,
    source: SourceRevision,
    result: SkillLearningResult,
    disposition: str,
    reason: str,
    *,
    stamp: ReviewStamp | None = None,
    fingerprint: str | None = None,
    skill_name: str | None = None,
    version_id: str | None = None,
) -> str:
    """A durable disposition: ledger row first, then the cursor moves."""
    return await record(
        user_id,
        source,
        result,
        disposition,
        reason,
        advance=True,
        stamp=stamp,
        fingerprint=fingerprint,
        skill_name=skill_name,
        version_id=version_id,
    )


async def record(
    user_id: str,
    source: SourceRevision,
    result: SkillLearningResult,
    disposition: str,
    reason: str,
    *,
    advance: bool,
    stamp: ReviewStamp | None = None,
    fingerprint: str | None = None,
    skill_name: str | None = None,
    version_id: str | None = None,
    model: str | None = None,
) -> str:
    result.reviewed += 1
    await skill_reviews_db().upsert_review(
        user_id,
        source_id=source.source_id,
        source_revision=source.revision,
        policy_version=POLICY_VERSION,
        run_id=result.run_id,
        disposition=disposition,
        reason=reason,
        change_fingerprint=fingerprint,
        skill_name=skill_name,
        applied_version_id=version_id,
        model=stamp.model if stamp else model,
        input_tokens=stamp.input_tokens if stamp else 0,
        output_tokens=stamp.output_tokens if stamp else 0,
        cost_microdollars=stamp.cost_microdollars if stamp else None,
        completed=advance or disposition in _TERMINAL_DISPOSITIONS,
    )
    if advance:
        await skill_learning_db().advance_source_cursor(
            user_id, source.source_id, source.revision
        )
    return disposition


def count(result: SkillLearningResult, disposition: str) -> None:
    result.dispositions[disposition] = result.dispositions.get(disposition, 0) + 1


def finish(
    result: SkillLearningResult, *, skip_reason: str | None = None
) -> SkillLearningResult:
    if skip_reason is not None:
        result.skipped = True
        result.skip_reason = skip_reason
    result.completed_at = datetime.now(timezone.utc)
    result.elapsed_seconds = (result.completed_at - result.started_at).total_seconds()
    logger.info(
        "Skill learning %s for user %s (%s): reviewed=%d applied=%d proposed=%d "
        "model_calls=%d skipped=%s error=%s",
        result.run_id[:12],
        result.user_id[:12],
        result.trigger,
        result.reviewed,
        result.applied,
        result.proposed,
        result.model_calls,
        result.skip_reason,
        result.error,
    )
    return result
