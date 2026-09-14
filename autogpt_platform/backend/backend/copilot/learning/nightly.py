"""The nightly skill-learning pass (also run immediately for "learn this").

For one user: take a per-user lease, finish any publication interrupted
by a crash, select a bounded, fair set of eligible source revisions, and
review each one:

  revalidate → gate on concrete outcome signals → gather bounded evidence
  → reviewer model → deterministic validation → publish (or propose)
  → durable disposition → advance the source cursor.

Idle accounts cost nothing: when no source has an unreviewed revision the
pass returns before any model call. Provider errors, stale approval, no
novel procedure, exhausted budget, and inaccessible evidence are distinct
dispositions in the ledger, and only durable ones move the cursor.
"""

from __future__ import annotations

import logging
import uuid as uuidlib
from datetime import datetime, timezone
from typing import Literal

from backend.copilot.config import ChatConfig
from backend.copilot.dream.billing import check_dream_budget
from backend.copilot.tools.skills import (
    ParsedSkill,
    list_user_skills,
    read_user_skill_with_body,
)
from backend.data.db_accessors import skill_learning_db, skill_versions_db
from backend.data.redis_client import get_redis_async
from backend.data.skill_learning import LearningSourceRecord
from backend.data.skill_publication import ReviewStamp
from backend.executor.cluster_lock import AsyncClusterLock
from backend.util.feature_flag import Flag, is_feature_enabled

from .chat_source import source_revision_from_record
from .content_checks import safe_diagnostic
from .contract import check_approval_precondition, get_source_adapter
from .dispositions import (
    POLICY_VERSION,
    SkillLearningResult,
    already_settled,
    count,
    finish,
    record,
    record_provider_error,
    settle,
    settle_ineligible,
    settle_outcome,
)
from .proposal import build_publish_request, validate_proposal
from .publish import publish_learned_version, reconcile_pending
from .reviewer import DreamLLMError, record_review_cost, review_evidence

logger = logging.getLogger(__name__)

MAX_SOURCES_PER_RUN = 20
MAX_CHANGES_PER_RUN = 5
MAX_REVIEWS_PER_OWNER = 8
MAX_OPEN_PROPOSALS_PER_OWNER = 10
MAX_EXISTING_SKILL_BODIES = 12
MAX_EVIDENCE_CHARS = 24_000
LEASE_KEY_PREFIX = "copilot:skill_learning:lease:"
LEASE_TTL_SECONDS = 1800


Trigger = Literal["nightly", "requested", "admin"]


async def run_skill_learning_pass(
    user_id: str,
    *,
    source_ids: list[str] | None = None,
    trigger: Trigger = "nightly",
    config: ChatConfig | None = None,
) -> SkillLearningResult:
    """Never raises; every failure lands in ``SkillLearningResult``."""
    result = SkillLearningResult(
        user_id=user_id,
        run_id=str(uuidlib.uuid4()),
        trigger=trigger,
        started_at=datetime.now(timezone.utc),
    )
    try:
        if not await is_feature_enabled(Flag.DREAM_SKILL_LEARNING_ENABLED, user_id):
            return finish(result, skip_reason="skill_learning_disabled")
        lock = await _acquire_lease(user_id)
        if lock is None:
            return finish(result, skip_reason="lease_held")
        try:
            await _run_under_lease(user_id, result, source_ids, config or ChatConfig())
        finally:
            await _release_lease(lock)
    except Exception as exc:
        logger.exception("Skill learning pass crashed for user %s", user_id[:12])
        result.error = f"{type(exc).__name__}: {exc}"
    return finish(result)


async def _run_under_lease(
    user_id: str,
    result: SkillLearningResult,
    source_ids: list[str] | None,
    config: ChatConfig,
) -> None:
    result.reconciled_writes = await reconcile_pending(user_id)
    pending = await _select_sources(user_id, source_ids)
    if not pending:
        result.skipped = True
        result.skip_reason = "no_eligible_work"
        return
    budget_ok, budget_skip = await check_dream_budget(user_id, config=config)
    if not budget_ok:
        if budget_skip == "rate_limit_unavailable":
            result.error = f"billing: {budget_skip}"
        else:
            result.skipped = True
            result.skip_reason = budget_skip or "insufficient_credits"
        return
    budget_exhausted = False
    for stored in pending:
        if result.applied + result.proposed >= MAX_CHANGES_PER_RUN:
            count(result, "deferred")
            continue
        if budget_exhausted:
            count(result, "budget_exhausted")
            continue
        try:
            disposition = await _process_source(user_id, stored, result, config)
            budget_exhausted = disposition == "budget_exhausted"
        except Exception as exc:
            logger.warning(
                "Skill learning source %s failed for user %s",
                stored.id[:12],
                user_id[:12],
                exc_info=True,
            )
            result.per_source_errors.append(f"{stored.id[:12]}: {type(exc).__name__}")
            disposition = "provider_error"
        count(result, disposition)


async def _select_sources(
    user_id: str, source_ids: list[str] | None
) -> list[LearningSourceRecord]:
    """Bounded and fair: a per-owner page first, then round-robin.

    Each owner (personal scope or expert) contributes at most
    ``MAX_REVIEWS_PER_OWNER`` of its oldest pending sources before the
    global cap applies, so one expert's large backlog cannot hide another
    expert's fresh work for many nights.
    """
    learning = skill_learning_db()
    if source_ids:
        found = [await learning.get_source(user_id, sid) for sid in source_ids]
        return [s for s in found if s is not None and s.has_unprocessed_revision]
    queues: dict[str, list[LearningSourceRecord]] = {}
    for owner_key in await learning.list_pending_owner_keys(user_id):
        queues[owner_key] = await learning.list_pending_sources(
            user_id, limit=MAX_REVIEWS_PER_OWNER, owner_key=owner_key
        )
    selected: list[LearningSourceRecord] = []
    while len(selected) < MAX_SOURCES_PER_RUN and any(queues.values()):
        for queue in queues.values():
            if queue:
                selected.append(queue.pop(0))
                if len(selected) >= MAX_SOURCES_PER_RUN:
                    break
    return selected


async def _process_source(
    user_id: str,
    stored: LearningSourceRecord,
    result: SkillLearningResult,
    config: ChatConfig,
) -> str:
    source = source_revision_from_record(stored)
    adapter = get_source_adapter(source.source_kind)
    if adapter is None:
        return await settle(user_id, source, result, "skipped", "unknown source kind")

    eligibility = await adapter.revalidate(
        source_id=source.source_id,
        revision=source.revision,
        scope=source.scope,
        approval_event_id=source.approval.event_id if source.approval else None,
    )
    problem = check_approval_precondition(adapter, eligibility, source)
    if problem is not None:
        return await settle_ineligible(
            user_id, source, result, eligibility.state, problem
        )

    if not source.has_verifying_signal:
        return await settle(
            user_id,
            source,
            result,
            "skipped",
            "no concrete outcome (tool check or explicit confirmation) recorded",
        )
    if (
        await _open_proposals(user_id, source.scope.owner_key)
        >= MAX_OPEN_PROPOSALS_PER_OWNER
    ):
        return await record(
            user_id,
            source,
            result,
            "deferred",
            "too many open proposals",
            advance=False,
        )

    bundle = await adapter.load_evidence(source, max_chars=MAX_EVIDENCE_CHARS)
    if bundle.is_empty:
        return await settle(
            user_id, source, result, "inaccessible_evidence", "no readable evidence"
        )
    if not bundle.verification_complete:
        return await record(
            user_id,
            source,
            result,
            "deferred",
            "verification evidence is clipped or missing; not reviewed",
            advance=False,
        )

    settled = await already_settled(user_id, source, result)
    if settled is not None:
        return settled
    budget_ok, budget_skip = await check_dream_budget(user_id, config=config)
    if not budget_ok:
        return await record(
            user_id,
            source,
            result,
            "budget_exhausted",
            budget_skip or "insufficient_credits",
            advance=False,
        )
    existing = await _existing_skills(user_id, source.scope.expert_id)
    try:
        completion = await review_evidence(config, bundle, existing)
    except DreamLLMError as exc:
        return await record_provider_error(user_id, source, result, exc, config)
    result.model_calls += 1
    cost = await record_review_cost(
        user_id=user_id, run_id=result.run_id, usage=completion.usage
    )
    stamp = ReviewStamp(
        source=stored,
        source_revision=source.revision,
        policy_version=POLICY_VERSION,
        run_id=result.run_id,
        model=completion.usage.model,
        input_tokens=completion.usage.input_tokens,
        output_tokens=completion.usage.output_tokens,
        cost_microdollars=cost,
    )
    proposal = completion.value
    rejection = validate_proposal(proposal, bundle, existing)
    if rejection is not None:
        disposition = "no_novel_procedure" if proposal.decision == "skip" else "skipped"
        return await settle(
            user_id,
            source,
            result,
            disposition,
            safe_diagnostic(rejection),
            stamp=stamp,
        )

    request = build_publish_request(user_id, source, proposal, bundle, stamp)
    outcome = await publish_learned_version(request)
    return await settle_outcome(user_id, source, result, outcome, stamp, request)


async def _existing_skills(user_id: str, expert_id: str | None) -> list[ParsedSkill]:
    skills = await list_user_skills(user_id, expert_id, heal_missing=False)
    out: list[ParsedSkill] = []
    for skill in skills[:MAX_EXISTING_SKILL_BODIES]:
        full = await read_user_skill_with_body(user_id, skill.name, expert_id=expert_id)
        out.append(full or skill)
    return out


async def _open_proposals(user_id: str, owner_key: str) -> int:
    return len(await skill_versions_db().list_open_decisions(user_id, owner_key))


# ---------------------------------------------------------------------------
# Lease
# ---------------------------------------------------------------------------


async def _acquire_lease(user_id: str) -> AsyncClusterLock | None:
    lock = AsyncClusterLock(
        redis=await get_redis_async(),
        key=f"{LEASE_KEY_PREFIX}{user_id}",
        owner_id=uuidlib.uuid4().hex,
        timeout=LEASE_TTL_SECONDS,
    )
    if await lock.try_acquire() != lock.owner_id:
        return None
    return lock


async def _release_lease(lock: AsyncClusterLock) -> None:
    try:
        await lock.release()
    except Exception:
        logger.warning(
            "Skill learning lease release failed; TTL will clear", exc_info=True
        )
