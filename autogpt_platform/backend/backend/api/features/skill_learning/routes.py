"""User-facing skill-learning API: inspect, control, and recover learning.

Scope is always the authenticated caller's own: an ``expert_id`` must name
an active, privately owned expert, and every read or write goes through
the data layer's owner-checked functions. There is no target-user path.
"""

from __future__ import annotations

import logging
from typing import Annotated

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, HTTPException, Path, Query, Security

from backend.api.features.experts import experts_db
from backend.api.features.skill_learning.models import (
    DecisionRequest,
    EditSkillRequest,
    ExpertLearningPolicy,
    ExpertLearningPolicyRequest,
    LearningActionResult,
    LearningHistoryItem,
    LearningHistoryResponse,
    LearningStatus,
    OpenDecisionsResponse,
    OutcomeReportRequest,
    RestoreVersionRequest,
    SkillLearningDetail,
    SkillLearningPolicy,
    SkillPolicyRequest,
    SourceExclusionResult,
)
from backend.api.features.skill_learning.views import (
    disposition_label,
    history_item_from_review,
    history_item_from_version,
    state_label,
    use_summary,
    version_summary,
)
from backend.copilot.learning.contract import list_source_kinds
from backend.copilot.learning.nightly import (
    SkillLearningResult,
    run_skill_learning_pass,
)
from backend.copilot.learning.owner_actions import (
    apply_owner_edit,
    decide_proposal,
    restore_version,
)
from backend.copilot.learning.publish import PublishOutcome
from backend.copilot.learning.revocation import invalidate_versions_for_source
from backend.copilot.tools.skills import invalidate_skills_index_cache
from backend.data import skill_learning as learning_data
from backend.data import skill_reviews as reviews_data
from backend.data import skill_use as use_data
from backend.data import skill_versions as versions_data
from backend.data.skill_learning import owner_key_for
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/skill-learning",
    tags=["skill-learning", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)

_EXPERT_QUERY = Query(description="Expert scope; omit for personal")
_EXPERT_PATH = Path(
    min_length=1, max_length=128, description="Expert id owned by the caller"
)
_NAME_PATH = Path(min_length=1, max_length=64, description="Skill slug")
_VERSION_PATH = Path(min_length=1, max_length=128, description="Skill version id")
_LIMIT_QUERY = Query(ge=1, le=200)


async def _require_scope(user_id: str, expert_id: str | None) -> str | None:
    if expert_id is None:
        return None
    if not await experts_db.owns_private_active_expert(user_id, expert_id):
        raise HTTPException(status_code=404, detail="Expert not found")
    return expert_id


def _action_result(outcome: PublishOutcome, viewer_id: str) -> LearningActionResult:
    return LearningActionResult(
        status=outcome.status,
        status_label=disposition_label(outcome.status),
        reason=outcome.reason,
        version=None,
        pattern_class=outcome.pattern_class,
        blocked_step=outcome.blocked_step,
    )


async def _with_version(
    result: LearningActionResult, outcome: PublishOutcome, viewer_id: str
) -> LearningActionResult:
    if outcome.version is not None:
        result.version = await version_summary(
            viewer_id, outcome.version, include_body=False
        )
    return result


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


@router.get("/status", operation_id="get_skill_learning_status")
async def get_status(
    expert_id: Annotated[str | None, _EXPERT_QUERY] = None,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> LearningStatus:
    """Operator view: backlog, freshness, retries, cost, and pause state."""
    scope = await _require_scope(user_id, expert_id)
    summary = await reviews_data.summarize_learning(user_id)
    paused = False
    if scope is not None:
        expert = await experts_db.get_expert(user_id, scope, include_workflows=False)
        paused = bool(expert and expert.learning_paused_at)
    return LearningStatus(
        enabled=await is_feature_enabled(Flag.DREAM_SKILL_LEARNING_ENABLED, user_id),
        expert_id=scope,
        learning_paused=paused,
        pending_sources=summary.pending_sources,
        oldest_pending_at=summary.oldest_pending_at,
        last_review_at=summary.last_review_at,
        last_applied_at=summary.last_applied_at,
        retrying_reviews=summary.retrying_reviews,
        cost_microdollars_30d=summary.cost_microdollars_30d,
        source_kinds=list_source_kinds(),
    )


@router.get("/history", operation_id="list_skill_learning_history")
async def list_history(
    expert_id: Annotated[str | None, _EXPERT_QUERY] = None,
    origin: Annotated[str | None, Query(max_length=32)] = None,
    state: Annotated[str | None, Query(max_length=32)] = None,
    limit: Annotated[int, _LIMIT_QUERY] = 50,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> LearningHistoryResponse:
    """Versions and review dispositions, newest first, for one scope."""
    scope = await _require_scope(user_id, expert_id)
    owner_key = owner_key_for(scope)
    versions = await versions_data.list_recent_versions(
        user_id, owner_key=owner_key, origin=origin, state=state, limit=limit
    )
    items: list[LearningHistoryItem] = [
        await history_item_from_version(user_id, v) for v in versions
    ]
    if origin is None and state is None:
        reviews = await reviews_data.list_reviews(
            user_id, owner_key=owner_key, limit=limit
        )
        items.extend(
            [
                await history_item_from_review(user_id, r)
                for r in reviews
                if r.disposition not in ("applied", "applied_pending")
            ]
        )
    items.sort(key=lambda item: item.created_at, reverse=True)
    return LearningHistoryResponse(expert_id=scope, items=items[:limit])


@router.get("/decisions", operation_id="list_skill_learning_decisions")
async def list_decisions(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> OpenDecisionsResponse:
    """One list of open proposals across every scope the caller owns."""
    open_versions = await versions_data.list_open_decisions(user_id)
    return OpenDecisionsResponse(
        items=[
            await version_summary(user_id, v, include_body=True) for v in open_versions
        ]
    )


@router.get("/skills/{name}", operation_id="get_skill_learning_detail")
async def get_skill_detail(
    name: Annotated[str, _NAME_PATH],
    expert_id: Annotated[str | None, _EXPERT_QUERY] = None,
    version_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> SkillLearningDetail:
    """Current version, evidence, limits, sources, history, policy, and reuse."""
    scope = await _require_scope(user_id, expert_id)
    slug = name.strip().lower()
    owner_key = owner_key_for(scope)
    head = await versions_data.get_head(user_id, owner_key, slug)
    if head is None:
        raise HTTPException(
            status_code=404, detail="No learning history for this skill"
        )
    versions = await versions_data.list_versions(user_id, owner_key, slug)
    by_id = {v.id: v for v in versions}

    async def include_version(
        required_id: str | None,
    ) -> versions_data.SkillVersionRecord | None:
        if required_id is None:
            return None
        version = by_id.get(required_id)
        if version is None:
            version = await versions_data.get_version(user_id, required_id)
        if (
            version is None
            or version.owner_key != owner_key
            or version.skill_name != slug
        ):
            raise HTTPException(status_code=404, detail="Skill version not found")
        by_id[version.id] = version
        return version

    selected = await include_version(version_id)
    current = await include_version(head.current_version_id)
    for version in (selected, current):
        if version:
            await include_version(version.base_version_id)
    versions = sorted(by_id.values(), key=lambda v: v.version, reverse=True)
    open_decision = next((v for v in versions if v.state == "needs_decision"), None)
    events = await use_data.list_use_events(user_id, owner_key, slug)
    if head.use_paused_at is not None:
        state, label = "paused", "Paused (not in use)"
    elif head.learning_paused_at is not None:
        state, label = "paused", "Paused (learning)"
    elif current is not None:
        state, label = current.state, state_label(current.state)
    else:
        state, label = "candidate", state_label("candidate")
    return SkillLearningDetail(
        skill_name=slug,
        expert_id=scope,
        state=state,
        state_label=label,
        current_version=(
            await version_summary(user_id, current, include_body=True, events=events)
            if current
            else None
        ),
        open_decision=(
            await version_summary(
                user_id, open_decision, include_body=True, events=events
            )
            if open_decision
            else None
        ),
        versions=[
            await version_summary(user_id, v, include_body=True, events=events)
            for v in versions
        ],
        policy=SkillLearningPolicy(
            auto_improve=head.auto_improve,
            learning_paused=head.learning_paused_at is not None,
            use_paused=head.use_paused_at is not None,
        ),
        use=use_summary(events, head.current_version_id),
    )


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


@router.post("/skills/{name}/policy", operation_id="update_skill_learning_policy")
async def update_policy(
    name: Annotated[str, _NAME_PATH],
    body: SkillPolicyRequest,
    expert_id: Annotated[str | None, _EXPERT_QUERY] = None,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> SkillLearningPolicy:
    """Pause learning, stop using, or toggle automatic improvements — separately."""
    scope = await _require_scope(user_id, expert_id)
    slug = name.strip().lower()
    head = await versions_data.ensure_head(user_id, scope, slug)
    updated = await versions_data.update_head_policy(
        user_id,
        head.owner_key,
        slug,
        auto_improve=body.auto_improve,
        learning_paused=body.learning_paused,
        use_paused=body.use_paused,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    if body.use_paused is not None:
        await invalidate_skills_index_cache(user_id, scope)
    return SkillLearningPolicy(
        auto_improve=updated.auto_improve,
        learning_paused=updated.learning_paused_at is not None,
        use_paused=updated.use_paused_at is not None,
    )


@router.post("/skills/{name}/edit", operation_id="edit_learned_skill")
async def edit_skill(
    name: Annotated[str, _NAME_PATH],
    body: EditSkillRequest,
    expert_id: Annotated[str | None, _EXPERT_QUERY] = None,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> LearningActionResult:
    """Apply the owner's correction now; later automated changes become
    proposals unless ``keep_auto_improve`` is set.

    A content-check block is a domain outcome, not a transport error: the
    response is 200 with ``status == "blocked_content"`` and only the
    pattern class and ordinal step, so the client keeps the draft and shows
    the reason without ever echoing the blocked value.
    """
    scope = await _require_scope(user_id, expert_id)
    outcome = await apply_owner_edit(
        user_id=user_id,
        expert_id=scope,
        skill_name=name.strip().lower(),
        description=body.description,
        body=body.body,
        triggers=body.triggers,
        keep_auto_improve=body.keep_auto_improve,
        allowed_pattern_classes=body.allowed_pattern_classes,
        expected_version_id=body.expected_version_id,
    )
    return await _with_version(_action_result(outcome, user_id), outcome, user_id)


@router.post("/skills/{name}/restore", operation_id="restore_learned_skill_version")
async def restore(
    name: Annotated[str, _NAME_PATH],
    body: RestoreVersionRequest,
    expert_id: Annotated[str | None, _EXPERT_QUERY] = None,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> LearningActionResult:
    """Restore prior content as a new version. Affects future use only."""
    scope = await _require_scope(user_id, expert_id)
    outcome = await restore_version(
        user_id=user_id,
        expert_id=scope,
        skill_name=name.strip().lower(),
        version_id=body.version_id,
        actor_user_id=user_id,
    )
    if (
        outcome.status == "conflict"
        and outcome.version is None
        and "not found" in outcome.reason
    ):
        raise HTTPException(status_code=404, detail=outcome.reason)
    return await _with_version(_action_result(outcome, user_id), outcome, user_id)


@router.post(
    "/skills/{name}/decisions/{version_id}",
    operation_id="decide_skill_learning_proposal",
)
async def decide(
    name: Annotated[str, _NAME_PATH],
    version_id: Annotated[str, _VERSION_PATH],
    body: DecisionRequest,
    expert_id: Annotated[str | None, _EXPERT_QUERY] = None,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> LearningActionResult:
    """Keep current, apply the proposal, or apply an edited alternative."""
    scope = await _require_scope(user_id, expert_id)
    outcome = await decide_proposal(
        user_id=user_id,
        expert_id=scope,
        skill_name=name.strip().lower(),
        version_id=version_id,
        action=body.action,
        edited_body=body.edited_body,
        actor_user_id=user_id,
    )
    if outcome.status == "conflict" and "not found" in outcome.reason:
        raise HTTPException(status_code=404, detail=outcome.reason)
    return await _with_version(_action_result(outcome, user_id), outcome, user_id)


@router.post("/skills/{name}/outcome", operation_id="report_skill_outcome")
async def report_outcome(
    name: Annotated[str, _NAME_PATH],
    body: OutcomeReportRequest,
    expert_id: Annotated[str | None, _EXPERT_QUERY] = None,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> LearningActionResult:
    """Attach an explicit outcome to the exact version used (evidence for a
    later review; it does not change the skill)."""
    scope = await _require_scope(user_id, expert_id)
    slug = name.strip().lower()
    version = await versions_data.get_version(user_id, body.version_id)
    if version is None or version.skill_name != slug:
        raise HTTPException(status_code=404, detail="Version not found")
    kind = "outcome_unknown" if body.outcome == "unknown" else body.outcome
    await use_data.record_use_event(
        user_id,
        expert_id=scope,
        skill_name=slug,
        kind=kind,
        version_id=version.id,
        detail=body.detail,
        actor_user_id=user_id,
    )
    return LearningActionResult(
        status="recorded", status_label="Outcome recorded", reason=body.outcome
    )


@router.post(
    "/experts/{expert_id}/policy", operation_id="update_expert_learning_policy"
)
async def update_expert_policy(
    expert_id: Annotated[str, _EXPERT_PATH],
    body: ExpertLearningPolicyRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ExpertLearningPolicy:
    """Pause or resume nightly learning for one expert (use is unaffected)."""
    expert = await experts_db.set_expert_learning_paused(
        user_id, expert_id, body.learning_paused
    )
    if expert is None:
        raise HTTPException(status_code=404, detail="Expert not found")
    return ExpertLearningPolicy(
        expert_id=expert.id, learning_paused_at=expert.learning_paused_at
    )


@router.post(
    "/sources/{source_kind}/{source_ref}/exclude",
    operation_id="exclude_skill_learning_source",
)
async def exclude_source(
    source_kind: Annotated[str, Path(min_length=1, max_length=64)],
    source_ref: Annotated[str, Path(min_length=1, max_length=256)],
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> SourceExclusionResult:
    """Exclude a source from learning, including queued and published work.

    Versions that depend on it become unavailable; a prior eligible version
    is restored where one exists, otherwise the skill is paused.
    """
    record = await learning_data.get_source_by_ref(user_id, source_kind, source_ref)
    if record is None:
        raise HTTPException(status_code=404, detail="Source not found")
    excluded = await learning_data.set_source_eligibility(
        user_id, record.id, "excluded", excluded_by_user_id=user_id
    )
    if excluded is None:
        raise HTTPException(status_code=404, detail="Source not found")
    invalidated = await invalidate_versions_for_source(
        user_id, excluded, "source excluded from learning by its owner"
    )
    return SourceExclusionResult(
        source_id=record.id, excluded=True, invalidated_version_ids=invalidated
    )


@router.post("/run", operation_id="run_skill_learning_now")
async def run_now(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> SkillLearningResult:
    """Run one learning pass for the caller now (same validation as nightly)."""
    return await run_skill_learning_pass(user_id, trigger="admin")
