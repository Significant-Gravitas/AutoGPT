"""The plan job: the greeting's prompts ranked and owned before the dump completes.

The last step of the background pipeline, run once the greeting job and the
team job have both stored their results. It reads those back, has Jev rank
the prompts and pick their owners (``plan.judge_plan``), or orders and owns
them deterministically when Jev is unavailable
(``plan_owners.fallback_assign``), caps the team at three
(``plan_owners.cap_team``), and writes the prompts and the team together.
When the greeting fell back to the generic starters, a user who picked a
role gets that role's set instead, owned from the same team.

Same contract as the jobs it follows: it never raises and never touches the
dump status. A plan it cannot write leaves the greeting's own prompts and
team in place, which still read as a plan, just unranked and Otto-owned.
"""

import logging
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.api.features.onboarding_dump import db, plan, plan_owners, role_tasks
from backend.api.features.onboarding_dump.models import (
    ExpertRecommendations,
    RecommendedExpert,
    SuggestedPrompt,
)

logger = logging.getLogger(__name__)

# Each attempt is one read and one conditional write, as for task updates.
_WRITE_ATTEMPTS = 3


class _Judgement(BaseModel):
    """Jev's answer for one list of prompts, asked once across retries."""

    judged: plan.JudgedPlan | None = None
    prompts: list[SuggestedPrompt] | None = None
    latency_ms: float | None = None


async def store_ranked_plan(
    user_id: str,
    recording_id: str,
    transcript: str,
    user_role: str | None,
    pain_points: list[str],
) -> None:
    try:
        await _store(user_id, recording_id, transcript, user_role, pain_points)
    except Exception as e:  # background task, nowhere to raise
        logger.warning(f"Brain dump plan not stored for user {user_id}: {e}")


async def _store(
    user_id: str,
    recording_id: str,
    transcript: str,
    user_role: str | None,
    pain_points: list[str],
) -> None:
    judgement = _Judgement()
    for _ in range(_WRITE_ATTEMPTS):
        dump = await db.get_dump(user_id)
        if dump is None or dump.recordingId != recording_id:
            return
        prompts = _rankable_prompts(dump.suggestedPrompts)
        team = stored_team(dump.recommendedExperts)
        if not prompts or team is None:
            return
        source = "llm"
        if user_role and role_tasks.is_generic_set(prompts):
            source = "role set"
            ranked = role_tasks.role_prompts(user_role, team.experts)
            order = list(range(len(ranked)))
        else:
            ranked, order = await _rank(
                judgement, transcript, user_role, pain_points, prompts, team.experts
            )
        ranked, capped = plan_owners.cap_team(ranked, team.experts)
        if await db.replace_plan(
            user_id,
            recording_id,
            dump.updatedAt,
            _rows(ranked, order),
            team.model_copy(update={"experts": capped}).model_dump(mode="json"),
        ):
            _log(user_id, source, ranked, judgement)
            return
    logger.warning(f"Brain dump plan: gave up after {_WRITE_ATTEMPTS} lost writes")


async def _rank(
    judgement: _Judgement,
    transcript: str,
    user_role: str | None,
    pain_points: list[str],
    prompts: list[SuggestedPrompt],
    team: list[RecommendedExpert],
) -> tuple[list[SuggestedPrompt], list[int]]:
    """Jev's order and owners, or the greeting's order with hinted owners.

    Jev is asked once per list of prompts; a lost write re-reads only to
    retry the write.
    """
    if prompts != judgement.prompts:
        started = time.monotonic()
        judgement.judged = await plan.judge_plan(
            transcript, user_role, pain_points, prompts, team
        )
        judgement.prompts = prompts
        judgement.latency_ms = (time.monotonic() - started) * 1000
    if judgement.judged:
        return judgement.judged.prompts, judgement.judged.order
    return plan_owners.fallback_assign(prompts, team), list(range(len(prompts)))


def _rows(prompts: list[SuggestedPrompt], order: list[int]) -> list[dict[str, Any]]:
    """The rows to store, each keeping the id its position in the greeting's
    list gave it: a reader may already have handed those ids out."""
    return [
        {**prompt.model_dump(mode="json"), "id": str(index)}
        for prompt, index in zip(prompts, order)
    ]


def _rankable_prompts(raw: object) -> list[SuggestedPrompt]:
    """The greeting's prompts as written, or nothing when they are not.

    Rows that already carry a state have been worked on, so the job leaves
    those alone.
    """
    if not isinstance(raw, list):
        return []
    if any(not isinstance(item, dict) or "state" in item for item in raw):
        return []
    try:
        return [SuggestedPrompt.model_validate(item) for item in raw]
    except ValidationError:
        return []


def stored_team(raw: object) -> ExpertRecommendations | None:
    """The team the team job stored, or None when there is none to read."""
    if raw is None:
        return None
    try:
        return ExpertRecommendations.model_validate(raw)
    except ValidationError:
        return None


def _log(
    user_id: str, source: str, ranked: list[SuggestedPrompt], judgement: _Judgement
) -> None:
    owners = sum(1 for prompt in ranked if prompt.expert_template_id)
    judged = judgement.judged
    latency = (
        f"jev {judgement.latency_ms:.0f}ms"
        if judgement.latency_ms is not None
        else "jev not asked"
    )
    detail = (
        f"scores {judged.scores}, "
        f"owner p {[round(p, 2) for p in judged.owner_probabilities]}"
        if judged
        else "fallback order"
    )
    logger.info(
        f"Brain dump plan for user {user_id}: source={source}, {len(ranked)} tasks, "
        f"{owners} owned, fallback={judged is None}, {latency}, {detail}"
    )
