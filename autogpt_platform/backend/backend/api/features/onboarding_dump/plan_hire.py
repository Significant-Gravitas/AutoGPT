"""The hire that follows the trial: the plan's team joins the user's.

``hire_plan`` settles the plan first, so the team hired is the one whose
tasks Home lists. It is idempotent and never fails the request for one bad
template: whoever could not join is reported, and their tasks read as
Otto's on Home. A call that changed anything is sent to PostHog once, as
``plan_experts_hired``.
"""

import logging

from pydantic import BaseModel

from backend.api.features.experts import experts_db
from backend.api.features.experts.models import Expert
from backend.api.features.onboarding_dump import plan_service, suggested_tasks
from backend.api.features.onboarding_dump.models import RecommendedExpert
from backend.api.features.onboarding_dump.plan_models import (
    PlanHiredExpert,
    PlanHireFailure,
    PlanHireResponse,
)
from backend.util.exceptions import ConflictError, ExpertLimitExceededError
from backend.util.funnel_analytics import emit_funnel_event
from backend.util.posthog_events import PostHogEvent

logger = logging.getLogger(__name__)


async def hire_plan(user_id: str) -> PlanHireResponse:
    """Hire the plan's team, skipping anyone already on it.

    A second call hires nobody and reports the same people. One failed hire
    is reported and the rest go ahead; the active expert limit stops the
    hiring there, and everyone not yet hired is reported with it. A plan that
    could not be settled hires nobody and raises ``ConflictError``: its team
    is one the pipeline may still replace.
    """
    # Settled first: a team hired while the pipeline still runs is the one
    # whose tasks Home then lists, not a team a later plan would replace.
    plan = await plan_service.settle_plan(user_id)
    if not plan.ready:
        raise ConflictError("The plan is still changing, try again")
    team = plan.team
    active = _on_team(await suggested_tasks.roster_by_template(user_id))
    outcome = _HireOutcome()
    for position, expert in enumerate(team):
        if expert.template_id in active:
            outcome.hired.append(active[expert.template_id])
            continue
        try:
            await _hire(user_id, expert, outcome)
        except ExpertLimitExceededError:
            # Nobody else can join; teammates already on the team still are.
            for rest in team[position:]:
                if rest.template_id in active:
                    outcome.hired.append(active[rest.template_id])
                else:
                    outcome.failed.append(_failure(rest))
            break
    _report(user_id, len(team), outcome)
    return PlanHireResponse(hired=outcome.hired, failed=outcome.failed)


class _HireOutcome(BaseModel):
    hired: list[PlanHiredExpert] = []
    failed: list[PlanHireFailure] = []
    # How many of ``hired`` this call put on the team.
    joined: int = 0


async def _hire(user_id: str, expert: RecommendedExpert, outcome: _HireOutcome) -> None:
    """Hire one teammate into ``outcome``. A failed hire is recorded there;
    only the active expert limit, which stops the whole hire, is raised."""
    try:
        result = await experts_db.hire_expert(
            user_id, expert.template_id, None, surface="onboarding", via="plan"
        )
    except ExpertLimitExceededError:
        raise
    except Exception as e:
        logger.warning(f"Plan hire of {expert.template_id} failed: {e}")
        outcome.failed.append(_failure(expert))
        return
    # Counted from the hire itself: with the roster unread, or a second call
    # racing this one, a teammate already on the team reaches here too.
    outcome.joined += result.joined
    outcome.hired.append(
        PlanHiredExpert(
            template_id=expert.template_id,
            expert_id=result.expert.id,
            name=result.expert.name,
        )
    )


def _report(user_id: str, team_size: int, outcome: _HireOutcome) -> None:
    """Log the hire, and send ``plan_experts_hired`` when it hired or failed
    anyone: a repeat that finds the whole team already on it sends nothing."""
    joined, failed = outcome.joined, len(outcome.failed)
    already = len(outcome.hired) - joined
    logger.info(
        f"Brain dump plan hire for user {user_id}: {team_size} in the team, "
        f"{joined} joined, {already} already on it, {failed} failed"
    )
    if not joined and not failed:
        return
    emit_funnel_event(
        user_id,
        PostHogEvent.PLAN_EXPERTS_HIRED,
        {"hired": joined, "already_on_team": already, "failed": failed},
    )


def _on_team(roster: dict[str, Expert]) -> dict[str, PlanHiredExpert]:
    return {
        template_id: PlanHiredExpert(
            template_id=template_id, expert_id=expert.id, name=expert.name
        )
        for template_id, expert in roster.items()
    }


def _failure(expert: RecommendedExpert) -> PlanHireFailure:
    return PlanHireFailure(template_id=expert.template_id, name=expert.name)
