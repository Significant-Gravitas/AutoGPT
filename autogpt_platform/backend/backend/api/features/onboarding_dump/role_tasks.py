"""Starter tasks for a plan built from the wizard's role alone.

Used when there is no brain dump to read: it was skipped, it failed, or it
left nothing usable. The sets themselves are copy, in ``role_task_sets``. A
role the roster has no expert for, or one outside the team, leaves the task
with Otto, so the plan never shows a face that will not be hired.
"""

from collections.abc import Sequence

from backend.api.features.experts.models import Expert
from backend.api.features.onboarding_dump import intro
from backend.api.features.onboarding_dump.models import (
    RecommendedExpert,
    SuggestedPrompt,
)
from backend.api.features.onboarding_dump.plan_owners import MAX_TEAM, owner_for_role
from backend.api.features.onboarding_dump.recommend_experts import (
    fallback_expert_recommendations,
)
from backend.api.features.onboarding_dump.role_task_sets import (
    OTHER_ROLE,
    ROLE_FALLBACK_TASKS,
)

# Every title a default set can hold: the role sets and the generic starters.
_DEFAULT_TITLES = frozenset(
    {task.title for tasks in ROLE_FALLBACK_TASKS.values() for task in tasks}
    | {prompt.title for prompt in intro.fallback_prompts()}
)
_GENERIC_TITLES = frozenset(prompt.title for prompt in intro.fallback_prompts())


def role_plan(
    user_role: str | None, pain_points: list[str], templates: list[Expert]
) -> tuple[list[SuggestedPrompt], list[RecommendedExpert]]:
    """The role's tasks with owners from its fallback team, and that team.

    The team is the role fallback team capped at three.
    """
    team = fallback_expert_recommendations(user_role, pain_points, templates).experts
    team = team[:MAX_TEAM]
    return role_prompts(user_role, team), team


def role_prompts(
    user_role: str | None, team: list[RecommendedExpert]
) -> list[SuggestedPrompt]:
    """The role's tasks, each owned by whoever on ``team`` has its role.

    ``Other`` and anything the wizard does not offer (Other's typed text)
    take the ``Other`` set; a task whose role has nobody on ``team`` is
    Otto's.
    """
    tasks = ROLE_FALLBACK_TASKS.get(user_role or "", ROLE_FALLBACK_TASKS[OTHER_ROLE])
    names = {expert.template_id: expert.name for expert in team}
    owners = [
        owner_for_role(task.owner_role, team) if task.owner_role else None
        for task in tasks
    ]
    return [
        SuggestedPrompt(
            title=task.title,
            prompt=task.prompt,
            icon=task.icon,
            why=task.why,
            expert_template_id=owner,
            expert_name=names.get(owner) if owner else None,
        )
        for task, owner in zip(tasks, owners)
    ]


def is_default_set(prompts: Sequence[SuggestedPrompt]) -> bool:
    """Whether ``prompts`` are a role set or the generic starters, rather
    than tasks the greeting wrote from what the user said."""
    return bool(prompts) and all(p.title in _DEFAULT_TITLES for p in prompts)


def is_generic_set(prompts: Sequence[SuggestedPrompt]) -> bool:
    """Whether ``prompts`` are the generic starters the greeting falls back
    to when the model gave it nothing usable."""
    return bool(prompts) and all(p.title in _GENERIC_TITLES for p in prompts)
