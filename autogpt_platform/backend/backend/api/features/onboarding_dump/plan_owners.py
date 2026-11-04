"""Who takes each task when Jev does not say, and the team the plan shows.

``fallback_assign`` keeps the greeting's order and matches owners by title,
then icon. ``cap_team`` keeps the team to the three people the plan step
shows, so every face there is someone who will be hired. ``owners_for`` is the
one rule for matching work to a team, shared with the role sets: by the
skills each role's words show (``role_skills``), never by the whole role,
which the roster renames, and only among the three the plan keeps.
"""

import re
from collections.abc import Sequence
from typing import TypeVar

from backend.api.features.onboarding_dump import role_skills
from backend.api.features.onboarding_dump.models import (
    RecommendedExpert,
    SuggestedPrompt,
)

P = TypeVar("P", bound=SuggestedPrompt)

MAX_TEAM = 3

# What a prompt's title, then its icon, says about who takes it: skills in
# order of preference. A hint whose skills nobody on the team has gives no
# owner, so the task stays Otto's.
TITLE_HINTS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (
        re.compile(
            r"\b(specs?|prds?|roadmaps?|release notes|features?|feedback|"
            r"user stories|backlog)\b",
            re.IGNORECASE,
        ),
        ("product",),
    ),
    (
        re.compile(r"\b(competitors?|competitive)\b", re.IGNORECASE),
        ("competitors", "research", "marketing"),
    ),
    (
        re.compile(
            r"\b(metrics|kpis?|dashboards?|analytics|numbers|data|reports?|"
            r"funnels?|cohorts?)\b",
            re.IGNORECASE,
        ),
        ("data", "finance", "ops"),
    ),
    (
        re.compile(
            r"\b(leads?|prospects?|outreach|crm|deals?|pipeline)\b", re.IGNORECASE
        ),
        ("sales",),
    ),
    (
        re.compile(
            r"\b(posts?|newsletters?|social|instagram|linkedin|seo|blog|campaigns?)\b",
            re.IGNORECASE,
        ),
        ("marketing",),
    ),
    (
        re.compile(
            r"\b(invoices?|expenses?|bookkeeping|books|payroll)\b", re.IGNORECASE
        ),
        ("finance", "ops"),
    ),
    (
        re.compile(r"\b(applicants?|candidates?|hiring|interviews?)\b", re.IGNORECASE),
        ("recruiting",),
    ),
    (
        re.compile(r"\b(code|bugs?|deploys?|pull requests?)\b", re.IGNORECASE),
        ("engineering",),
    ),
    (
        re.compile(r"\b(tickets?|support|churn)\b", re.IGNORECASE),
        ("support",),
    ),
    (
        re.compile(r"\b(meetings?|calendar|schedul\w*)\b", re.IGNORECASE),
        ("assistant", "ops"),
    ),
    (
        re.compile(r"\bresearch\b", re.IGNORECASE),
        ("research", "competitors"),
    ),
]
OWNER_HINTS: dict[str, tuple[str, ...]] = {
    "envelope": ("sales", "marketing"),
    "chats": ("sales", "support"),
    "users": ("sales",),
    "target": ("sales",),
    "megaphone": ("marketing",),
    "newspaper": ("marketing",),
    "globe": ("marketing",),
    "chart-bar": ("data", "finance", "ops"),
    "currency-dollar": ("finance", "ops"),
    "file-text": ("product", "ops"),
    "magnifying-glass": ("competitors", "research", "marketing"),
    "bell": ("competitors", "research", "marketing"),
    "calendar-check": ("assistant", "ops"),
    "clock": ("assistant", "ops"),
    "code": ("engineering",),
    "robot": ("engineering",),
}


def fallback_assign(
    prompts: list[SuggestedPrompt], team: list[RecommendedExpert]
) -> list[SuggestedPrompt]:
    """The greeting's order unchanged, owners matched by title, then icon."""
    owners = owners_for([_hinted_skills(prompt) for prompt in prompts], team)
    return [
        prompt.model_copy(update={"expert_template_id": owner})
        for prompt, owner in zip(prompts, owners)
    ]


def cap_team(
    prompts: list[P], team: list[RecommendedExpert]
) -> tuple[list[P], list[RecommendedExpert]]:
    """The plan's team (``kept_team``) and the prompts with owners only from
    it. A task owned by anyone else becomes Otto's, so no face is shown that
    will not be hired; a kept owner's name is written beside them.
    """
    by_id = {expert.template_id: expert for expert in team}
    picked = kept_team([prompt.expert_template_id for prompt in prompts], team)
    kept = {expert.template_id for expert in picked}
    return [_owned_by(prompt, by_id, kept) for prompt in prompts], picked


def kept_team(
    owners: Sequence[str | None], team: list[RecommendedExpert]
) -> list[RecommendedExpert]:
    """The distinct owners of the first three tasks, in that order, padded
    from the team's own order up to three."""
    by_id = {expert.template_id: expert for expert in team}
    first = list(dict.fromkeys(o for o in owners[:MAX_TEAM] if o and o in by_id))
    padding = [e.template_id for e in team if e.template_id not in first]
    return [by_id[template_id] for template_id in (first + padding)[:MAX_TEAM]]


def owners_for(
    skills_by_task: Sequence[tuple[str, ...]], team: list[RecommendedExpert]
) -> list[str | None]:
    """Each task's owner: whoever has its skills (``owner_for_skills``) among
    the three the plan keeps, so a task whose best match is cut goes to the
    next match who stays rather than to Otto."""
    first = [owner_for_skills(skills, team) for skills in skills_by_task]
    keep = {expert.template_id for expert in kept_team(first, team)}
    # In the team's order, so the first three tasks keep their owners.
    members = [expert for expert in team if expert.template_id in keep]
    return [owner_for_skills(skills, members) for skills in skills_by_task]


def _owned_by(prompt: P, by_id: dict[str, RecommendedExpert], kept: set[str]) -> P:
    """``prompt`` with its owner (and the owner's name) only if kept."""
    owner = prompt.expert_template_id if prompt.expert_template_id in kept else None
    name = by_id[owner].name if owner else None
    if owner == prompt.expert_template_id and name == prompt.expert_name:
        return prompt
    return prompt.model_copy(update={"expert_template_id": owner, "expert_name": name})


def owner_for_skills(
    skills: tuple[str, ...], team: list[RecommendedExpert]
) -> str | None:
    """The template id of whoever on ``team`` has the first of ``skills``
    anyone has, in the team's order."""
    owner = role_skills.first_with_skill(skills, team)
    return owner.template_id if owner else None


def _hinted_skills(prompt: SuggestedPrompt) -> tuple[str, ...]:
    title_skills = [
        skill
        for pattern, hinted in TITLE_HINTS
        if pattern.search(prompt.title)
        for skill in hinted
    ]
    return tuple(dict.fromkeys([*title_skills, *OWNER_HINTS.get(prompt.icon, ())]))
