"""Who takes each task when Jev does not say, and the team the plan shows.

``fallback_assign`` keeps the greeting's order and matches owners by icon and
title. ``cap_team`` keeps the team to the three people the plan step shows,
so every face there is someone who will be hired. ``owner_for_role`` is the
one rule for matching a template role to someone on a team, shared with the
role sets.
"""

import re
from typing import TypeVar

from backend.api.features.onboarding_dump.models import (
    RecommendedExpert,
    SuggestedPrompt,
)

P = TypeVar("P", bound=SuggestedPrompt)

MAX_TEAM = 3

# What a prompt's icon, then its title, says about who takes it: template
# roles in order of preference. A hint that names no role on the team gives
# no owner, so the task stays Otto's. "research" names no template today; it
# is kept first so a research expert, once the roster has one, takes these.
OWNER_HINTS: dict[str, tuple[str, ...]] = {
    "envelope": ("sales",),
    "chats": ("sales",),
    "users": ("sales",),
    "target": ("sales",),
    "megaphone": ("marketing",),
    "newspaper": ("marketing",),
    "globe": ("marketing",),
    "chart-bar": ("bookkeeping", "ops"),
    "currency-dollar": ("bookkeeping", "ops"),
    "file-text": ("bookkeeping", "ops"),
    "magnifying-glass": ("research", "marketing"),
    "bell": ("research", "marketing"),
    "calendar-check": ("ops", "executive assistant"),
    "clock": ("ops", "executive assistant"),
    "code": ("engineering",),
    "robot": ("engineering",),
}
TITLE_HINTS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
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
        ("bookkeeping", "ops"),
    ),
    (
        re.compile(r"\b(applicants?|candidates?|hiring|interviews?)\b", re.IGNORECASE),
        ("recruiting", "talent sourcing"),
    ),
    (
        re.compile(r"\b(code|bugs?|deploys?|pull requests?)\b", re.IGNORECASE),
        ("engineering",),
    ),
    (
        re.compile(r"\b(research|competitors?)\b", re.IGNORECASE),
        ("research", "marketing"),
    ),
]


def fallback_assign(
    prompts: list[SuggestedPrompt], team: list[RecommendedExpert]
) -> list[SuggestedPrompt]:
    """The greeting's order unchanged, owners matched by icon and title."""
    return [
        prompt.model_copy(update={"expert_template_id": _hinted_owner(prompt, team)})
        for prompt in prompts
    ]


def cap_team(
    prompts: list[P], team: list[RecommendedExpert]
) -> tuple[list[P], list[RecommendedExpert]]:
    """The plan's team and the prompts with owners only from it.

    The team is the distinct owners of the first three tasks, in that order,
    padded from the recommendation's own order up to three. A task owned by
    anyone else becomes Otto's, so no face is shown that will not be hired;
    a kept owner's name is written beside them.
    """
    by_id = {expert.template_id: expert for expert in team}
    owners = list(
        dict.fromkeys(
            prompt.expert_template_id
            for prompt in prompts[:MAX_TEAM]
            if prompt.expert_template_id in by_id
        )
    )
    padding = [e.template_id for e in team if e.template_id not in owners]
    picked = (owners + padding)[:MAX_TEAM]
    kept = set(picked)
    return (
        [_owned_by(prompt, by_id, kept) for prompt in prompts],
        [by_id[template_id] for template_id in picked],
    )


def _owned_by(prompt: P, by_id: dict[str, RecommendedExpert], kept: set[str]) -> P:
    """``prompt`` with its owner (and the owner's name) only if kept."""
    owner = prompt.expert_template_id if prompt.expert_template_id in kept else None
    name = by_id[owner].name if owner else None
    if owner == prompt.expert_template_id and name == prompt.expert_name:
        return prompt
    return prompt.model_copy(update={"expert_template_id": owner, "expert_name": name})


def owner_for_role(role: str, team: list[RecommendedExpert]) -> str | None:
    """The template id of the first expert on ``team`` with ``role``."""
    return next(
        (e.template_id for e in team if e.role.strip().lower() == role),
        None,
    )


def _hinted_owner(prompt: SuggestedPrompt, team: list[RecommendedExpert]) -> str | None:
    title_roles = [
        role
        for pattern, hinted in TITLE_HINTS
        if pattern.search(prompt.title)
        for role in hinted
    ]
    roles = [*OWNER_HINTS.get(prompt.icon, ()), *title_roles]
    return next(
        (
            owner
            for role in dict.fromkeys(roles)
            if (owner := owner_for_role(role, team)) is not None
        ),
        None,
    )
