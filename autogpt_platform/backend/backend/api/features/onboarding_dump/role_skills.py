"""What a roster template does, read from the words of its role.

Template roles are written for people ("Research, Data & KPI Analysis",
"Market & Competitor Intelligence") and change as the roster does, so work is
matched by the words a role contains, never by the whole role: a role has a
skill when it contains one of that skill's words or phrases, in any case. A
role that names none of them has no skill, takes no task, and leaves the task
with Otto.
"""

import re
from collections.abc import Iterable, Sequence
from functools import cache
from typing import Protocol, TypeVar

# Each skill's words and phrases, matched whole ("ops" is not in "RevOps").
SKILL_WORDS: dict[str, tuple[str, ...]] = {
    "product": ("product", "project management"),
    "data": ("data", "kpi", "kpis", "analytics", "metrics"),
    "competitors": ("competitor", "competitors", "competitive", "intelligence"),
    "research": ("research",),
    "marketing": (
        "marketing",
        "seo",
        "content",
        "social",
        "ads",
        "email",
        "lifecycle",
        "pr",
        "communications",
        "brand",
        "video",
        "go-to-market",
    ),
    "sales": ("sales", "partnerships", "deal desk", "revops", "crm"),
    "crm": ("crm", "revops"),
    "ops": ("ops", "operations", "procurement"),
    "finance": ("finance", "bookkeeping", "invoicing", "fundraising"),
    "recruiting": ("recruiting", "hiring", "talent"),
    "people": ("people", "hr"),
    "assistant": ("assistant",),
    "support": ("customer support", "customer success", "help desk", "retention"),
    "engineering": ("engineering", "code", "qa"),
    "security": ("security", "dependency", "dependencies"),
    "legal": ("contracts", "compliance", "privacy"),
}


class HasRole(Protocol):
    @property
    def role(self) -> str: ...


T = TypeVar("T", bound=HasRole)


def has_skill(role: str, skill: str) -> bool:
    """Whether ``role`` contains one of ``skill``'s words."""
    return any(_pattern(word).search(role) for word in SKILL_WORDS[skill])


def skills_of(role: str) -> frozenset[str]:
    """Every skill ``role`` has."""
    return frozenset(skill for skill in SKILL_WORDS if has_skill(role, skill))


def first_with_skill(skills: Sequence[str], people: Iterable[T]) -> T | None:
    """The first of ``people`` with the first of ``skills`` anyone has.

    Skills are in order of preference, so a later one is only read when
    nobody has an earlier one.
    """
    listed = list(people)
    return next(
        (
            person
            for skill in skills
            for person in listed
            if has_skill(person.role, skill)
        ),
        None,
    )


def best_template(skill: str, templates: Iterable[T]) -> T | None:
    """The template most focused on ``skill``: the one with the fewest other
    skills, then the shortest role, then the roster's order. A "Sales" role
    beats "Deal Desk & Proposal Support" for sales work."""
    candidates = [t for t in templates if has_skill(t.role, skill)]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda t: (len(skills_of(t.role)), len(" ".join(t.role.split()))),
    )


@cache
def _pattern(word: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){re.escape(word)}(?![\w-])", re.IGNORECASE)
