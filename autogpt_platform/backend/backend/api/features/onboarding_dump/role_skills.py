"""What a roster template does, read from the words of its role.

Template roles are written for people ("Research, Data & KPI Analysis",
"Market & Competitor Intelligence") and change as the roster does, so work is
matched by the words a role contains, never by the whole role: a role has a
skill when it contains one of that skill's words or phrases, in any case. A
role that names none of them has no skill, takes no task, and leaves the task
with Otto. Among several people with a skill, the most focused one takes the
work: "Marketing" over "Motion Graphics and Video Creation" for a post.
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
    "sales": ("sales", "partnerships", "deal desk"),
    "crm": ("crm", "revops"),
    "ops": ("ops", "operations", "procurement"),
    # Any money work; bookkeeping and fundraising name the specialists.
    "finance": ("finance", "bookkeeping", "invoicing", "accounting", "fundraising"),
    "bookkeeping": ("bookkeeping", "invoicing", "accounting"),
    "fundraising": ("fundraising", "investor relations"),
    "recruiting": ("recruiting", "hiring", "talent"),
    "people": ("people", "hr"),
    "assistant": ("assistant",),
    "support": ("customer support", "customer success", "help desk", "retention"),
    "engineering": ("engineering", "code", "qa"),
    "security": ("security", "dependency", "dependencies"),
    "legal": ("legal", "contracts", "compliance", "privacy"),
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


def best_for(skills: Sequence[str], people: Iterable[T]) -> T | None:
    """Whoever of ``people`` is most focused (``most_focused``) on the first
    of ``skills`` anyone has.

    Skills are in order of preference, so a later one is only read when
    nobody has an earlier one.
    """
    listed = list(people)
    for skill in skills:
        best = most_focused(skill, listed)
        if best is not None:
            return best
    return None


def most_focused(skill: str, people: Iterable[T]) -> T | None:
    """The one of ``people`` most focused on ``skill``: the fewest other
    skills, then the shortest role, then the order given. A "Sales" role
    beats "Deal Desk & Proposal Support" for sales work."""
    candidates = [person for person in people if has_skill(person.role, skill)]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda p: (len(skills_of(p.role)), len(" ".join(p.role.split()))),
    )


@cache
def _pattern(word: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){re.escape(word)}(?![\w-])", re.IGNORECASE)
