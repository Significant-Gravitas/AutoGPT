"""The session owner's skills as capability entries.

Skills belong to a user, and within a user to one owner folder — an
expert's, or personal Otto's — so unlike the other sources they cannot sit
in the process-wide registry: ``tools.session_registry`` layers them onto
it per call.  Entries are metadata only.  Running one loads it through
``read_skill``, exactly as the ``<available_skills>`` index tells the
model to; the index injection stays, this makes the same skills reachable
through search as well.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from backend.copilot.capabilities.models import (
    SKILL_TOOL,
    CapabilityEntry,
    Implementation,
    clip_purpose,
    normalize_text,
)
from backend.copilot.capabilities.text import tokenize

if TYPE_CHECKING:
    from backend.copilot.tools.skills import ParsedSkill

__all__ = ["SKILL_ID_PREFIX", "SKILL_TOOL", "skill_entries", "skill_name"]

SKILL_ID_PREFIX = "skill:"


def skill_entries(skills: "Iterable[ParsedSkill]") -> list[CapabilityEntry]:
    return [_skill_entry(skill) for skill in skills]


def skill_name(capability_id: str) -> str | None:
    """The skill a ``skill:<name>`` id names, or None for any other id."""
    key = capability_id.strip()
    if not key.lower().startswith(SKILL_ID_PREFIX):
        return None
    name = key[len(SKILL_ID_PREFIX) :].strip().lower()
    return name or None


def _skill_entry(skill: "ParsedSkill") -> CapabilityEntry:
    # Triggers ride in the description: they are what the skill says it is
    # for, so they should match a query and show in describe_capability.
    description = skill.description
    if skill.triggers:
        description += " Triggers: " + ", ".join(skill.triggers) + "."
    tags = sorted(set(tokenize(skill.name)))
    tags.append("skill")
    return CapabilityEntry(
        id=f"{SKILL_ID_PREFIX}{skill.name}",
        kind="skill",
        klass="service",
        name=skill.name,
        purpose=clip_purpose(skill.description),
        description=normalize_text(description),
        tags=tags,
        context="direct",
        implementations=[
            Implementation(
                kind="skill", ref=skill.name, name=skill.name, context="direct"
            )
        ],
    )
