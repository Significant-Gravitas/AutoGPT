"""Roster templates and the user's hired experts as capability entries.

A template is someone the user can hire: ``expert:<template_id>`` runs as
``hire_expert`` with the template bound, so hiring goes through the approval
card that tool always returns.  A hired expert is already on the team:
``teammate:<expert_id>`` runs as ``delegate_to_expert`` with the expert bound.
Both depend on the user, so ``tools.session_registry`` layers them on per
call, as it does skills; the roster never reaches the injected prompt.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from backend.copilot.capabilities.models import (
    CapabilityEntry,
    Implementation,
    clip_purpose,
    normalize_text,
)

if TYPE_CHECKING:
    from backend.api.features.experts.models import Expert, ExpertTemplate

EXPERT_ID_PREFIX = "expert:"
TEAMMATE_ID_PREFIX = "teammate:"
HIRE_TOOL = "hire_expert"
DELEGATE_TOOL = "delegate_to_expert"

_DISPATCH = (
    (EXPERT_ID_PREFIX, HIRE_TOOL, "template_id"),
    (TEAMMATE_ID_PREFIX, DELEGATE_TOOL, "expert_id"),
)


def expert_entries(
    templates: "Iterable[ExpertTemplate]", hired: "Iterable[Expert]"
) -> list[CapabilityEntry]:
    """Templates the user has not hired, then the hired experts themselves:
    re-hiring an active template is a no-op, so it is offered once, as the
    teammate."""
    hired = list(hired)
    hired_templates = {expert.source_template_id for expert in hired}
    return [
        *(
            _entry(
                template,
                hired=False,
                skills=[skill.title for skill in template.bundled_skills],
            )
            for template in templates
            if template.id not in hired_templates
        ),
        *(_entry(expert, hired=True, skills=[]) for expert in hired),
    ]


def expert_dispatch(capability_id: str) -> tuple[str, dict[str, str]] | None:
    """The tool an expert or teammate id runs as and the argument it binds,
    or None for any other id."""
    key = capability_id.strip()
    for prefix, tool, argument in _DISPATCH:
        if key.lower().startswith(prefix):
            ref = key[len(prefix) :].strip()
            return (tool, {argument: ref}) if ref else None
    return None


def _entry(expert: "Expert", *, hired: bool, skills: list[str]) -> CapabilityEntry:
    prefix, tool = (
        (TEAMMATE_ID_PREFIX, DELEGATE_TOOL)
        if hired
        else (
            EXPERT_ID_PREFIX,
            HIRE_TOOL,
        )
    )
    title = expert.job_title or expert.role
    workflows = ", ".join(w.name for w in expert.workflows if w.name)
    # Skill titles say what the expert does in the words a request uses; the
    # bio is left out because its prose put experts on unrelated searches.
    description = " ".join(
        part
        for part in (
            expert.role,
            expert.job_title,
            expert.tagline,
            f"Workflows: {workflows}." if workflows else "",
            f"Skills: {', '.join(skills)}." if skills else "",
        )
        if part
    )
    return CapabilityEntry(
        id=f"{prefix}{expert.id}",
        kind="expert",
        klass="service",
        name=expert.name,
        purpose=clip_purpose(f"{title}: {expert.tagline}" if expert.tagline else title),
        description=normalize_text(description),
        tags=[*expert.categories, "expert", "hired" if hired else "hire"],
        context="direct",
        implementations=[
            Implementation(kind="tool", ref=tool, name=tool, context="direct")
        ],
        hired=hired,
    )
