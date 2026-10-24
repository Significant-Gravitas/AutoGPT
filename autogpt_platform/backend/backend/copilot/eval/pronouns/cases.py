"""Cases for the user-pronoun check, and the check itself.

Each case is the first message of a real session, built from production's own
pieces: the ``<user_context>`` block the owner's profile renders into, the team
roster Otto reads, and for an expert the preamble ``delegate_to_expert``
prepends. Only the text under check is deterministic; producing it takes a
live model (``runner.py``).
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from backend.api.features.experts.models import PROTECTED_SOUL_RULES, Expert
from backend.copilot.eval.style.assembly import user_prefix
from backend.copilot.service import format_user_context_prefix
from backend.copilot.tools.delegate_to_expert import _handoff_message
from backend.copilot.tools.expert_delegation import first_name
from backend.data.understanding import (
    BusinessUnderstanding,
    format_understanding_for_prompt,
)

CASES_PATH = Path(__file__).resolve().parent / "cases.json"

# Every English pronoun that gives a person a gender. In the text under check
# the user is the only third person, so a match is a guess about the user.
GENDERED_PRONOUN = re.compile(
    r"\b(he|she|him|her|hers|his|himself|herself)\b", re.IGNORECASE
)

CaseKind = Literal["otto_brief", "expert_task"]


class Teammate(BaseModel):
    name: str
    role: str
    identity: str


class Persona(BaseModel):
    id: str
    user_name: str | None
    business_name: str
    industry: str
    user_role: str


class CaseFile(BaseModel):
    teammate: Teammate
    personas: list[Persona]
    otto_briefs: list[str]
    expert_tasks: list[str]


class PronounCase(BaseModel):
    id: str
    kind: CaseKind
    persona: Persona
    text: str


def load_case_file(path: Path = CASES_PATH) -> CaseFile:
    return CaseFile.model_validate_json(path.read_text(encoding="utf-8"))


def expand_cases(case_file: CaseFile) -> list[PronounCase]:
    """Every persona against every brief and every task."""
    groups: list[tuple[CaseKind, list[str]]] = [
        ("otto_brief", case_file.otto_briefs),
        ("expert_task", case_file.expert_tasks),
    ]
    return [
        PronounCase(id=f"{persona.id}/{kind}-{i}", kind=kind, persona=persona, text=t)
        for persona in case_file.personas
        for kind, texts in groups
        for i, t in enumerate(texts, start=1)
    ]


def teammate_expert(teammate: Teammate) -> Expert:
    """The teammate as a hired expert, with production's protected rules."""
    return Expert(
        id=f"pronoun-eval-{teammate.name.lower()}",
        name=teammate.name,
        avatar_url=None,
        role=teammate.role,
        tagline=None,
        bio=None,
        skills=[],
        identity=teammate.identity,
        voice_preferences="Plain and direct.",
        boundaries="",
        protected_soul_rules=list(PROTECTED_SOUL_RULES),
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    )


def first_message(case: PronounCase, teammate: Expert) -> str:
    """The session's first user message as production stores it."""
    roster = [teammate]
    if case.kind == "otto_brief":
        return user_prefix(None, roster) + _user_context(case.persona) + case.text
    task = _handoff_message("Otto", "", case.text, first_name(case.persona.user_name))
    return user_prefix(teammate, roster) + _user_context(case.persona) + task


def gendered_pronouns(text: str) -> list[str]:
    """Each gendered pronoun in ``text``, in order, as written."""
    return [m.group(0) for m in GENDERED_PRONOUN.finditer(text)]


def _user_context(persona: Persona) -> str:
    now = datetime.now(UTC)
    understanding = BusinessUnderstanding(
        id="pronoun-eval",
        user_id="pronoun-eval",
        created_at=now,
        updated_at=now,
        user_name=persona.user_name,
        business_name=persona.business_name,
        industry=persona.industry,
        user_role=persona.user_role,
    )
    return format_user_context_prefix(format_understanding_for_prompt(understanding))
