"""The soul a newly raised expert writes from its opening answers.

Runs as a background task after the user answers the expert's first
questions. The fast model writes the identity and boundaries; if it is
missing or fails, a template records the answers verbatim so they still
reach every later turn. Either way the write is compare-and-set against the
soul the request saw, so an edit the user made meanwhile is never lost.
"""

import logging
import os

from openai import OpenAIError
from pydantic import BaseModel, Field

from backend.api.features.experts import experts_db
from backend.api.features.experts.models import EXPERT_IDENTITY_MAX_LENGTH, Expert
from backend.api.features.onboarding_dump.structured import request_json
from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)

_MODEL = os.environ.get("RAISE_SOUL_MODEL", "anthropic/claude-haiku-5.5")
_TIMEOUT_SECONDS = 30
MAX_IDENTITY_CHARS = 2_000
MAX_BOUNDARIES_CHARS = 600

_PROMPT = """You are writing the standing instructions ("soul") for a new \
teammate the user just created. It reads these at the start of every task.

Name: {name}
Role: {role}
What it said its job is: {intro}
The user's answers to its setup questions:
{answers}

Return ONLY valid JSON with exactly two keys:
- "identity": first person, 80 to 200 words. Who it is, what it is \
responsible for, the project or system it works on, how it works and \
reports back, all grounded in the answers. No greeting, no dashes.
- "boundaries": first person, one to three short sentences on what it must \
check with the user before doing, taken from the answers. Empty string if \
the answers say nothing about it.

Never use the words "agent", "graph" or "assistant". Call automations \
"workflows". Do not invent facts the answers do not give.
"""


class SoulAnswer(BaseModel):
    question: str = Field(min_length=1, max_length=200)
    answer: str = Field(min_length=1, max_length=500)


class SoulDraftRequest(BaseModel):
    intro: str = Field(default="", max_length=500)
    answers: list[SoulAnswer] = Field(min_length=1, max_length=5)


class SoulDraftQueued(BaseModel):
    # The identity the write compares against; the client polls until the
    # expert's identity moves off it.
    expected_identity: str


class SoulDraft(BaseModel):
    identity: str
    boundaries: str


async def write_soul(user_id: str, expert: Expert, request: SoulDraftRequest) -> None:
    draft = await generate_soul(expert, request) or fallback_soul(expert, request)
    written = await experts_db.update_soul_fields_if_current(
        user_id,
        expert.id,
        identity=draft.identity[:EXPERT_IDENTITY_MAX_LENGTH],
        boundaries=draft.boundaries,
        expected_identity=expert.identity,
        expected_boundaries=expert.boundaries,
    )
    if not written:
        logger.info(f"Expert #{expert.id} soul changed before the draft landed")


async def generate_soul(expert: Expert, request: SoulDraftRequest) -> SoulDraft | None:
    client = get_openai_client(prefer_openrouter=True)
    if client is None:
        return None
    prompt = _PROMPT.format(
        name=expert.name,
        role=expert.job_title or "not given",
        intro=request.intro or "not given",
        answers=_answer_lines(request),
    )
    try:
        reply = await request_json(
            client,
            model=_MODEL,
            prompt=prompt,
            temperature=0.3,
            max_tokens=1_200,
            timeout=_TIMEOUT_SECONDS,
        )
    except (OpenAIError, TimeoutError) as e:  # the template records them
        logger.warning(f"Raise soul generation failed: {e}")
        return None
    data = reply.data or {}
    identity = data.get("identity")
    boundaries = data.get("boundaries")
    if not isinstance(identity, str) or not identity.strip():
        logger.warning(f"Raise soul: unusable output ({reply.describe()})")
        return None
    return SoulDraft(
        identity=identity.strip()[:MAX_IDENTITY_CHARS],
        boundaries=(
            boundaries.strip()[:MAX_BOUNDARIES_CHARS]
            if isinstance(boundaries, str)
            else expert.boundaries
        ),
    )


def fallback_soul(expert: Expert, request: SoulDraftRequest) -> SoulDraft:
    opening = f"I'm {expert.name}" + (
        f", the {expert.job_title} on your team." if expert.job_title else "."
    )
    lines = [opening]
    if request.intro:
        lines.append(request.intro)
    lines.append("")
    lines.append("What you told me when you created me:")
    lines.extend(f"- {line}" for line in _answer_lines(request).splitlines())
    return SoulDraft(identity="\n".join(lines), boundaries=expert.boundaries)


def _answer_lines(request: SoulDraftRequest) -> str:
    return "\n".join(
        f"{answer.question.strip()} {answer.answer.strip()}"
        for answer in request.answers
    )
