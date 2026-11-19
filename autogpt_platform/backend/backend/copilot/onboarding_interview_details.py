"""What an onboarding interview has heard so far, as the ending screens show
it: one small structured extraction, shared by the voice relay (live, from
the captions, and after the call, from the transcript of record) and the
REST API (the typed interview).

A cheap model reads the conversation and returns the person's business, role,
team, where their week goes, the tools they use, which of the five interview
topics they have covered, and the first task they would hand off.

This module imports neither the relay's stack nor the onboarding pipeline:
only the model's client types and pydantic.
"""

import json
import logging
import time
from typing import Literal

from google.genai import Client, types
from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)

# The five things the interview asks about, in its order.
Topic = Literal["work", "team", "week", "first_task", "tools"]
TOPICS: tuple[Topic, ...] = ("work", "team", "week", "first_task", "tools")

MAX_ITEMS = 6
MAX_FIELD_CHARS = 200
# How much of the conversation one extraction reads, from its end.
MAX_CONVERSATION_CHARS = 12_000


class InterviewDetails(BaseModel):
    """What the interview has heard, for the ending screens."""

    model_config = ConfigDict(frozen=True)

    business: str | None = None
    role: str | None = None
    team: str | None = None
    time_sinks: list[str] = []
    tools: list[str] = []
    topics: list[Topic] = []
    first_task: str | None = None


SCHEMA = {
    "type": "object",
    "properties": {
        "business": {"type": "string"},
        "role": {"type": "string"},
        "team": {"type": "string"},
        "time_sinks": {"type": "array", "items": {"type": "string"}},
        "tools": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string", "enum": list(TOPICS)}},
        "first_task": {"type": "string"},
    },
    "required": ["time_sinks", "tools", "topics"],
}

PROMPT = """\
You read an onboarding interview between Otto, an AI, and a USER who is telling Otto about their work. Extract only what the USER said about their own work. Never use what Otto suggested unless the USER agreed to it in their own words.

Return JSON:
- "business": what their business or work is, in a few words. Omit if not said.
- "role": their role, in a few words. Omit if not said.
- "team": who they work with, in a few words. Omit if not said.
- "time_sinks": up to six things that take up their week, a few words each.
- "tools": up to six tools, apps or services they named, as they named them.
- "topics": which of these the USER has spoken to: "work" (what they do), "team" (who they work with), "week" (where their time goes), "first_task" (the first thing they would hand off), "tools" (the tools they use).
- "first_task": the first thing the USER wants their team to take on, in a short phrase, if the USER said or agreed to one. Omit otherwise.

Use the USER's own names for businesses, products and tools. Leave a field out rather than guess.

CONVERSATION:
{conversation}
"""


def conversation_text(lines: list[tuple[str, str]]) -> str:
    """The conversation as the prompt reads it, one ``OTTO:``/``USER:`` line
    per turn, its latest part when it is long."""
    text = "\n".join(
        f"{'USER' if who == 'user' else 'OTTO'}: {' '.join(said.split())}"
        for who, said in lines
        if said.strip()
    )
    return text[-MAX_CONVERSATION_CHARS:]


def _clean(text: object, limit: int = MAX_FIELD_CHARS) -> str | None:
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.split())[:limit]
    return cleaned or None


def _clean_list(items: object) -> list[str]:
    if not isinstance(items, list):
        return []
    kept: list[str] = []
    for item in items:
        cleaned = _clean(item)
        if cleaned and cleaned not in kept:
            kept.append(cleaned)
    return kept[:MAX_ITEMS]


def parse_details(text: str | None) -> InterviewDetails | None:
    """The model's answer as details; None when it cannot be read."""
    try:
        answer = json.loads(text or "")
    except ValueError:
        return None
    if not isinstance(answer, dict):
        return None
    topics: list[Topic] = [
        topic for topic in TOPICS if topic in (answer.get("topics") or [])
    ]
    try:
        return InterviewDetails(
            business=_clean(answer.get("business")),
            role=_clean(answer.get("role")),
            team=_clean(answer.get("team")),
            time_sinks=_clean_list(answer.get("time_sinks")),
            tools=_clean_list(answer.get("tools")),
            topics=topics,
            first_task=_clean(answer.get("first_task")),
        )
    except ValidationError:
        return None


class DetailsRead(BaseModel):
    """One extraction: the details, None when unreadable, and what it cost."""

    model_config = ConfigDict(frozen=True)

    details: InterviewDetails | None
    tokens_in: int
    tokens_out: int
    latency_s: float


async def extract_details(
    client: Client,
    *,
    model: str,
    lines: list[tuple[str, str]],
    thinking: types.ThinkingLevel = types.ThinkingLevel.MINIMAL,
) -> DetailsRead:
    """Read the details from the conversation *lines*, each ``("user", ...)``
    or ``("otto", ...)``, with the cheap model *model*. Raises what the
    client raises; a caller drops the update then."""
    started = time.monotonic()
    response = await client.aio.models.generate_content(
        model=model,
        contents=PROMPT.format(conversation=conversation_text(lines)),
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_json_schema=SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_level=thinking),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )
    usage = response.usage_metadata
    details = parse_details(response.text)
    return DetailsRead(
        details=details,
        tokens_in=(usage.prompt_token_count or 0) if usage else 0,
        tokens_out=(
            (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
            if usage
            else 0
        ),
        latency_s=round(time.monotonic() - started, 3),
    )
