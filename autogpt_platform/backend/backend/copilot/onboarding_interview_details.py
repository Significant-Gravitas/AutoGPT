"""What an onboarding interview has heard so far, as the ending screens show
it: one small structured extraction, shared by the voice relay (live, from
the captions, and after the call, from the transcript of record) and the
REST API (the typed interview).

A cheap model reads the conversation and returns the person's business, role,
team, where their week goes, the tools they use, a few facts each tied to a
verbatim quote of their own words, which of the five interview topics they
have covered, and the first task their team starts on. The quotes are
checked here, in code: a fact whose quote is not in the person's own words is
dropped, so a screen never shows the person a fact as theirs that they did
not say.

This module imports neither the relay's stack nor the onboarding pipeline:
only the model's client types and pydantic.
"""

import json
import logging
import re
import time
from typing import Literal

from google.genai import Client, types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

# The five things the interview asks about, in its order.
Topic = Literal["work", "team", "week", "first_task", "tools"]
TOPICS: tuple[Topic, ...] = ("work", "team", "week", "first_task", "tools")

MAX_FACTS = 8
MAX_ITEMS = 6
MAX_FIELD_CHARS = 200
MAX_QUOTE_CHARS = 240
MAX_WORDS_LINES = 40
MAX_WORDS_LINE_CHARS = 1_000
# How much of the conversation one extraction reads, from its end.
MAX_CONVERSATION_CHARS = 12_000


class Fact(BaseModel):
    """Something the person said about their work, and their words for it."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(max_length=MAX_FIELD_CHARS)
    quote: str = Field(max_length=MAX_QUOTE_CHARS)


class InterviewDetails(BaseModel):
    """What the interview has heard, for the ending screens."""

    model_config = ConfigDict(frozen=True)

    business: str | None = None
    role: str | None = None
    team: str | None = None
    time_sinks: list[str] = []
    tools: list[str] = []
    facts: list[Fact] = []
    topics: list[Topic] = []
    first_task: str | None = None
    # The person's own lines the facts were checked against, for the screen
    # that shows their words with the facts marked; left out of a live read.
    words: list[str] = []


SCHEMA = {
    "type": "object",
    "properties": {
        "business": {"type": "string"},
        "role": {"type": "string"},
        "team": {"type": "string"},
        "time_sinks": {"type": "array", "items": {"type": "string"}},
        "tools": {"type": "array", "items": {"type": "string"}},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["label", "quote"],
            },
        },
        "topics": {"type": "array", "items": {"type": "string", "enum": list(TOPICS)}},
        "first_task": {"type": "string"},
    },
    "required": ["time_sinks", "tools", "facts", "topics"],
}

# While the interview runs only the card fills in, so a live read asks for no
# quoted facts: a shorter answer, sooner.
LIVE_SCHEMA = {
    **SCHEMA,
    "properties": {
        name: spec for name, spec in SCHEMA["properties"].items() if name != "facts"
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
{facts_rule}- "topics": which of these the USER has spoken to: "work" (what they do), "team" (who they work with), "week" (where their time goes), "first_task" (the first thing they would hand off), "tools" (the tools they use).
- "first_task": the first thing the USER wants their team to take on, in a short phrase, if the USER said or agreed to one. Omit otherwise.

Use the USER's own names for businesses, products and tools. Leave a field out rather than guess.

CONVERSATION:
{conversation}
"""


FACTS_RULE = """\
- "facts": up to eight facts. Each has "label", a short phrase in plain words, and "quote", the exact words the USER said that state it, copied character for character from a USER line, at most twenty words. Never paraphrase a quote.
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


def _normalised(text: str) -> str:
    """*text* for matching a quote: lower case, curly quotes made straight,
    and runs of anything but letters, digits and apostrophes as one space."""
    straight = text.replace("’", "'").replace("‘", "'").lower()
    return " ".join(re.sub(r"[^\w']+", " ", straight).split())


def quote_in_words(quote: str, words: str) -> bool:
    """Whether *quote* is said in *words*, word for word: case, punctuation
    and spacing aside, never a paraphrase."""
    needle = _normalised(quote)
    return bool(needle) and f" {needle} " in f" {_normalised(words)} "


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


def parse_details(text: str | None, person_words: str) -> InterviewDetails | None:
    """The model's answer as details, every fact's quote checked against
    *person_words*; None when the answer cannot be read."""
    try:
        answer = json.loads(text or "")
    except ValueError:
        return None
    if not isinstance(answer, dict):
        return None
    facts: list[Fact] = []
    for raw in answer.get("facts") or []:
        if not isinstance(raw, dict):
            continue
        label = _clean(raw.get("label"))
        quote = _clean(raw.get("quote"), MAX_QUOTE_CHARS)
        if not label or not quote or not quote_in_words(quote, person_words):
            continue
        facts.append(Fact(label=label, quote=quote))
    topics = [topic for topic in TOPICS if topic in (answer.get("topics") or [])]
    try:
        return InterviewDetails(
            business=_clean(answer.get("business")),
            role=_clean(answer.get("role")),
            team=_clean(answer.get("team")),
            time_sinks=_clean_list(answer.get("time_sinks")),
            tools=_clean_list(answer.get("tools")),
            facts=facts[:MAX_FACTS],
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
    live: bool = False,
) -> DetailsRead:
    """Read the details from the conversation *lines*, each ``("user", ...)``
    or ``("otto", ...)``, with the cheap model *model*; a *live* read, while
    the interview runs, asks for no quoted facts. Raises what the client
    raises; a caller drops the update then."""
    started = time.monotonic()
    person_words = "\n".join(said for who, said in lines if who == "user")
    response = await client.aio.models.generate_content(
        model=model,
        contents=PROMPT.format(
            conversation=conversation_text(lines),
            facts_rule="" if live else FACTS_RULE,
        ),
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_json_schema=LIVE_SCHEMA if live else SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_level=thinking),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )
    usage = response.usage_metadata
    details = parse_details(response.text, person_words)
    if details is not None and not live:
        details = details.model_copy(
            update={
                "words": [
                    said.strip()[:MAX_WORDS_LINE_CHARS]
                    for who, said in lines
                    if who == "user" and said.strip()
                ][-MAX_WORDS_LINES:]
            }
        )
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
