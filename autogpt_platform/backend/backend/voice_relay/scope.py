"""The call's scope guard: a cheap model labels each of the person's turns,
and the interview's own rule decides from the labels when the call ends.

The rule is the typed interview's (``onboarding_interview_scope.scope_verdict``)
so text and voice hold one line. The labels are this module's: a flash model
labels the person's turn from the live captions as soon as Otto starts
answering it, so the judgement runs beside the call and adds nothing to its
latency. A label that fails or cannot be read leaves the turn unjudged; the
caps still bound the call.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Protocol

from google import genai
from google.genai import types

from backend.copilot.onboarding_interview_scope import ScopeLabel
from backend.voice_relay.config import CLASSIFIER_MODEL

logger = logging.getLogger(__name__)

# The classifier's labels as the interview's rule reads them.
LABELS: dict[str, ScopeLabel] = {
    "work": "work",
    "task": "request",
    "harmful": "harmful",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "why": {"type": "string"},
    },
    "required": ["label", "why"],
}

PROMPT = """\
You label one turn of a voice onboarding call. Otto, an AI, is on the call only to learn about the user's work.

Label the USER turn:
- "work": the user talks about their own work, business, job search, team or tools; answers, corrects or confirms Otto; greets or says goodbye; asks how the call works or what happens with their information; or anything unclear.
- "task": the user asks Otto to do something now instead of describing their work, such as write, code, debug, research, give advice or answer a general question.
- "harmful": the user asks for help with something that harms someone else, such as getting into another person's account, tracking or exposing a person, or harassment; or the user abuses Otto.

If the turn mixes a request with real facts about their work, label it by what the user mainly asks for.
Return JSON: "label", and "why" in at most eight words.

OTTO BEFORE: {before}
USER: {user}
OTTO AFTER: {after}
"""


@dataclass(frozen=True)
class ScopeJudgement:
    """A label for one of the person's turns, and what it cost."""

    # None when the model's answer could not be read: the turn stays unjudged.
    label: ScopeLabel | None
    tokens_in: int
    tokens_out: int
    latency_s: float


class ScopeClassifier(Protocol):
    async def __call__(self, *, before: str, user: str) -> ScopeJudgement:
        """Label the person's turn *user*, which answered Otto's *before*."""
        ...


class GeminiScopeClassifier:
    def __init__(self, client: genai.Client, *, model: str = CLASSIFIER_MODEL) -> None:
        self._client = client
        self._model = model

    async def __call__(self, *, before: str, user: str) -> ScopeJudgement:
        started = time.monotonic()
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=PROMPT.format(
                before=before or "(nothing yet)", user=user, after="(nothing)"
            ),
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1024,
                response_mime_type="application/json",
                response_json_schema=SCHEMA,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                ),
            ),
        )
        usage = response.usage_metadata
        return ScopeJudgement(
            label=parse_label(response.text),
            tokens_in=(usage.prompt_token_count or 0) if usage else 0,
            tokens_out=(
                (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
                if usage
                else 0
            ),
            latency_s=round(time.monotonic() - started, 3),
        )


def parse_label(text: str | None) -> ScopeLabel | None:
    """The interview's label for the classifier's answer, or None if unreadable."""
    try:
        answer = json.loads(text or "")
    except ValueError:
        return None
    if not isinstance(answer, dict):
        return None
    return LABELS.get(str(answer.get("label")))
