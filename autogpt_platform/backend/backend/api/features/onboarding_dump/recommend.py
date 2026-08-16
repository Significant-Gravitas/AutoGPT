"""Provider recommendations from the brain-dump transcript.

Runs as its own background job beside the greeting pipeline — the two are
deliberately decoupled so a slow or failed recommendation never delays the
greeting, and vice versa. The model sees the transcript plus the live
provider registry and picks the handful worth connecting first; anything
it hallucinates outside the registry is dropped.
"""

import asyncio
import logging
import os

from backend.api.features.onboarding_dump.models import RecommendedProvider
from backend.api.features.onboarding_dump.parsing import parse_response_json
from backend.api.features.onboarding_dump.providers import (
    known_providers,
    provider_lines,
)
from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)

# Picking six ids off a list is a matching task, not a writing one, and
# the onboarding loading screen holds the user until it answers — so this
# runs on the fast model rather than the one that writes the greeting.
_MODEL = os.environ.get("BRAIN_DUMP_RECOMMEND_MODEL", "anthropic/claude-haiku-4-5")
_TIMEOUT_SECONDS = 30

MAX_RECOMMENDATIONS = 6
MAX_REASON_CHARS = 140

_PROMPT = """A new user just recorded a spoken brain dump about their work. \
Below is their transcript and the list of integration providers this \
platform can connect to.

Pick the providers this specific user should connect first, based only on \
tools and workflows they actually mentioned or clearly implied. Return ONLY \
valid JSON with exactly one key, "providers": an array of at most \
{max_recommendations} objects, most useful first, each with:
- "provider": a provider id copied EXACTLY from the list below
- "reason": one short sentence (max 100 characters), second person, tying \
the provider to something they said

Rules: never invent a provider id that is not in the list; skip generic \
picks the transcript gives no evidence for; an empty array is a valid \
answer for a thin transcript.

Providers:
{providers}

Transcript:
{transcript}
"""


async def generate_recommendations(transcript: str) -> list[RecommendedProvider]:
    """Return the model's provider picks for ``transcript``.

    Never raises: any failure returns an empty list, which the caller
    still persists — "no recommendations" is a valid, final result.
    """
    text = transcript.strip()
    if not text:
        return []

    client = get_openai_client(prefer_openrouter=True)
    if client is None:
        logger.warning("Brain dump recommendations: no LLM client configured")
        return []

    known = known_providers()
    prompt = _PROMPT.format(
        max_recommendations=MAX_RECOMMENDATIONS,
        providers=provider_lines(known),
        transcript=text,
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
            ),
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception as e:  # background job, empty list is fine
        logger.warning("Brain dump recommendation generation failed: %s", e)
        return []

    data = parse_response_json(response.choices[0].message.content or "")
    return _parse_recommendations(data, set(known))


def _parse_recommendations(data: object, known: set[str]) -> list[RecommendedProvider]:
    items = data.get("providers") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    seen: set[str] = set()
    recommendations = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("provider"), str):
            continue
        provider = item["provider"].strip()
        if provider not in known or provider in seen:
            continue
        seen.add(provider)
        reason = item.get("reason")
        recommendations.append(
            RecommendedProvider(
                provider=provider,
                reason=(
                    reason.strip()[:MAX_REASON_CHARS] if isinstance(reason, str) else ""
                ),
            )
        )
        if len(recommendations) == MAX_RECOMMENDATIONS:
            break
    return recommendations
