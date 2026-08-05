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

from backend.api.features.integrations.models import (
    get_all_provider_names,
    get_provider_description,
)
from backend.api.features.onboarding_dump.intro import _parse_response_json
from backend.api.features.onboarding_dump.models import RecommendedProvider
from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)

_MODEL = os.environ.get("BRAIN_DUMP_RECOMMEND_MODEL", "anthropic/claude-sonnet-5")
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

    known = _known_providers()
    prompt = _PROMPT.format(
        max_recommendations=MAX_RECOMMENDATIONS,
        providers="\n".join(
            f"- {name}: {description}" if description else f"- {name}"
            for name, description in known.items()
        ),
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

    data = _parse_response_json(response.choices[0].message.content or "")
    return _parse_recommendations(data, set(known))


def _known_providers() -> dict[str, str | None]:
    """The live provider registry as ``{id: description}``.

    Mirrors the ``/providers`` endpoint: block modules must be imported
    before AutoRegistry knows about SDK-registered providers.
    """
    try:
        from backend.blocks import load_all_blocks

        load_all_blocks()
    except Exception as e:  # static providers still work
        logger.warning("Brain dump recommendations: block load failed: %s", e)
    return {name: get_provider_description(name) for name in get_all_provider_names()}


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
