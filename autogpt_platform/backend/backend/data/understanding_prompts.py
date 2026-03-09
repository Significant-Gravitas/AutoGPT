"""Helpers for generating quick prompts from saved business understanding."""

import asyncio
import json
import logging

from openai import AsyncOpenAI

from backend.data.understanding import (
    BusinessUnderstanding,
    format_understanding_for_prompt,
)
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

_LLM_TIMEOUT = 30

_PROMPTS_PROMPT = """\
You generate three short starter prompts for a user to click in a chat UI.

Return a JSON object with this exact shape:
{"prompts":["...","...","..."]}

Requirements:
- Exactly 3 prompts
- Each prompt must be written in first person, as if the user is speaking
- Each prompt must be shorter than 20 words
- Keep them specific to the user's business context
- Do not number the prompts
- Do not add labels or explanations

Business context:
"""

_PROMPTS_SUFFIX = "\n\nReturn ONLY valid JSON."


def has_prompt_generation_context(understanding: BusinessUnderstanding) -> bool:
    return bool(format_understanding_for_prompt(understanding).strip())


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.split())


def _validate_prompts(value: object) -> list[str]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Prompt response must contain exactly three prompts")

    prompts: list[str] = []
    seen: set[str] = set()

    for item in value:
        if not isinstance(item, str):
            raise ValueError("Each prompt must be a string")

        prompt = _normalize_prompt(item)
        if not prompt:
            raise ValueError("Prompts cannot be empty")
        if len(prompt.split()) >= 20:
            raise ValueError("Prompts must be fewer than 20 words")
        if prompt in seen:
            raise ValueError("Prompts must be unique")

        seen.add(prompt)
        prompts.append(prompt)

    return prompts


async def generate_understanding_prompts(
    understanding: BusinessUnderstanding,
) -> list[str]:
    """Generate validated quick prompts from a saved understanding snapshot."""
    context = format_understanding_for_prompt(understanding)
    if not context.strip():
        raise ValueError("Understanding does not contain usable context")

    settings = Settings()
    client = AsyncOpenAI(
        api_key=settings.secrets.open_router_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"{_PROMPTS_PROMPT}{context}{_PROMPTS_SUFFIX}",
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            ),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Understanding prompts: generation timed out")
        raise

    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Understanding prompts: invalid JSON response")
        raise

    if not isinstance(data, dict):
        raise ValueError("Prompt response must be a JSON object")

    return _validate_prompts(data.get("prompts"))
