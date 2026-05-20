"""LLM helper used by the three-phase dream orchestrator.

Wraps ``get_openai_client(prefer_openrouter=True)`` with a structured
JSON-output convention: pass a Pydantic model class as ``response_model``,
get back a parsed, validated instance. If the model returns invalid
JSON we surface the raw content + parse error so the orchestrator can
log it and skip the phase rather than crashing the whole pass.

Why OpenRouter and not the direct Anthropic SDK at this layer:
  * One client; works across every model the user has configured.
  * ``response_format={"type":"json_object"}`` is supported by both
    OpenAI and OpenRouter, so the same code path covers cloud and
    local LLMs that proxy through the OpenAI-compat schema.
  * The Anthropic batch path (P-0.1 follow-up) plugs in below this
    layer in its own module; this file stays sync-only.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class DreamLLMError(RuntimeError):
    """Raised when a dream-pass LLM call cannot be parsed into the target schema."""


async def structured_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
) -> T:
    """Call the LLM in JSON mode and parse into ``response_model``.

    Raises ``DreamLLMError`` if the response is empty, unparseable, or
    fails Pydantic validation. Callers should treat that as "this
    phase failed" — the orchestrator either skips downstream phases
    or returns a partial result.
    """
    client = get_openai_client(prefer_openrouter=True)
    if client is None:
        raise DreamLLMError("OpenRouter client unavailable — set OPEN_ROUTER_API_KEY")

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise DreamLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise DreamLLMError("LLM returned empty content")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise DreamLLMError(
            f"LLM returned non-JSON content: {exc} — first 200 chars: "
            f"{content[:200]}"
        ) from exc

    try:
        return response_model.model_validate(payload)
    except ValidationError as exc:
        raise DreamLLMError(
            f"LLM JSON did not match {response_model.__name__}: {exc}"
        ) from exc
