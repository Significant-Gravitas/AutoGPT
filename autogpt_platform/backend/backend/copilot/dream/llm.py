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
import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class DreamLLMError(RuntimeError):
    """Raised when a dream-pass LLM call cannot be parsed into the target schema."""


@dataclass(slots=True)
class CompletionUsage:
    """Token + cost telemetry from a single LLM call.

    Carries the provider-reported cost when present (OpenRouter
    surfaces ``usage.cost`` when the request asks for ``usage:
    {"include": True}``); falls back to None when absent. Token counts
    are always present.
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None


@dataclass(slots=True)
class StructuredCompletion(Generic[T]):
    """Return value of ``structured_completion``: parsed model + usage."""

    value: T
    usage: CompletionUsage


async def structured_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
) -> StructuredCompletion[T]:
    """Call the LLM in JSON mode and parse into ``response_model``.

    Returns a ``StructuredCompletion`` carrying both the parsed value
    and a ``CompletionUsage`` block so the dream orchestrator can roll
    token counts + cost up into a ``DreamPassUsage``.

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
            # Ask OpenRouter to surface real generation cost on the usage
            # block; OpenAI/Anthropic-direct ignore the field, so it's
            # additive and safe.
            extra_body={"usage": {"include": True}},
        )
    except Exception as exc:
        raise DreamLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

    usage = _extract_usage(response, model)

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise DreamLLMError("LLM returned empty content")

    content = _strip_json_code_fence(content)

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        # Some models prefix the JSON with chain-of-thought prose ("I'll
        # analyze the proposals..."). Try to recover by extracting the
        # first balanced JSON object/array.
        extracted = _extract_first_json_object(content)
        if extracted is None:
            raise DreamLLMError(
                f"LLM returned non-JSON content — first 200 chars: {content[:200]}"
            )
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise DreamLLMError(
                f"LLM returned non-JSON content even after extraction: {exc} — "
                f"first 200 chars: {content[:200]}"
            ) from exc

    try:
        return StructuredCompletion(
            value=response_model.model_validate(payload),
            usage=usage,
        )
    except ValidationError as exc:
        raise DreamLLMError(
            f"LLM JSON did not match {response_model.__name__}: {exc}"
        ) from exc


def _extract_usage(response, model: str) -> CompletionUsage:
    """Pull tokens + provider-reported cost off an OpenAI-compat response.

    Provider quirks:
      * OpenRouter returns ``cost`` on ``usage.model_extra`` when the
        request body sets ``usage: {"include": True}``.
      * Direct OpenAI / Anthropic don't return cost — callers compute
        from the rate card in ``model_pricing.py``.
      * Cache token names differ: ``cache_write_tokens`` (OpenRouter
        Anthropic) vs ``cache_creation_input_tokens`` (Anthropic direct).
    """
    usage = response.usage
    if usage is None:
        return CompletionUsage(model=model)

    prompt_tokens = int(usage.prompt_tokens or 0)
    completion_tokens = int(usage.completion_tokens or 0)

    cache_read = 0
    cache_creation = 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        cache_read = int(getattr(ptd, "cached_tokens", 0) or 0)
        ptd_extras = getattr(ptd, "model_extra", None) or {}
        cache_creation = int(
            ptd_extras.get("cache_write_tokens")
            or ptd_extras.get("cache_creation_input_tokens")
            or 0
        )

    cost_usd: float | None = None
    extras = getattr(usage, "model_extra", None) or {}
    raw_cost = extras.get("cost")
    if raw_cost is not None:
        try:
            val = float(raw_cost)
            if math.isfinite(val) and val >= 0:
                cost_usd = val
        except (TypeError, ValueError):
            logger.warning("dream usage.cost non-numeric: %r", raw_cost)

    return CompletionUsage(
        model=model,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        cost_usd=cost_usd,
    )


def _extract_first_json_object(content: str) -> str | None:
    """Find the first balanced JSON object or array in ``content``.

    Used as a fallback when the model wraps JSON in chain-of-thought
    prose ("I'll analyze...\n\n{...}"). Returns ``None`` if no balanced
    structure can be found. Naive but good enough: we scan for the
    first ``{`` or ``[``, then walk forward counting braces/brackets
    while respecting string literals so braces inside strings don't
    throw off the count.
    """
    n = len(content)
    start = -1
    for i, ch in enumerate(content):
        if ch in "{[":
            start = i
            break
    if start == -1:
        return None
    open_ch = content[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, n):
        ch = content[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return content[start : i + 1]
    return None


def _strip_json_code_fence(content: str) -> str:
    """Strip ```json ...``` or ``` ... ``` fences if the model added them.

    Even with response_format={"type":"json_object"}, some OpenRouter
    upstreams (notably non-OpenAI models) still wrap output in markdown
    fences. Strip them defensively so the parser sees pure JSON.
    """
    stripped = content.strip()
    if not stripped.startswith("```"):
        return content
    # Drop the opening fence (with or without "json" tag) and any trailing fence.
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return content
    body = stripped[first_newline + 1 :]
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()
