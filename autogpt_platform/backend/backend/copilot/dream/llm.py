"""Pydantic-typed LLM entry point for the dream pass.

Thin wrapper on top of ``backend/util/llm/providers.call_provider``
that adds the dream-specific concerns the shared helper deliberately
doesn't own:

  * **Pydantic schema validation** — converts the LLM's JSON response
    into a typed instance of ``response_model`` and surfaces a clean
    ``DreamLLMError`` if the model emits a shape that doesn't fit.
  * **JSON prose-prefix recovery** — when a model wraps its JSON in
    chain-of-thought ("Looking at the inputs, I need to ... {...}")
    we extract the first balanced JSON object/array. The strengthened
    system prompts in ``prompts.py`` cut down on prose preambles, but
    this is the parser-level safety net.
  * **Markdown fence stripping** — ``` ```json ... ``` ``` shows up
    on some OpenRouter upstreams even with ``force_json_output``.
  * **CompletionUsage** carrier — the orchestrator rolls these per
    phase into a ``DreamPassUsage`` for the cost ledger.

Why OpenRouter and not direct Anthropic here:
  * One credential covers every model the dream config might want.
  * ``response_format={"type":"json_object"}`` is supported across
    OpenAI and OpenRouter, so the same code path covers cloud + local
    OpenAI-compat LLMs.
  * Step 4 of the rollout (see ``plans/idempotent-launching-moth.md``
    component E) plugs the Anthropic batch path in *below* this layer
    via ``call_provider(execution_mode="batch")``. This wrapper stays
    sync-only; the batch path lands on the orchestrator's branch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from backend.util.llm.providers import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ProviderResponse,
    call_provider,
)
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
_settings = Settings()

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
    api_key = _settings.secrets.open_router_api_key
    if not api_key:
        raise DreamLLMError("OpenRouter client unavailable — set OPEN_ROUTER_API_KEY")

    try:
        response = await call_provider(
            provider="open_router",
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
            force_json_output=True,
            timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
    except DreamLLMError:
        raise
    except Exception as exc:
        raise DreamLLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

    if not isinstance(response, ProviderResponse):
        # ``call_provider`` only returns a non-ProviderResponse when the
        # caller passed ``execution_mode="batch"`` (lands later). The
        # dream's sync wrapper never opts in, so anything else is a bug.
        raise DreamLLMError(
            "structured_completion expected a sync ProviderResponse but got a "
            f"{type(response).__name__} — execution_mode must stay 'sync' here."
        )

    usage = _usage_from_provider_response(response, model)

    content = (response.content or "").strip()
    if not content:
        raise DreamLLMError("LLM returned empty content")

    payload = _parse_json_with_prose_fallback(content)

    try:
        return StructuredCompletion(
            value=response_model.model_validate(payload),
            usage=usage,
        )
    except ValidationError as exc:
        raise DreamLLMError(
            f"LLM JSON did not match {response_model.__name__}: {exc}"
        ) from exc


def _usage_from_provider_response(
    response: ProviderResponse, model: str
) -> CompletionUsage:
    """Convert a ``ProviderResponse`` into the dream's ``CompletionUsage``."""
    return CompletionUsage(
        model=model,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_creation_tokens=response.cache_creation_tokens,
        cost_usd=response.cost_usd,
    )


def _parse_json_with_prose_fallback(content: str) -> object:
    """Parse JSON from a model response, recovering from common preamble bugs.

    Two layers of defense, in order:

    1. Strip a leading ``` ```json ... ``` ``` markdown fence (some
       OpenRouter upstreams still emit them even with json_object mode).
    2. If ``json.loads`` fails, walk the string looking for the first
       balanced ``{...}`` / ``[...]`` and re-parse that — handles the
       "Looking at the inputs, I need to..." prose-prefix case.

    Raises ``DreamLLMError`` if neither layer recovers a parseable
    document, surfacing the first 200 chars of the offending content so
    the orchestrator can log enough to debug what the model emitted.
    """
    cleaned = _strip_json_code_fence(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    extracted = _extract_first_json_object(cleaned)
    if extracted is None:
        raise DreamLLMError(
            f"LLM returned non-JSON content — first 200 chars: {content[:200]}"
        )
    try:
        return json.loads(extracted)
    except json.JSONDecodeError as exc:
        raise DreamLLMError(
            f"LLM returned non-JSON content even after extraction: {exc} — "
            f"first 200 chars: {content[:200]}"
        ) from exc


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

    Even with ``response_format={"type":"json_object"}``, some OpenRouter
    upstreams (notably non-OpenAI models) still wrap output in markdown
    fences. Strip them defensively so the parser sees pure JSON.
    """
    stripped = content.strip()
    if not stripped.startswith("```"):
        return content
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return content
    body = stripped[first_newline + 1 :]
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()
