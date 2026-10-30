"""Parse a background LLM call's structured answer into its Pydantic model.

``complete.py`` makes the call, and the dream's
batch callbacks read their result rows; this module turns the text either
gets back into the typed value, recovering from the ways models wrap their
JSON:

  * **Markdown fences** — ``` ```json ... ``` ``` shows up on some
    OpenRouter upstreams even with ``force_json_output``.
  * **Prose prefixes** — a model that opens with chain-of-thought
    ("Looking at the inputs, I need to ... {...}") has its first balanced
    JSON object or array extracted. The system prompts in ``prompts.py``
    cut these down; this is the parser-level safety net.

Every failure is an ``InferenceError`` carrying the call's usage: the
usage stays available for telemetry even when the caller absorbs the cost.
Its message and code never quote the model answer; only a debug log may show its start.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .context import InferenceError, InferenceUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# How much of an answer that cannot be parsed a debug log shows.
_EXCERPT_CHARS = 200


def parse_structured_output(
    content: str, response_model: type[T], usage: InferenceUsage
) -> T:
    """*content* validated into *response_model*.

    Raises ``InferenceError`` carrying *usage* when the content is empty,
    is not JSON even after fence and prose recovery, or does not fit the
    schema.
    """
    if not content:
        raise InferenceError("LLM returned empty content", usage, code="empty_content")
    try:
        return response_model.model_validate(parse_json_with_prose_fallback(content))
    except InferenceError as exc:
        exc.usage = usage
        raise
    except ValidationError as exc:
        logger.debug("LLM JSON did not match %s: %s", response_model.__name__, exc)
        raise InferenceError(
            f"LLM JSON did not match {response_model.__name__} "
            f"({exc.error_count()} errors)",
            usage,
            code="schema_mismatch",
        ) from exc


def parse_json_with_prose_fallback(content: str) -> object:
    """Parse JSON from a model response, recovering from common preamble bugs.

    Two layers of defense, in order:

    1. Strip a leading ``` ```json ... ``` ``` markdown fence (some
       OpenRouter upstreams still emit them even with json_object mode).
    2. If ``json.loads`` fails, walk the string looking for the first
       balanced ``{...}`` / ``[...]`` and re-parse that — handles the
       "Looking at the inputs, I need to..." prose-prefix case.

    Raises ``InferenceError`` if neither layer recovers a parseable
    document; a debug log shows the start of the offending content.
    """
    cleaned = _strip_json_code_fence(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    extracted = _extract_first_json_object(cleaned)
    if extracted is None:
        logger.debug("LLM returned non-JSON content: %s", content[:_EXCERPT_CHARS])
        raise InferenceError("LLM returned non-JSON content", code="non_json_content")
    try:
        return json.loads(extracted)
    except json.JSONDecodeError as exc:
        logger.debug(
            "LLM returned non-JSON content even after extraction (%s): %s",
            exc,
            content[:_EXCERPT_CHARS],
        )
        raise InferenceError(
            "LLM returned non-JSON content even after extraction",
            code="non_json_content",
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
