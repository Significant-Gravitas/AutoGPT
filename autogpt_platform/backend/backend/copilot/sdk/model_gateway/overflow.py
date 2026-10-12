"""Turn an upstream's "the prompt does not fit" into Anthropic's exact wording.

CLI 2.1.284 compacts and retries on a 400 only when the message reads
``prompt is too long: N tokens > M maximum``; any other wording ends the turn
with "API Error: 400". No local server says it that way, so the rewrite is
what makes reactive compaction work at all off Anthropic's own API.

The wordings below are the ones the servers in scope actually send:
llama.cpp (``exceed_context_size_error``, with ``n_prompt_tokens``/``n_ctx``),
vLLM and OpenAI-style servers ("maximum context length is M tokens"),
Ollama ("prompt too long; exceeded max context length by K tokens"),
LM Studio ("Trying to keep the first N tokens when context the overflows"),
SGLang, TGI, Mistral, Gemini's OpenAI layer, LiteLLM's wrapper, and the
Codex backend's ``context_length_exceeded``.
"""

import re
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from backend.copilot.sdk.model_gateway.errors import (
    GatewayError,
    UpstreamError,
    error_for_status,
    redact,
)

_FLAGS = re.IGNORECASE | re.DOTALL
_NUMBERED = [
    re.compile(p, _FLAGS)
    for p in (
        r"prompt is too long:\s*(?P<prompt>\d+)\s*tokens\s*>\s*(?P<limit>\d+)",
        r"request \((?P<prompt>\d+) tokens\) exceeds the available context size"
        r" \((?P<limit>\d+) tokens\)",
        r"input \((?P<prompt>\d+) tokens\) is longer than the model's context"
        r" length \((?P<limit>\d+) tokens\)",
        r"maximum context length is (?P<limit>\d+) tokens.{0,40}?"
        r"(?:requested|resulted in|request has|messages resulted in) (?P<prompt>\d+)",
        r"prompt too long; exceeded (?:max )?context length by (?P<excess>\d+)",
        r"trying to keep the first (?P<prompt>\d+) tokens when context the"
        r" overflows(?:.{0,200}?context length of only (?P<limit>\d+) tokens)?",
        r"`inputs` tokens \+ `max_new_tokens` must be <= (?P<limit>\d+)\."
        r" Given: (?P<prompt>\d+) `inputs` tokens",
        r"prompt contains (?P<prompt>\d+) tokens.{0,80}?too large for model with"
        r" (?P<limit>\d+) maximum context length",
        r"input token count \((?P<prompt>\d+)\) exceeds the maximum number of"
        r" tokens allowed \((?P<limit>\d+)\)",
    )
]
# Deliberately not "maximum context length" alone: vLLM also says that when
# max_tokens by itself is too large, which no compaction can fix.
_UNNUMBERED = re.compile(
    r"context_length_exceeded|exceed_context_size_error|"
    r"exceeds the (?:available )?context (?:window|size)|"
    r"context ?window ?exceeded",
    _FLAGS,
)
_OVERFLOW_CODES = frozenset({"context_length_exceeded", "exceed_context_size_error"})


class ContextOverflow(BaseModel):
    """What the upstream said about the overflow; any number may be missing."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int | None = None
    limit: int | None = None
    excess: int | None = None


def detect_overflow(error: UpstreamError) -> ContextOverflow | None:
    """The overflow this error describes, or ``None`` if it is something else."""
    for pattern in _NUMBERED:
        if match := pattern.search(error.message):
            return _from_match(match)
    fields = _from_fields(error)
    if fields is not None:
        return fields
    if error.code in _OVERFLOW_CODES or error.error_type in _OVERFLOW_CODES:
        return ContextOverflow()
    if _UNNUMBERED.search(error.message):
        return ContextOverflow()
    return None


def upstream_failure(
    error: UpstreamError,
    *,
    status: int,
    estimate: int,
    window: int | None,
    secrets: Iterable[str],
    headers: Mapping[str, str] | None = None,
) -> GatewayError:
    """An upstream error restated for the CLI: an overflow in Anthropic's exact
    wording, anything else with its status kept and its message redacted."""
    overflow = detect_overflow(error)
    if overflow is not None:
        return prompt_too_long(overflow, estimate=estimate, window=window)
    return error_for_status(
        status,
        redact(error.message, secrets),
        error_type=error.error_type,
        headers=headers,
    )


def prompt_too_long(
    overflow: ContextOverflow, *, estimate: int, window: int | None
) -> GatewayError:
    """The 400 the CLI compacts on, with the best numbers available.

    The upstream's own figures win; the route's window and the gateway's
    estimate fill any gap. The prompt is always stated as larger than the
    limit, because the CLI sizes what it drops by the difference.
    """
    prompt, limit = _figures(overflow, estimate=estimate, window=window)
    return GatewayError(
        400,
        "invalid_request_error",
        f"prompt is too long: {prompt} tokens > {limit} maximum",
    )


def _figures(
    overflow: ContextOverflow, *, estimate: int, window: int | None
) -> tuple[int, int]:
    limit = overflow.limit or window
    prompt = overflow.prompt_tokens
    if prompt is None and overflow.excess and limit:
        prompt = limit + overflow.excess
    prompt = prompt or estimate
    if limit is None:
        limit = prompt - (overflow.excess or 1)
    limit = max(1, limit)
    return max(prompt, limit + 1), limit


def _from_match(match: re.Match[str]) -> ContextOverflow:
    groups = match.groupdict()
    return ContextOverflow(
        prompt_tokens=_int(groups.get("prompt")),
        limit=_int(groups.get("limit")),
        excess=_int(groups.get("excess")),
    )


def _from_fields(error: UpstreamError) -> ContextOverflow | None:
    prompt = _int(error.fields.get("n_prompt_tokens"))
    limit = _int(error.fields.get("n_ctx"))
    if prompt is None or limit is None:
        return None
    return ContextOverflow(prompt_tokens=prompt, limit=limit)


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value) or None
    return None
