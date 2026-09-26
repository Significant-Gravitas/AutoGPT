"""Structured output from one background LLM call, made on its context's route.

``structured_complete`` makes one sync ``call_provider`` call with the
platform's key for the route's provider and parses the answer into a Pydantic
model. It asks for JSON mode on every provider but the native Anthropic API,
which ignores it and gets one tool built from the model instead
(``dream/structured_output.py``): forced where the model accepts that, left
to the model (``auto``) where it doesn't. The parsing, and its recovery from
fences and prose around the JSON, lives in ``dream/llm.py``.

The dream's Anthropic batch path does not come through here: it submits via
``call_provider(execution_mode="batch")`` from ``dream/batch_submit.py``.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar
from urllib.parse import urlparse, urlunparse

import anthropic
from pydantic import BaseModel

from backend.copilot.dream.llm import parse_structured_output
from backend.copilot.dream.structured_output import (
    StructuredRequest,
    structured_payload,
    structured_request,
)
from backend.copilot.transport_routing import ProviderRoutingKwargs
from backend.util.llm.providers import (
    BatchSubmissionRef,
    ProviderResponse,
    call_provider,
    is_forced_tool_choice_rejection,
)

from .context import CostSource, InferenceContext, InferenceError, InferenceUsage
from .routing import platform_credentials

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_Send = Callable[[StructuredRequest], Awaitable[ProviderResponse | BatchSubmissionRef]]


class StructuredCompletion(BaseModel, Generic[T]):
    """What ``structured_complete`` returns: the parsed value and its usage."""

    value: T
    usage: InferenceUsage


async def structured_complete(
    ctx: InferenceContext,
    messages: list[dict[str, str]],
    response_model: type[T],
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
) -> StructuredCompletion[T]:
    """One call on ``ctx.route``, parsed into *response_model*.

    ``ctx.job.timeout_seconds`` is the call's wall-clock budget; ``None``
    leaves ``call_provider``'s shared default.

    Raises ``InferenceError`` when the route cannot be called (no key for it,
    a model its provider cannot take), the call fails, or the answer is
    empty, unparseable or off the schema. Only the last three carry usage:
    the provider billed those tokens.
    """
    if ctx.route.engine != "provider_sync":
        raise InferenceError(
            f"structured_complete makes sync calls; the route's engine is "
            f"{ctx.route.engine!r}"
        )
    platform = platform_credentials(ctx.route)
    try:
        request = structured_request(
            ctx.route.provider, ctx.route.model, response_model
        )
    except ValueError as exc:
        raise InferenceError(str(exc)) from exc
    response = await _call_provider_sync(
        ctx,
        platform,
        request,
        messages=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    usage = _usage(response, request.model, ctx)
    value = parse_structured_output(structured_payload(response), response_model, usage)
    return StructuredCompletion(value=value, usage=usage)


async def _call_provider_sync(
    ctx: InferenceContext,
    platform: ProviderRoutingKwargs,
    request: StructuredRequest,
    *,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
) -> ProviderResponse:
    """One sync ``call_provider`` round trip (two, when the model turns the
    forced output tool down); any failure is an ``InferenceError`` with no
    usage, since no response came back to bill."""

    async def send(attempt: StructuredRequest) -> ProviderResponse | BatchSubmissionRef:
        return await call_provider(
            provider=ctx.route.provider,
            model=attempt.model,
            api_key=platform.api_key,
            messages=attempt.prompt(messages),
            max_tokens=max_output_tokens,
            temperature=temperature,
            force_json_output=attempt.force_json_output,
            tools=attempt.tools,
            tool_choice=attempt.tool_choice,
            timeout_seconds=ctx.job.timeout_seconds,
            # ``call_provider`` only reads ``ollama_host`` for
            # ``provider="ollama"``, so passing it on cloud routes is harmless.
            ollama_host=_normalize_ollama_host(platform.base_url),
        )

    try:
        response = await _with_forced_tool_fallback(send, request)
    except Exception as exc:
        raise InferenceError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

    if not isinstance(response, ProviderResponse):
        # ``call_provider`` returns a ``BatchSubmissionRef`` only for
        # ``execution_mode="batch"``, which this sync path never asks for.
        raise InferenceError(
            "structured_complete expected a sync ProviderResponse but got a "
            f"{type(response).__name__} — execution_mode must stay 'sync' here."
        )
    return response


async def _with_forced_tool_fallback(
    send: _Send, request: StructuredRequest
) -> ProviderResponse | BatchSubmissionRef:
    """*request*, and once more with the output tool left to the model
    (``auto``) and the prompt asking for the call when the model turns the
    forced tool down (Anthropic's 400 ``BadRequestError`` naming
    ``tool_choice``, and only that): the self-heal for a model missing from
    the provider's forced-tool list. A second failure is final."""
    try:
        return await send(request)
    except anthropic.BadRequestError as exc:
        if not (request.forces_output_tool and is_forced_tool_choice_rejection(exc)):
            raise
        logger.warning(
            "Model %s rejected the forced output tool (%s); retrying once "
            "with tool_choice=auto. Add it to "
            "_ANTHROPIC_FORCED_TOOL_CHOICE_UNSUPPORTED in "
            "backend/util/llm/providers.py.",
            request.model,
            exc,
        )
    return await send(request.with_output_tool_optional())


def _usage(
    response: ProviderResponse, model: str, ctx: InferenceContext
) -> InferenceUsage:
    """The call's usage, named for the model that was called (the spelling
    the price card resolves), with the provider's cost when it reported one."""
    cost_source: CostSource = "provider" if response.cost_usd is not None else "none"
    return InferenceUsage(
        model=model,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_creation_tokens=response.cache_creation_tokens,
        cost_usd=response.cost_usd,
        cost_source=cost_source,
        payer=ctx.route.payer,
    )


def _normalize_ollama_host(base_url: str | None) -> str:
    """Turn ``CHAT_BASE_URL`` into the host string ollama.AsyncClient wants.

    ``CHAT_BASE_URL`` for local installs points at the OpenAI-compat surface
    (``http://localhost:11434/v1``); Ollama's native client takes the raw host
    and would POST to ``/v1/api/generate`` and 404 with the path left on. So
    path, query and fragment go, leaving ``http://localhost:11434``. With no
    ``base_url``, the platform default (``localhost:11434``).
    """
    if not base_url:
        return "localhost:11434"
    parsed = urlparse(base_url)
    # ``urlparse("localhost:11434")`` reports ``scheme="localhost"``: with no
    # ``//`` after it the colon reads as a scheme separator. Only http(s) are
    # real schemes here; anything else was a bare ``host:port``.
    if parsed.scheme not in ("http", "https"):
        return base_url
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
