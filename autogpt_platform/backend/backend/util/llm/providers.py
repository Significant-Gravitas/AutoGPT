"""Per-provider LLM call helper — the single seam every caller delegates to.

Owns the raw per-provider SDK call so the block-layer ``_llm_call``,
the dream-pass ``structured_completion``, and the copilot chat
dispatch all route through one implementation. Adding a new
execution mode (``batch``, ``flex``) or a new provider lands once
and every caller picks it up.

What this module IS responsible for:
  * Dispatching the call to the correct provider SDK based on
    ``provider`` argument
  * Normalizing the per-provider response shape into ``ProviderResponse``
  * Wrapping the call in a hard timeout
  * Sanitizing message content for UTF-8 transport
  * Switching on ``execution_mode`` (``sync`` today, ``batch`` +
    ``flex`` in later steps)

What this module is NOT responsible for (caller wraps):
  * ``LlmModel``-aware token-budget computation (caller passes
    pre-computed ``max_tokens``)
  * Prompt compression (``compress_context``) — block layer needs it,
    dream may not
  * Retry on validation failure (block layer's ``retry`` parameter)
  * Pydantic ``response_model`` validation (dream's
    ``structured_completion`` does this on top)
  * Cost-log writes / rate-limit charging — caller decides which
    billing path to use, ``ProviderResponse.cost_usd`` just surfaces
    the number
"""

from __future__ import annotations

import asyncio
import functools
import json as json_module
import logging
from datetime import datetime, timezone
from typing import Any, Literal, cast

import anthropic
import ollama
import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.shared_params import ResponseFormatJSONObject
from pydantic import Field
from pydantic.dataclasses import dataclass

from backend.util.clients import OPENROUTER_BASE_URL
from backend.util.llm.conversions import (
    ToolCall,
    ToolContentBlock,
    convert_openai_tool_fmt_to_anthropic,
    extract_openai_reasoning,
    extract_openai_tool_calls,
    extract_openrouter_cost,
    sanitize_messages_for_utf8,
)
from backend.util.openai_responses import (
    convert_tools_to_responses_format,
    extract_responses_content,
    extract_responses_reasoning,
    extract_responses_tool_calls,
    extract_responses_usage,
)
from backend.util.request import validate_url_host
from backend.util.settings import Settings

settings = Settings()
logger = logging.getLogger(__name__)


# Hard cap on a single provider HTTP request. Mirrors
# ``backend/blocks/llm.py::LLM_REQUEST_TIMEOUT_SECONDS``. Healthy
# non-streaming Messages / Responses calls finish in seconds; anything
# past 120s is almost certainly a stalled socket and retries-on-timeout
# would compound into multi-hour worst cases. Batch and flex paths
# override this where appropriate.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120

# Provider names accepted by ``call_provider``. Kept as a string Literal
# (not an enum) so the helper stays provider-name-agnostic — callers
# pass whatever string their model metadata reports.
ProviderLiteral = Literal[
    "openai",
    "anthropic",
    "groq",
    "ollama",
    "open_router",
    "llama_api",
    "aiml_api",
    "v0",
]

ExecutionMode = Literal["sync", "batch", "flex"]

# Providers that accept ``service_tier="flex"``. OpenAI exposes it
# natively on its Responses API; OpenRouter forwards the param through
# to OpenAI- and Google-backed models (the only flex-capable upstreams
# they support). Anthropic + Groq + Ollama + the open-weight gateways
# have no flex equivalent — callers asking for flex on them get a
# sync fallback with a log line.
_FLEX_SUPPORTED_PROVIDERS: set[str] = {"openai", "open_router"}

# Anthropic deprecated ``temperature`` on its newest model generation —
# the API rejects it outright with "`temperature` is deprecated for
# this model." (verified live: opus-4-7 and opus-4-8 reject;
# sonnet-4-6 accepts). Strip the param for known-rejecting families.
# The sync path additionally retries once without it on that exact
# error, so unknown future models self-heal; this list only matters
# for batch submissions, whose errors come back hours later in the
# result rows.
_ANTHROPIC_TEMPERATURE_DEPRECATED_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
)


def _anthropic_accepts_temperature(model: str) -> bool:
    return not model.startswith(_ANTHROPIC_TEMPERATURE_DEPRECATED_PREFIXES)


def _is_temperature_deprecation_error(exc: anthropic.BadRequestError) -> bool:
    error_text = str(exc).lower()
    return "temperature" in error_text and "deprecated" in error_text


# ---------------------------------------------------------------------------
# Result + submission carriers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProviderResponse:
    """Normalized response from any sync-mode provider call.

    Fields with default 0 / None / empty are absent on providers that
    don't surface them (e.g. Anthropic doesn't return USD cost,
    Ollama / Groq don't have cache tokens).
    """

    content: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    tool_calls: list[ToolContentBlock] | None = None
    reasoning: str | None = None
    cost_usd: float | None = None
    # Provider-native response object for callers that need raw access.
    # Kept out of repr + pydantic serialization (it's an opaque SDK object).
    raw_response: Any = Field(default=None, repr=False, exclude=True)


@dataclass(slots=True)
class BatchSubmissionRef:
    """Handle returned when ``execution_mode='batch'``.

    Apply step happens asynchronously when the batch poller (Step 4)
    delivers results keyed by ``custom_id``. Callers persist this
    handle and register a per-``custom_id`` callback with the
    ``BatchExecutor`` service.
    """

    provider: ProviderLiteral
    provider_batch_id: str
    custom_id: str
    submitted_at: datetime


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def call_provider(
    *,
    provider: ProviderLiteral,
    model: str,
    api_key: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None = None,
    execution_mode: ExecutionMode = "sync",
    tools: list[dict] | None = None,
    tool_choice: dict | None = None,
    force_json_output: bool = False,
    parallel_tool_calls: bool | openai.Omit = openai.omit,
    ollama_host: str = "localhost:11434",
    custom_id: str | None = None,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> ProviderResponse | BatchSubmissionRef:
    """Dispatch a single LLM call to the correct provider SDK.

    Sync mode (``execution_mode="sync"``) returns a ``ProviderResponse``
    with content + token counts + optional cost. Batch mode (Step 4)
    returns a ``BatchSubmissionRef`` — apply happens later via the
    batch poller. Flex mode (Step 8) returns a ``ProviderResponse``
    like sync but with the provider's discounted tier latency.

    Raises ``TimeoutError`` if the call exceeds ``timeout_seconds``
    (sync mode only; batch submissions are bounded by the provider's
    own SLA and the poller's own timeout policy).
    """
    if execution_mode == "batch":
        if not custom_id:
            raise ValueError(
                "execution_mode='batch' requires a non-empty custom_id "
                "for result routing once the batch completes."
            )
        sanitize_messages_for_utf8(messages)
        return await _submit_batch_one_request(
            provider=provider,
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            custom_id=custom_id,
        )
    # Flex tier is OpenAI's discounted-async-latency tier (~50% off, may
    # queue for up to 15 min). OpenRouter forwards ``service_tier=flex``
    # through to the upstream OpenAI/Google model. Other providers have
    # no flex equivalent — Anthropic's cost-saver is batch (Step 4), not
    # flex; Groq + the open-weight gateways just don't expose it. For
    # those we log and fall through to sync so dream/chat callers don't
    # have to know which providers support what.
    service_tier: str | None = None
    if execution_mode == "flex":
        if provider in _FLEX_SUPPORTED_PROVIDERS:
            service_tier = "flex"
        else:
            logger.warning(
                "execution_mode='flex' requested for provider=%s which has "
                "no flex tier; falling through to sync.",
                provider,
            )

    sanitize_messages_for_utf8(messages)

    try:
        return await asyncio.wait_for(
            _dispatch_sync(
                provider=provider,
                model=model,
                api_key=api_key,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                force_json_output=force_json_output,
                parallel_tool_calls=parallel_tool_calls,
                ollama_host=ollama_host,
                timeout_seconds=timeout_seconds,
                service_tier=service_tier,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"LLM request to {provider}/{model} exceeded "
            f"{timeout_seconds}s and was cancelled."
        ) from exc


# ---------------------------------------------------------------------------
# Sync dispatch
# ---------------------------------------------------------------------------


async def _dispatch_sync(
    *,
    provider: ProviderLiteral,
    model: str,
    api_key: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None,
    tools: list[dict] | None,
    tool_choice: dict | None,
    force_json_output: bool,
    parallel_tool_calls: bool | openai.Omit,
    ollama_host: str,
    timeout_seconds: float,
    service_tier: str | None = None,
) -> ProviderResponse:
    if provider == "openai":
        return await _call_openai_responses(
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            force_json_output=force_json_output,
            parallel_tool_calls=parallel_tool_calls,
            timeout_seconds=timeout_seconds,
            service_tier=service_tier,
        )
    if provider == "anthropic":
        return await _call_anthropic_messages(
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            timeout_seconds=timeout_seconds,
        )
    if provider == "groq":
        return await _call_groq(
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            force_json_output=force_json_output,
            timeout_seconds=timeout_seconds,
        )
    if provider == "ollama":
        return await _call_ollama(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            force_json_output=force_json_output,
            ollama_host=ollama_host,
            timeout_seconds=timeout_seconds,
        )
    if provider == "open_router":
        return await _call_openai_compat(
            base_url=OPENROUTER_BASE_URL,
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            force_json_output=force_json_output,
            parallel_tool_calls=parallel_tool_calls,
            timeout_seconds=timeout_seconds,
            include_openrouter_extras=True,
            service_tier=service_tier,
        )
    if provider == "llama_api":
        return await _call_openai_compat(
            base_url="https://api.llama.com/compat/v1/",
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            force_json_output=False,  # llama_api doesn't honor it
            parallel_tool_calls=parallel_tool_calls,
            timeout_seconds=timeout_seconds,
            include_openrouter_extras=False,
            extra_headers={"HTTP-Referer": "https://agpt.co", "X-Title": "AutoGPT"},
        )
    if provider == "aiml_api":
        return await _call_openai_compat(
            base_url="https://api.aimlapi.com/v2",
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=None,  # AI/ML API path historically passes no tools
            force_json_output=False,
            parallel_tool_calls=openai.omit,
            timeout_seconds=timeout_seconds,
            include_openrouter_extras=False,
            default_headers={
                "X-Project": "AutoGPT",
                "X-Title": "AutoGPT",
                "HTTP-Referer": "https://github.com/Significant-Gravitas/AutoGPT",
            },
        )
    if provider == "v0":
        return await _call_openai_compat(
            base_url="https://api.v0.dev/v1",
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            force_json_output=force_json_output,
            parallel_tool_calls=parallel_tool_calls,
            timeout_seconds=timeout_seconds,
            include_openrouter_extras=False,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


# ---------------------------------------------------------------------------
# OpenAI Responses API (native)
# ---------------------------------------------------------------------------


async def _call_openai_responses(
    *,
    model: str,
    api_key: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None,
    tools: list[dict] | None,
    force_json_output: bool,
    parallel_tool_calls: bool | openai.Omit,
    timeout_seconds: float,
    service_tier: str | None = None,
) -> ProviderResponse:
    client = openai.AsyncOpenAI(api_key=api_key)
    tools_param = convert_tools_to_responses_format(tools) if tools else openai.omit
    text_config: Any = openai.omit
    if force_json_output:
        text_config = {"format": {"type": "json_object"}}

    # ``service_tier`` lives in ``extra_body`` because OpenAI ships it on
    # the Responses payload but the openai-python SDK doesn't surface it
    # as a typed kwarg on every SDK release. ``extra_body`` is the
    # forward-compatible escape hatch.
    extra_body: dict[str, Any] = {}
    if service_tier:
        extra_body["service_tier"] = service_tier

    response = await client.responses.create(
        model=model,
        input=messages,  # type: ignore[arg-type]
        tools=tools_param,  # type: ignore[arg-type]
        max_output_tokens=max_tokens,
        # Reasoning models (o-series) reject ``temperature`` — same guard
        # as the Anthropic path: only send the field when the caller set
        # one, never a default.
        temperature=temperature if temperature is not None else openai.omit,
        parallel_tool_calls=parallel_tool_calls,
        text=text_config,  # type: ignore[arg-type]
        store=False,
        timeout=timeout_seconds,
        # ``omit`` is only valid for TYPED params — the SDK strips it
        # there. ``extra_body`` flows raw into ``options.extra_json``
        # (``make_request_options`` checks ``is not None``), and
        # ``_merge_mappings`` raises ``TypeError: 'Omit' object is not a
        # mapping`` on the sentinel. Absent extra_body must be ``None``.
        extra_body=extra_body or None,
    )

    raw_tool_calls = extract_responses_tool_calls(response)
    tool_calls = None
    if raw_tool_calls:
        tool_calls = [
            ToolContentBlock(
                id=tc["id"],
                type=tc["type"],
                function=ToolCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in raw_tool_calls
        ]
    prompt_tokens, completion_tokens = extract_responses_usage(response)

    return ProviderResponse(
        content=extract_responses_content(response),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tool_calls=tool_calls,
        reasoning=extract_responses_reasoning(response),
        raw_response=response,
    )


# ---------------------------------------------------------------------------
# Anthropic Messages API (native)
# ---------------------------------------------------------------------------


async def _call_anthropic_messages(
    *,
    model: str,
    api_key: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None,
    tools: list[dict] | None,
    tool_choice: dict | None,
    timeout_seconds: float,
) -> ProviderResponse:
    an_tools = convert_openai_tool_fmt_to_anthropic(tools)
    # Cache tool definitions alongside the system prompt — placing
    # cache_control on the last tool caches all tool schemas as a
    # single prefix; reads cost 10% of normal input tokens.
    if isinstance(an_tools, list) and an_tools:
        an_tools[-1] = {**an_tools[-1], "cache_control": {"type": "ephemeral"}}

    system_messages = [p["content"] for p in messages if p["role"] == "system"]
    sysprompt = " ".join(system_messages)

    anth_messages: list[dict] = []
    last_role: str | None = None
    for p in messages:
        if p["role"] in ["user", "assistant"]:
            if (
                p["role"] == last_role
                and anth_messages
                and isinstance(anth_messages[-1]["content"], str)
                and isinstance(p["content"], str)
            ):
                anth_messages[-1]["content"] += p["content"]
            else:
                anth_messages.append({"role": p["role"], "content": p["content"]})
                last_role = p["role"]

    client = anthropic.AsyncAnthropic(api_key=api_key)
    create_kwargs: dict[str, Any] = dict(
        model=model,
        messages=anth_messages,
        max_tokens=max_tokens,
        tools=an_tools,
        timeout=timeout_seconds,
    )
    if temperature is not None and _anthropic_accepts_temperature(model):
        create_kwargs["temperature"] = temperature
    if tool_choice is not None:
        create_kwargs["tool_choice"] = tool_choice
    if sysprompt.strip():
        # Anthropic rejects empty text blocks (HTTP 400) — only attach
        # the system field for non-whitespace prompts.
        create_kwargs["system"] = [
            {
                "type": "text",
                "text": sysprompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    try:
        resp = await client.messages.create(**create_kwargs)
    except anthropic.BadRequestError as exc:
        # Self-heal for models the deny-list doesn't know yet.
        if "temperature" not in create_kwargs or not (
            _is_temperature_deprecation_error(exc)
        ):
            raise
        logger.warning(
            "Anthropic model %s rejected temperature — retrying without it. "
            "Add the model to _ANTHROPIC_TEMPERATURE_DEPRECATED_PREFIXES.",
            model,
        )
        create_kwargs.pop("temperature")
        resp = await client.messages.create(**create_kwargs)
    if not resp.content:
        raise ValueError("No content returned from Anthropic.")

    tool_calls: list[ToolContentBlock] | None = None
    for content_block in resp.content:
        if content_block.type == "tool_use":
            if tool_calls is None:
                tool_calls = []
            tool_calls.append(
                ToolContentBlock(
                    id=content_block.id,
                    type=content_block.type,
                    function=ToolCall(
                        name=content_block.name,
                        arguments=json_module.dumps(content_block.input),
                    ),
                )
            )

    if not tool_calls and resp.stop_reason == "tool_use":
        logger.warning(
            "Anthropic returned stop_reason='tool_use' but no tool calls "
            "were extractable from content."
        )

    reasoning: str | None = None
    for content_block in resp.content:
        if hasattr(content_block, "type") and content_block.type == "thinking":
            reasoning = content_block.thinking
            break

    first_block = resp.content[0]
    content_text = (
        first_block.name
        if isinstance(first_block, anthropic.types.ToolUseBlock)
        else getattr(first_block, "text", "")
    )

    return ProviderResponse(
        content=content_text,
        prompt_tokens=resp.usage.input_tokens,
        completion_tokens=resp.usage.output_tokens,
        cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", None) or 0,
        cache_creation_tokens=getattr(resp.usage, "cache_creation_input_tokens", None)
        or 0,
        tool_calls=tool_calls,
        reasoning=reasoning,
        raw_response=resp,
    )


# ---------------------------------------------------------------------------
# Groq (OpenAI-shaped chat completions, no tools)
# ---------------------------------------------------------------------------


async def _call_groq(
    *,
    model: str,
    api_key: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None,
    tools: list[dict] | None,
    force_json_output: bool,
    timeout_seconds: float,
) -> ProviderResponse:
    if tools:
        raise ValueError("Groq does not support tools.")

    from groq import AsyncGroq  # local import — heavy SDK

    client = AsyncGroq(api_key=api_key)
    response_format = {"type": "json_object"} if force_json_output else None
    create_kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        response_format=response_format,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
    )
    # Same guard as the Anthropic path — only send ``temperature`` when
    # the caller set one, so models that reject it never see the field.
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    response = await client.chat.completions.create(**create_kwargs)
    if not response.choices:
        raise ValueError("Groq returned empty choices in response")
    return ProviderResponse(
        content=response.choices[0].message.content or "",
        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
        completion_tokens=(response.usage.completion_tokens if response.usage else 0),
        raw_response=response.choices[0].message,
    )


# ---------------------------------------------------------------------------
# Ollama (local-runner chat API)
# ---------------------------------------------------------------------------


async def _call_ollama(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None,
    tools: list[dict] | None,
    force_json_output: bool,
    ollama_host: str,
    timeout_seconds: float,
) -> ProviderResponse:
    if tools:
        raise ValueError("Ollama does not support tools.")

    # SSRF guard — the host can be user-supplied (block-layer input
    # field), so it must match an operator-configured trust list.
    await validate_url_host(ollama_host, trusted_hostnames=_trusted_ollama_hostnames())

    # ``num_predict`` is Ollama's OUTPUT token cap. ``num_ctx`` (the
    # input context window) is deliberately left at the model default —
    # setting it from ``max_tokens`` silently truncated long prompts to
    # the output budget.
    options: dict[str, Any] = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature

    client = ollama.AsyncClient(host=ollama_host, timeout=timeout_seconds)
    response = await client.chat(
        model=model,
        messages=messages,
        stream=False,
        format="json" if force_json_output else None,
        options=options,
    )
    return ProviderResponse(
        content=response.message.content or "",
        prompt_tokens=response.prompt_eval_count or 0,
        completion_tokens=response.eval_count or 0,
        raw_response=response,
    )


def _trusted_ollama_hostnames() -> list[str]:
    """Hosts an Ollama call may target without IP-resolution checks.

    Two operator-set config values can name the local LLM endpoint, and
    both are equally trusted:

      * ``Config.ollama_host`` — the block layer's Ollama setting.
      * ``ChatConfig.base_url`` (``CHAT_BASE_URL``) — the copilot chat
        endpoint, which the dream pass normalizes into ``ollama_host``
        on the local transport.

    Deriving the allowlist from only the former made every local dream
    phase fail the SSRF check whenever the two were configured
    differently (e.g. backend in Docker with
    ``CHAT_BASE_URL=http://host.docker.internal:11434/v1`` while
    ``Config.ollama_host`` sat at its ``localhost:11434`` default).
    """
    trusted = [settings.config.ollama_host]
    chat_base_url = _chat_config_base_url()
    if chat_base_url:
        trusted.append(chat_base_url)
    return trusted


def _chat_config_base_url() -> str | None:
    """Read the copilot ``ChatConfig.base_url`` for the Ollama trust list.

    A broken chat config must not take down block-layer Ollama calls —
    on any failure fall back to the block-layer trust list only. The
    failure is deliberately NOT memoized (``lru_cache`` doesn't cache
    exceptions), so a fixed config is picked up on a later dispatch.
    """
    try:
        return _read_chat_config_base_url()
    except Exception:
        logger.warning(
            "Could not read ChatConfig.base_url for the Ollama trusted-host "
            "list; falling back to Config.ollama_host only.",
            exc_info=True,
        )
        return None


@functools.lru_cache(maxsize=1)
def _read_chat_config_base_url() -> str | None:
    """Construct ``ChatConfig`` once and memoize its ``base_url``.

    ``ChatConfig`` is a pydantic-settings class with ``env_file=".env"``,
    so each construction does synchronous dotenv disk I/O plus the full
    validator chain (including repeated validator warnings) — per-call
    construction meant that ran on every Ollama dispatch for a value
    that only changes with the environment.

    Lazy import: ``backend.copilot.config`` imports this module (for
    ``ProviderLiteral``), so a top-level import here would be circular.
    """
    from backend.copilot.config import ChatConfig

    return ChatConfig().base_url


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completions (OpenRouter, Llama API, AI/ML, v0)
# ---------------------------------------------------------------------------


async def _call_openai_compat(
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None,
    tools: list[dict] | None,
    force_json_output: bool,
    parallel_tool_calls: bool | openai.Omit,
    timeout_seconds: float,
    include_openrouter_extras: bool,
    extra_headers: dict[str, str] | None = None,
    default_headers: dict[str, str] | None = None,
    service_tier: str | None = None,
) -> ProviderResponse:
    client_kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key}
    if default_headers:
        client_kwargs["default_headers"] = default_headers
    client = openai.AsyncOpenAI(**client_kwargs)

    call_kwargs: dict[str, Any] = dict(
        model=model,
        messages=cast(list[ChatCompletionMessageParam], messages),
        max_tokens=max_tokens,
        tools=(cast(list[ChatCompletionToolParam], tools) if tools else openai.omit),
        parallel_tool_calls=parallel_tool_calls,
        response_format=(
            ResponseFormatJSONObject(type="json_object")
            if force_json_output
            else openai.omit
        ),
        timeout=timeout_seconds,
    )
    # Same guard as the Anthropic path — only send ``temperature`` when
    # the caller set one, so upstreams that reject it (some reasoning
    # models behind OpenRouter) never see the field.
    if temperature is not None:
        call_kwargs["temperature"] = temperature
    extra_body: dict[str, Any] = {}
    if include_openrouter_extras:
        # Ask OpenRouter to surface per-request USD cost on `usage.cost`.
        # Same shape used by backend/executor/simulator.py — keep aligned.
        extra_body["usage"] = {"include": True}
        call_kwargs["extra_headers"] = {
            "HTTP-Referer": "https://agpt.co",
            "X-Title": "AutoGPT",
        }
    elif extra_headers:
        call_kwargs["extra_headers"] = extra_headers
    if service_tier:
        # OpenRouter forwards ``service_tier`` to OpenAI- and Google-backed
        # models. Setting it for other upstreams is harmless — OpenRouter
        # drops unknown params silently rather than failing the request.
        extra_body["service_tier"] = service_tier
    if extra_body:
        call_kwargs["extra_body"] = extra_body

    response = await client.chat.completions.create(**call_kwargs)
    if not response.choices:
        raise ValueError(f"{base_url} returned empty choices in response")

    cache_read, cache_creation = _extract_openai_compat_cache_tokens(response)

    return ProviderResponse(
        content=response.choices[0].message.content or "",
        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
        completion_tokens=(response.usage.completion_tokens if response.usage else 0),
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        tool_calls=extract_openai_tool_calls(response),
        reasoning=extract_openai_reasoning(response),
        cost_usd=(
            extract_openrouter_cost(response) if include_openrouter_extras else None
        ),
        raw_response=response.choices[0].message,
    )


def _extract_openai_compat_cache_tokens(response: Any) -> tuple[int, int]:
    """Extract (cache_read, cache_creation) from an OpenAI-compat usage block.

    Provider quirks this navigates:
      * Standard OpenAI: ``usage.prompt_tokens_details.cached_tokens``
      * OpenRouter routing to Anthropic: cache writes surface as
        ``prompt_tokens_details.model_extra["cache_write_tokens"]``
      * Anthropic-via-OpenAI-compat (rare): writes under
        ``cache_creation_input_tokens`` on the same ``model_extra`` blob

    Returns ``(0, 0)`` when no usage object is present so cost
    computation degrades gracefully on providers that don't return it.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is None:
        return 0, 0
    cache_read = int(getattr(ptd, "cached_tokens", 0) or 0)
    ptd_extras = getattr(ptd, "model_extra", None) or {}
    cache_creation = int(
        ptd_extras.get("cache_write_tokens")
        or ptd_extras.get("cache_creation_input_tokens")
        or 0
    )
    return cache_read, cache_creation


# ---------------------------------------------------------------------------
# Streaming dispatch (chat baseline)
# ---------------------------------------------------------------------------
#
# The chat baseline path needs the raw provider stream so it can iterate
# chunks for delta-emitted SSE events, accumulate per-chunk token /
# cost / cache deltas into its ``_BaselineStreamState``, and react to
# tool-call partials as they arrive. Returning a normalized
# ``ProviderResponse`` would collapse those incremental signals — so
# the streaming entry-point returns the SDK's ``AsyncStream`` object
# directly. The caller still owns chunk parsing; the helper centralizes
# client instantiation + the OpenAI-compat ``create_kwargs`` assembly
# (``extra_body``, ``extra_headers``, ``stream_options``, tools).
#
# Client factory injection:
#   The chat layer instantiates ``openai.AsyncOpenAI`` via a
#   Langfuse-wrapped subclass (``langfuse.openai.AsyncOpenAI``) to
#   thread OTel-style spans through every chat turn. To preserve that
#   tracing, callers pass ``client_factory=LangfuseAsyncOpenAI`` (or
#   any subclass with the same constructor) and the helper builds the
#   client from it. Defaults to ``openai.AsyncOpenAI`` so non-traced
#   callers (eval scripts, integration tests) need no extra wiring.


# Wider than ``ProviderResponse`` because the caller picks up chunks
# directly. Typed as ``Any`` to avoid importing the SDK's private
# stream typevars across the helper boundary — the openai SDK is the
# source of truth for chunk shape and callers are already openai-shape
# aware (they came here from a direct SDK call).
StreamResponse = Any


async def call_provider_stream(
    *,
    model: str,
    messages: list[dict],
    client: openai.AsyncOpenAI | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    extra_body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    stream_options: dict[str, Any] | None = None,
    tools: list[dict] | None = None,
    max_tokens: int | openai.Omit = openai.omit,
    timeout: float | openai.Omit = openai.omit,
    client_factory: type[openai.AsyncOpenAI] = openai.AsyncOpenAI,
) -> StreamResponse:
    """Open an OpenAI-compatible streaming chat completion and return
    the raw ``AsyncStream`` for the caller to iterate.

    Used by the chat baseline path. The non-streaming sync helper
    (``call_provider``) is the right choice for everything else — only
    consumers that need chunk-level deltas (tool partials, incremental
    SSE tokens) belong here.

    Either pass a pre-built ``client`` (the chat layer caches its
    Langfuse-wrapped client at module level) OR pass
    ``base_url`` + ``api_key`` and the helper builds one via
    ``client_factory``. The chat layer prefers the pre-built path so
    TCP connections stay pooled across turns; tests pass a mocked
    client. Default factory is ``openai.AsyncOpenAI`` — callers
    needing Langfuse spans pass the wrapped subclass.

    Does NOT wrap the call in ``asyncio.wait_for`` and defaults
    ``timeout`` to ``openai.omit`` so the SDK / client's own per-request
    default applies. A streaming chat request can legitimately run
    minutes (long reasoning, large tool-call payloads) and the
    streaming protocol already exposes keepalive failures via chunk
    timeouts; the caller's outer cancel scope is the right place to
    bound total time. Pass an explicit ``timeout`` only when you
    really mean per-request inter-chunk read timeout.
    """
    sanitize_messages_for_utf8(messages)
    if client is None:
        if base_url is None or api_key is None:
            raise ValueError(
                "call_provider_stream: pass either `client` or both "
                "`base_url` and `api_key`."
            )
        client = client_factory(base_url=base_url, api_key=api_key)
    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": cast(list[ChatCompletionMessageParam], messages),
        "stream": True,
    }
    if not isinstance(timeout, openai.Omit):
        create_kwargs["timeout"] = timeout
    # Only set ``max_tokens`` when the caller passed a real value. Some
    # transports (OpenRouter on thinking routes) inject their own
    # default and 400 on a redundant client-side limit, so the chat
    # baseline path passes ``openai.omit`` to skip the field
    # entirely.
    if not isinstance(max_tokens, openai.Omit):
        create_kwargs["max_tokens"] = max_tokens
    if extra_body:
        create_kwargs["extra_body"] = extra_body
    if extra_headers:
        create_kwargs["extra_headers"] = extra_headers
    if stream_options:
        create_kwargs["stream_options"] = stream_options
    if tools:
        create_kwargs["tools"] = cast(list[ChatCompletionToolParam], list(tools))
    return await client.chat.completions.create(**create_kwargs)


async def call_provider_openai_compat_sync(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    client: openai.AsyncOpenAI | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    extra_body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    client_factory: type[openai.AsyncOpenAI] = openai.AsyncOpenAI,
) -> Any:
    """Non-streaming OpenAI-compat chat call, returns the raw SDK
    ``ChatCompletion`` so callers that already navigate
    ``response.usage`` / ``response.choices`` directly don't need to
    rewrite their extraction logic.

    Used by the chat title-generation path. New callers without
    SDK-specific extraction logic should prefer ``call_provider``
    which normalizes into ``ProviderResponse``.

    Either pass a pre-built ``client`` OR ``base_url`` + ``api_key``
    + (optionally) ``client_factory``. See ``call_provider_stream``
    for rationale.
    """
    sanitize_messages_for_utf8(messages)
    if client is None:
        if base_url is None or api_key is None:
            raise ValueError(
                "call_provider_openai_compat_sync: pass either `client` "
                "or both `base_url` and `api_key`."
            )
        client = client_factory(base_url=base_url, api_key=api_key)
    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": cast(list[ChatCompletionMessageParam], messages),
        "max_tokens": max_tokens,
        "timeout": timeout_seconds,
    }
    if extra_body:
        create_kwargs["extra_body"] = extra_body
    if extra_headers:
        create_kwargs["extra_headers"] = extra_headers
    return await client.chat.completions.create(**create_kwargs)


# ---------------------------------------------------------------------------
# Batch submission (Anthropic Messages Batches API)
# ---------------------------------------------------------------------------


async def _submit_batch_one_request(
    *,
    provider: ProviderLiteral,
    model: str,
    api_key: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None,
    tools: list[dict] | None,
    tool_choice: dict | None,
    custom_id: str,
) -> BatchSubmissionRef:
    """Submit a single-request batch and return the provider's handle.

    The shared helper takes one ``call_provider(execution_mode="batch")``
    invocation = one batch submission with exactly one request. Callers
    that want to group several requests into one batch (e.g. dream
    submitting 3 phases together for a single ``custom_id`` namespace)
    can layer their own grouping on top — for the in-process orchestrator
    today, one request per submission is the right granularity because
    each phase has its own ``custom_id`` and ``apply`` handler anyway.

    Result apply is async — the BatchExecutor service polls Anthropic
    every 30s → 5min backoff, downloads results when ``processing_status
    == 'ended'``, and dispatches to a caller-registered callback keyed
    on ``custom_id``.
    """
    if provider == "anthropic":
        return await _submit_anthropic_batch(
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            custom_id=custom_id,
        )
    if provider == "openai":
        raise NotImplementedError(
            "OpenAI batch submission is a follow-up to Step 4 "
            "(Anthropic-first; OpenAI batch lands when we add OpenAI "
            "models to the dream pass)."
        )
    raise NotImplementedError(
        f"execution_mode='batch' is only supported for provider='anthropic' "
        f"today; got provider={provider!r}. Callers should route to sync_baseline."
    )


async def _submit_anthropic_batch(
    *,
    model: str,
    api_key: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float | None,
    tools: list[dict] | None,
    tool_choice: dict | None,
    custom_id: str,
) -> BatchSubmissionRef:
    """Call ``client.messages.batches.create`` for a single-request batch.

    Anthropic accepts ``requests=[{custom_id, params}]`` where ``params``
    is a ``MessageCreateParams`` dict. We reuse the same message reshape
    + system-prompt wrapping logic the sync Anthropic path uses, so a
    batch request and its sync equivalent produce semantically identical
    LLM behaviour.
    """
    an_tools = convert_openai_tool_fmt_to_anthropic(tools)
    if isinstance(an_tools, list) and an_tools:
        an_tools[-1] = {**an_tools[-1], "cache_control": {"type": "ephemeral"}}

    system_messages = [p["content"] for p in messages if p["role"] == "system"]
    sysprompt = " ".join(system_messages)

    anth_messages: list[dict] = []
    last_role: str | None = None
    for p in messages:
        if p["role"] in ("user", "assistant"):
            if (
                p["role"] == last_role
                and anth_messages
                and isinstance(anth_messages[-1]["content"], str)
                and isinstance(p["content"], str)
            ):
                anth_messages[-1]["content"] += p["content"]
            else:
                anth_messages.append({"role": p["role"], "content": p["content"]})
                last_role = p["role"]

    params: dict[str, Any] = {
        "model": model,
        "messages": anth_messages,
        "max_tokens": max_tokens,
    }
    # Batch errors surface hours later in the result rows — never submit
    # temperature to models that reject it (no cheap retry exists here).
    if temperature is not None and _anthropic_accepts_temperature(model):
        params["temperature"] = temperature
    # Only attach tools / tool_choice when the caller actually passed
    # them — Anthropic rejects empty arrays with HTTP 400.
    if isinstance(an_tools, list) and an_tools:
        params["tools"] = an_tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
    if sysprompt.strip():
        params["system"] = [
            {
                "type": "text",
                "text": sysprompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    client = anthropic.AsyncAnthropic(api_key=api_key)
    # Anthropic's typed Request union is too strict for our dynamic
    # params shape (we conditionally include ``system`` / ``tools`` /
    # ``tool_choice``). The runtime contract accepts any dict matching
    # ``MessageCreateParamsNonStreaming`` — cast to ``Any`` to bypass
    # pyright's nominal check.
    batch_requests: Any = [{"custom_id": custom_id, "params": params}]
    batch = await client.messages.batches.create(requests=batch_requests)

    return BatchSubmissionRef(
        provider="anthropic",
        provider_batch_id=batch.id,
        custom_id=custom_id,
        submitted_at=datetime.now(timezone.utc),
    )


async def cancel_batch(
    *, provider: ProviderLiteral, provider_batch_id: str, api_key: str
) -> bool:
    """Best-effort cancel of an in-flight provider batch.

    Used when local bookkeeping fails *after* the provider already
    accepted a batch submission (e.g. the BatchExecutor pending-queue
    write fails): without a cancel the provider would run the paid batch
    to completion with no callback to consume it, orphaning the spend.
    Never raises — returns ``True`` only when the provider acknowledged
    the cancel.
    """
    try:
        if provider == "anthropic":
            client = anthropic.AsyncAnthropic(api_key=api_key)
            await client.messages.batches.cancel(provider_batch_id)
            return True
        logger.warning(
            "cancel_batch: no cancel path for provider=%s (batch=%s)",
            provider,
            provider_batch_id,
        )
        return False
    except Exception:
        logger.exception(
            "cancel_batch: failed to cancel provider=%s batch=%s",
            provider,
            provider_batch_id,
        )
        return False


# ---------------------------------------------------------------------------
# Batch poll + download (used by the BatchExecutor service)
# ---------------------------------------------------------------------------


BatchStatusLiteral = Literal["pending", "processing", "ended", "failed"]


@dataclass(slots=True)
class BatchResultRow:
    """One row from a downloaded batch result.

    Mirrors what ``dream/batch/models.BatchResult`` carries — kept as a
    plain dataclass at this layer so callers don't have to pull the
    dream-pass module into shared infrastructure.
    """

    custom_id: str
    content: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    error: str | None = None
    # Provider-native result object for callers that need raw access.
    # Kept out of repr + pydantic serialization (it's an opaque SDK object).
    raw_result: Any = Field(default=None, repr=False, exclude=True)


async def poll_batch(
    *, provider: ProviderLiteral, provider_batch_id: str, api_key: str
) -> BatchStatusLiteral:
    """Return the current normalized status of a submitted batch.

    Maps Anthropic's ``processing_status`` (``in_progress`` /
    ``canceling`` / ``ended``) onto a small four-state enum the
    BatchExecutor switches on. ``canceling`` collapses to
    ``processing`` because callers can't act on it differently — we
    let Anthropic finish the cancellation and the next poll catches
    ``ended`` / ``failed``.
    """
    if provider != "anthropic":
        raise NotImplementedError(
            f"poll_batch only supports provider='anthropic' today; "
            f"got {provider!r}."
        )
    client = anthropic.AsyncAnthropic(api_key=api_key)
    batch = await client.messages.batches.retrieve(provider_batch_id)
    status = getattr(batch, "processing_status", None)
    if status == "ended":
        return "ended"
    if status == "in_progress":
        return "processing"
    if status == "canceling":
        return "processing"
    # Unknown / future states → log + report pending so the caller polls again.
    logger.warning(
        "Unknown Anthropic batch processing_status=%r — reporting pending.",
        status,
    )
    return "pending"


async def download_batch_results(
    *, provider: ProviderLiteral, provider_batch_id: str, api_key: str
) -> list[BatchResultRow]:
    """Download all per-request rows for a completed batch.

    Anthropic streams results as JSONL (one JSON object per line). For
    each line we extract:

      * ``custom_id`` — caller routing key
      * Either ``message.content`` (succeeded) → flatten to a JSON
        string for the dream parser, OR an error string
      * Per-row usage (input/output/cache_read/cache_creation)

    The downloaded rows are returned in submission order so callers
    can correlate by index when they don't care about ``custom_id``.
    """
    if provider != "anthropic":
        raise NotImplementedError(
            f"download_batch_results only supports provider='anthropic' "
            f"today; got {provider!r}."
        )
    client = anthropic.AsyncAnthropic(api_key=api_key)
    rows: list[BatchResultRow] = []
    async for entry in await client.messages.batches.results(provider_batch_id):
        custom_id = getattr(entry, "custom_id", "") or ""
        result = getattr(entry, "result", None)
        result_type = getattr(result, "type", None) if result is not None else None
        if result_type == "succeeded":
            message = getattr(result, "message", None)
            content_text = _anthropic_content_to_text(message)
            usage = getattr(message, "usage", None) if message is not None else None
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            rows.append(
                BatchResultRow(
                    custom_id=custom_id,
                    content=content_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                    raw_result=entry,
                )
            )
        else:
            # ``errored`` / ``canceled`` / ``expired`` — surface the
            # error string but keep token counts at 0 (the request
            # didn't run, so we don't owe Anthropic anything for it).
            error_obj = getattr(result, "error", None) if result is not None else None
            error_str = (
                str(error_obj) if error_obj is not None else (result_type or "errored")
            )
            rows.append(
                BatchResultRow(
                    custom_id=custom_id,
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    error=error_str,
                    raw_result=entry,
                )
            )
    return rows


def _anthropic_content_to_text(message: Any) -> str:
    """Flatten an Anthropic ``message.content`` blob into a string.

    Three shapes the BatchExecutor's downstream parser handles:
      * Single ``text`` block → return its ``text``.
      * Single ``tool_use`` block → return ``json.dumps(input)``. This
        is the "structured output" path (forced ``tool_choice`` makes
        Claude emit exactly one tool_use block; the dream pass parses
        the JSON straight into a Pydantic model).
      * Multi-block / unknown → join all ``text`` blocks with newlines.

    Returns ``""`` when the message is missing or empty.
    """
    if message is None:
        return ""
    content = getattr(message, "content", None) or []
    if not content:
        return ""
    first = content[0]
    first_type = getattr(first, "type", None)
    if first_type == "tool_use" and len(content) == 1:
        return json_module.dumps(getattr(first, "input", {}) or {})
    if first_type == "text" and len(content) == 1:
        return getattr(first, "text", "") or ""
    # Multi-block fallback — concatenate text blocks.
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "\n".join(parts)
