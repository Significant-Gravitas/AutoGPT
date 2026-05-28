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
import json as json_module
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

import anthropic
import ollama
import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.shared_params import ResponseFormatJSONObject

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
    # Not part of equality / serialization.
    raw_response: Any = field(default=None, repr=False, compare=False)


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
        raise NotImplementedError(
            "execution_mode='batch' lands in Step 4 of the rollout "
            "(see plans/idempotent-launching-moth.md component E). "
            "Today every caller falls back to sync."
        )
    if execution_mode == "flex":
        raise NotImplementedError(
            "execution_mode='flex' lands in Step 8 of the rollout. "
            "Today every caller falls back to sync."
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
) -> ProviderResponse:
    if provider == "openai":
        return await _call_openai_responses(
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            force_json_output=force_json_output,
            parallel_tool_calls=parallel_tool_calls,
            timeout_seconds=timeout_seconds,
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
            tools=tools,
            force_json_output=force_json_output,
            timeout_seconds=timeout_seconds,
        )
    if provider == "ollama":
        return await _call_ollama(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
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
            tools=tools,
            force_json_output=force_json_output,
            parallel_tool_calls=parallel_tool_calls,
            timeout_seconds=timeout_seconds,
            include_openrouter_extras=True,
        )
    if provider == "llama_api":
        return await _call_openai_compat(
            base_url="https://api.llama.com/compat/v1/",
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
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
    tools: list[dict] | None,
    force_json_output: bool,
    parallel_tool_calls: bool | openai.Omit,
    timeout_seconds: float,
) -> ProviderResponse:
    client = openai.AsyncOpenAI(api_key=api_key)
    tools_param = convert_tools_to_responses_format(tools) if tools else openai.omit
    text_config: Any = openai.omit
    if force_json_output:
        text_config = {"format": {"type": "json_object"}}

    response = await client.responses.create(
        model=model,
        input=messages,  # type: ignore[arg-type]
        tools=tools_param,  # type: ignore[arg-type]
        max_output_tokens=max_tokens,
        parallel_tool_calls=parallel_tool_calls,
        text=text_config,  # type: ignore[arg-type]
        store=False,
        timeout=timeout_seconds,
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
    if temperature is not None:
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
    tools: list[dict] | None,
    force_json_output: bool,
    timeout_seconds: float,
) -> ProviderResponse:
    if tools:
        raise ValueError("Groq does not support tools.")

    from groq import AsyncGroq  # local import — heavy SDK

    client = AsyncGroq(api_key=api_key)
    response_format = {"type": "json_object"} if force_json_output else None
    response = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        response_format=response_format,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        timeout=timeout_seconds,
    )
    if not response.choices:
        raise ValueError("Groq returned empty choices in response")
    return ProviderResponse(
        content=response.choices[0].message.content or "",
        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
        completion_tokens=(response.usage.completion_tokens if response.usage else 0),
        raw_response=response.choices[0].message,
    )


# ---------------------------------------------------------------------------
# Ollama (local-runner generate API)
# ---------------------------------------------------------------------------


async def _call_ollama(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    tools: list[dict] | None,
    ollama_host: str,
    timeout_seconds: float,
) -> ProviderResponse:
    if tools:
        raise ValueError("Ollama does not support tools.")

    # SSRF guard — user-supplied host must match the configured trust list.
    await validate_url_host(
        ollama_host, trusted_hostnames=[settings.config.ollama_host]
    )

    client = ollama.AsyncClient(host=ollama_host, timeout=timeout_seconds)
    sys_messages = [p["content"] for p in messages if p["role"] == "system"]
    usr_messages = [p["content"] for p in messages if p["role"] != "system"]
    response = await client.generate(
        model=model,
        prompt=f"{sys_messages}\n\n{usr_messages}",
        stream=False,
        options={"num_ctx": max_tokens},
    )
    return ProviderResponse(
        content=response.get("response") or "",
        prompt_tokens=response.get("prompt_eval_count") or 0,
        completion_tokens=response.get("eval_count") or 0,
        raw_response=response.get("response") or "",
    )


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
    tools: list[dict] | None,
    force_json_output: bool,
    parallel_tool_calls: bool | openai.Omit,
    timeout_seconds: float,
    include_openrouter_extras: bool,
    extra_headers: dict[str, str] | None = None,
    default_headers: dict[str, str] | None = None,
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
    if include_openrouter_extras:
        # Ask OpenRouter to surface per-request USD cost on `usage.cost`.
        # Same shape used by backend/executor/simulator.py — keep aligned.
        call_kwargs["extra_body"] = {"usage": {"include": True}}
        call_kwargs["extra_headers"] = {
            "HTTP-Referer": "https://agpt.co",
            "X-Title": "AutoGPT",
        }
    elif extra_headers:
        call_kwargs["extra_headers"] = extra_headers

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
