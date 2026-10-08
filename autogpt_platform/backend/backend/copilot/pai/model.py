"""The Pydantic AI model for a pai turn, resolved the way the baseline does.

The route comes from ``model_router.resolve_model_route`` (through the
baseline's own tier resolver) and is normalised for the transport the
baseline client dials (``ChatConfig.baseline_provider``):

* ``openrouter`` -> :class:`CopilotOpenRouterModel` over the baseline's
  Langfuse-wrapped OpenAI client, with OpenRouter's ``usage.include`` cost and
  the ``reasoning`` param for thinking routes, plus the baseline's Anthropic
  ``cache_control`` breakpoints (static instructions, tool list, skills block);
* ``anthropic`` -> ``AnthropicModel`` straight to api.anthropic.com, with the
  native ``thinking`` budget and cached instructions and tool definitions;
* ``local`` -> a plain OpenAI-compatible chat model over the same client.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
    OpenRouterStreamedResponse,
)
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

from backend.copilot.anthropic_rate_card import get_max_output_tokens
from backend.copilot.baseline.reasoning import (
    anthropic_thinking_extra_body,
    reasoning_extra_body,
)
from backend.copilot.baseline.service import (
    _apply_skills_cache_breakpoint,
    _fresh_anthropic_caching_headers,
    _fresh_ephemeral_cache_control,
    _is_anthropic_model,
    _mark_tools_with_cache_control,
    _resolve_baseline_model,
    _supports_prompt_cache_markers,
)
from backend.copilot.config import ChatConfig, CopilotLLMModel
from backend.copilot.model import RoutingSource
from backend.copilot.model_normalize import normalize_model_for_transport
from backend.copilot.service import _get_main_client

logger = logging.getLogger(__name__)

Provider = Literal["openrouter", "anthropic", "local"]


class PaiRoute(BaseModel):
    """The model a turn runs on, the routing layer that chose it, and the
    endpoint the request goes to."""

    model: str
    source: RoutingSource
    provider: Provider


async def resolve_route(
    tier: CopilotLLMModel | None, user_id: str | None, config: ChatConfig
) -> PaiRoute:
    """The baseline's routing, including its tier-default fallback when a
    per-user slug cannot run on this transport."""
    resolved = await _resolve_baseline_model(tier, user_id)
    source: RoutingSource = resolved.source
    try:
        model = normalize_model_for_transport(resolved.model, config)
    except ValueError:
        default = (
            config.fast_advanced_model
            if tier == "advanced"
            else config.fast_standard_model
        )
        model = normalize_model_for_transport(default, config)
        source = "env"
        logger.warning(
            f"[PAI] Model {resolved.model!r} rejected for this transport; "
            f"falling back to tier default {model}"
        )
    return PaiRoute(model=model, source=source, provider=config.baseline_provider)


def build_model(
    route: PaiRoute, config: ChatConfig, static_instructions: str
) -> tuple[Model, ModelSettings]:
    """The model object and its per-turn settings for *route*."""
    if route.provider == "anthropic":
        return _anthropic_model(route.model, config)
    if route.provider == "local":
        provider = OpenAIProvider(openai_client=_get_main_client())
        return OpenAIChatModel(route.model, provider=provider), ModelSettings()
    return _openrouter_model(route.model, config, static_instructions)


def response_cost_usd(response: ModelResponse) -> float | None:
    """The provider-reported USD cost of one model response, if any."""
    raw = (response.provider_details or {}).get("cost")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _openrouter_model(
    model: str, config: ChatConfig, static_instructions: str
) -> tuple[Model, ModelSettings]:
    provider = OpenRouterProvider(openai_client=_get_main_client())
    built = CopilotOpenRouterModel(
        model,
        provider=provider,
        static_instructions=static_instructions,
        cache_markers=_supports_prompt_cache_markers(model),
    )
    settings = OpenRouterModelSettings(openrouter_usage={"include": True})
    reasoning = reasoning_extra_body(model, config.claude_agent_max_thinking_tokens)
    if reasoning:
        settings["openrouter_reasoning"] = reasoning["reasoning"]
    if _is_anthropic_model(model):
        settings["extra_headers"] = _fresh_anthropic_caching_headers()
    return built, settings


def _anthropic_model(model: str, config: ChatConfig) -> tuple[Model, ModelSettings]:
    provider = AnthropicProvider(api_key=config.direct_anthropic_api_key)
    ttl: Literal["5m", "1h"] = (
        "5m" if config.baseline_prompt_cache_ttl == "5m" else "1h"
    )
    settings = AnthropicModelSettings(
        anthropic_cache_instructions=ttl, anthropic_cache_tool_definitions=ttl
    )
    thinking = anthropic_thinking_extra_body(
        model, config.claude_agent_max_thinking_tokens
    )
    if thinking:
        # Anthropic needs max_tokens above the budget; the baseline's numbers.
        model_max = get_max_output_tokens(model)
        budget = min(config.claude_agent_max_thinking_tokens, model_max - 1)
        settings["anthropic_thinking"] = {"type": "enabled", "budget_tokens": budget}
        settings["max_tokens"] = min(budget + 4096, model_max)
    return AnthropicModel(model, provider=provider), settings


class CopilotOpenRouterModel(OpenRouterModel):
    """OpenRouter with the baseline's prompt-cache markers and cost capture."""

    def __init__(
        self,
        model_name: str,
        *,
        provider: OpenRouterProvider,
        static_instructions: str,
        cache_markers: bool,
    ) -> None:
        super().__init__(model_name, provider=provider)
        self._static_instructions = static_instructions.strip()
        self._cache_markers = cache_markers

    @property
    def _streamed_response_cls(self):
        return CostCapturingStreamedResponse

    async def _map_messages(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> list[ChatCompletionMessageParam]:
        mapped = await super()._map_messages(messages, model_request_parameters)
        if not self._cache_markers:
            return mapped
        as_dicts = cast(list[dict[str, Any]], mapped)
        if as_dicts and as_dicts[0].get("role") == "system":
            as_dicts[0] = mark_static_prefix(as_dicts[0], self._static_instructions)
        return cast(
            list[ChatCompletionMessageParam], _apply_skills_cache_breakpoint(as_dicts)
        )

    def _get_tools(
        self, model_request_parameters: ModelRequestParameters
    ) -> list[ChatCompletionToolParam]:
        tools = super()._get_tools(model_request_parameters)
        if not self._cache_markers or not tools:
            return tools
        return cast(
            list[ChatCompletionToolParam], _mark_tools_with_cache_control(tools)
        )


def mark_static_prefix(system: dict[str, Any], static: str) -> dict[str, Any]:
    """Split the system message at the static/dynamic seam and put the cache
    breakpoint on the static half, so a new ``<turn_context>`` never misses the
    cache for the instructions in front of it."""
    content = system.get("content")
    if not isinstance(content, str) or not content:
        return system
    head, tail = (
        (static, content[len(static) :])
        if content.startswith(static)
        else (content, "")
    )
    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": head,
            "cache_control": _fresh_ephemeral_cache_control(),
        }
    ]
    if tail.strip():
        blocks.append({"type": "text", "text": tail})
    return {**system, "content": blocks}


@dataclass
class CostCapturingStreamedResponse(OpenRouterStreamedResponse):
    """OpenRouter puts ``usage.cost`` on a trailing chunk with no choices,
    which the stock stream skips before reading provider details; keep it."""

    _reported_cost: float | None = field(default=None, init=False)

    def _map_usage(self, response: Any):
        usage = response.usage
        if usage is not None:
            cost = usage.model_dump().get("cost")
            if isinstance(cost, (int, float)):
                self._reported_cost = (self._reported_cost or 0.0) + float(cost)
        return super()._map_usage(response)

    def get(self) -> ModelResponse:
        response = super().get()
        if self._reported_cost is not None:
            response.provider_details = {
                **(response.provider_details or {}),
                "cost": self._reported_cost,
            }
        return response
