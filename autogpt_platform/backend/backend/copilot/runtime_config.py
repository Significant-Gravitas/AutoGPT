"""Bridge the central runtime LLM resolver into CoPilot's ChatConfig."""

from backend.copilot.config import ChatConfig
from backend.util.llm.runtime_config import (
    EffectiveLlmConfig,
    resolve_effective_llm_config,
)

_RUNTIME_FIELDS = (
    "provider",
    "use_local",
    "use_openrouter",
    "base_url",
    "api_key",
    "direct_anthropic_api_key",
    "aux_api_key",
    "aux_base_url",
    "fast_standard_model",
    "fast_advanced_model",
    "thinking_standard_model",
    "thinking_advanced_model",
    "title_model",
    "claude_agent_fallback_model",
    "request_timeout_s",
    "max_retries",
    "local_request_timeout_s",
)


def chat_config_from_effective(
    effective: EffectiveLlmConfig,
    *,
    base: ChatConfig | None = None,
) -> ChatConfig:
    values = (base or ChatConfig()).model_dump()
    values.update(
        provider=effective.provider,
        use_local=effective.use_local,
        use_openrouter=effective.provider != "anthropic",
        base_url=effective.base_url,
        api_key=(
            effective.api_key.get_secret_value()
            if effective.api_key is not None
            else None
        ),
        direct_anthropic_api_key=(
            effective.api_key.get_secret_value()
            if effective.provider == "anthropic" and effective.api_key is not None
            else None
        ),
        aux_api_key=None,
        aux_base_url=None,
        fast_standard_model=effective.fast_standard_model,
        fast_advanced_model=effective.fast_advanced_model,
        thinking_standard_model=effective.thinking_standard_model,
        thinking_advanced_model=effective.thinking_advanced_model,
        title_model=effective.title_model,
        claude_agent_fallback_model=effective.fallback_model,
        request_timeout_s=effective.request_timeout_s,
        max_retries=effective.max_retries,
        local_request_timeout_s=effective.local_request_timeout_s,
    )
    return ChatConfig(**values)


async def resolve_runtime_chat_config(
    *, base: ChatConfig | None = None
) -> tuple[ChatConfig, EffectiveLlmConfig]:
    effective = await resolve_effective_llm_config()
    return chat_config_from_effective(effective, base=base), effective


def apply_runtime_chat_config(target: ChatConfig, source: ChatConfig) -> bool:
    changed = any(
        getattr(target, name) != getattr(source, name) for name in _RUNTIME_FIELDS
    )
    for name in _RUNTIME_FIELDS:
        object.__setattr__(target, name, getattr(source, name))
    object.__setattr__(
        target,
        "__pydantic_fields_set__",
        target.model_fields_set | set(_RUNTIME_FIELDS),
    )
    return changed
