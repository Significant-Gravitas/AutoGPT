"""Central environment-based configuration for chat LLM providers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, SecretStr

ChatProvider = Literal[
    "deepseek",
    "openai",
    "openrouter",
    "anthropic",
    "local",
    "custom",
]
ProviderDispatch = Literal[
    "openai",
    "deepseek",
    "custom",
    "anthropic",
    "groq",
    "ollama",
    "open_router",
    "llama_api",
    "aiml_api",
    "v0",
]

DEFAULT_REQUEST_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_RETRIES = 1


class LLMProviderProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ChatProvider
    dispatch_provider: ProviderDispatch
    default_base_url: str | None
    default_model: str
    fast_model: str
    advanced_model: str
    thinking_model: str
    title_model: str
    fallback_model: str
    openai_compatible: bool
    supports_streaming: bool = True
    supports_tool_calling: bool = True
    supports_agent_sdk: bool = False


class ResolvedLLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: ChatProvider
    dispatch_provider: ProviderDispatch
    base_url: str
    api_key: SecretStr | None = Field(default=None, exclude=True)
    api_key_source: str
    model: str
    fast_standard_model: str
    fast_advanced_model: str
    thinking_standard_model: str
    thinking_advanced_model: str
    title_model: str
    fallback_model: str
    request_timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0, le=10)
    supports_streaming: bool
    supports_tool_calling: bool
    supports_agent_sdk: bool

    @property
    def base_url_host(self) -> str:
        return urlparse(self.base_url).hostname or ""


_PROVIDER_PROFILES: dict[ChatProvider, LLMProviderProfile] = {
    "deepseek": LLMProviderProfile(
        provider="deepseek",
        dispatch_provider="deepseek",
        default_base_url="https://api.deepseek.com",
        default_model="deepseek-v4-flash",
        fast_model="deepseek-v4-flash",
        advanced_model="deepseek-v4-pro",
        thinking_model="deepseek-v4-flash",
        title_model="deepseek-v4-flash",
        fallback_model="deepseek-v4-pro",
        openai_compatible=True,
    ),
    "openai": LLMProviderProfile(
        provider="openai",
        dispatch_provider="openai",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        fast_model="gpt-4o-mini",
        advanced_model="gpt-4o",
        thinking_model="gpt-4o",
        title_model="gpt-4o-mini",
        fallback_model="gpt-4o-mini",
        openai_compatible=True,
    ),
    "openrouter": LLMProviderProfile(
        provider="openrouter",
        dispatch_provider="open_router",
        default_base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-sonnet-4-6",
        fast_model="anthropic/claude-sonnet-4-6",
        advanced_model="anthropic/claude-opus-4.7",
        thinking_model="anthropic/claude-sonnet-4-6",
        title_model="anthropic/claude-haiku-4-5",
        fallback_model="",
        openai_compatible=True,
        supports_agent_sdk=True,
    ),
    "anthropic": LLMProviderProfile(
        provider="anthropic",
        dispatch_provider="anthropic",
        default_base_url="https://api.anthropic.com/v1/",
        default_model="claude-sonnet-4-6",
        fast_model="claude-sonnet-4-6",
        advanced_model="claude-opus-4-7",
        thinking_model="claude-sonnet-4-6",
        title_model="claude-haiku-4-5",
        fallback_model="",
        openai_compatible=False,
        supports_agent_sdk=True,
    ),
    "local": LLMProviderProfile(
        provider="local",
        dispatch_provider="ollama",
        default_base_url=None,
        default_model="",
        fast_model="",
        advanced_model="",
        thinking_model="",
        title_model="",
        fallback_model="",
        openai_compatible=True,
        supports_agent_sdk=False,
    ),
    "custom": LLMProviderProfile(
        provider="custom",
        dispatch_provider="custom",
        default_base_url=None,
        default_model="",
        fast_model="",
        advanced_model="",
        thinking_model="",
        title_model="",
        fallback_model="",
        openai_compatible=True,
        supports_agent_sdk=False,
    ),
}


def get_provider_profile(provider: str) -> LLMProviderProfile:
    normalized = _normalize_provider(provider)
    return _PROVIDER_PROFILES[normalized]


def infer_chat_provider(
    base_url: str | None,
    *,
    use_local: bool = False,
    use_openrouter: bool | None = None,
) -> ChatProvider:
    if use_local:
        return "local"

    host = (urlparse(base_url).hostname or "").lower() if base_url else ""
    if host == "api.deepseek.com" or host.endswith(".api.deepseek.com"):
        return "deepseek"
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        return "openrouter"
    if host == "api.openai.com" or host.endswith(".api.openai.com"):
        return "openai"
    if host == "api.anthropic.com" or host.endswith(".api.anthropic.com"):
        return "anthropic"
    if base_url:
        return "custom"
    if use_openrouter is False:
        return "anthropic"
    return "openrouter"


def resolve_chat_config(
    environment: Mapping[str, str] | None = None,
) -> ResolvedLLMConfig:
    env = environment if environment is not None else os.environ
    explicit_provider = env.get("CHAT_PROVIDER", "").strip()
    base_url_override = env.get("CHAT_BASE_URL", "").strip() or None
    use_local = _parse_bool(env.get("CHAT_USE_LOCAL"), default=False)
    use_openrouter = _parse_optional_bool(env.get("CHAT_USE_OPENROUTER"))
    provider = (
        _normalize_provider(explicit_provider)
        if explicit_provider
        else infer_chat_provider(
            base_url_override,
            use_local=use_local,
            use_openrouter=use_openrouter,
        )
    )
    profile = _PROVIDER_PROFILES[provider]
    base_url = base_url_override or profile.default_base_url
    if not base_url:
        raise ValueError(f"CHAT_BASE_URL is required for provider={provider!r}")

    api_key, api_key_source = _resolve_api_key(env, provider)
    generic_model = env.get("CHAT_MODEL", "").strip()
    fast_standard = _first_model(
        env.get("CHAT_FAST_STANDARD_MODEL"), generic_model, profile.fast_model
    )
    fast_advanced = _first_model(
        env.get("CHAT_FAST_ADVANCED_MODEL"), profile.advanced_model, fast_standard
    )
    thinking_standard = _first_model(
        env.get("CHAT_THINKING_STANDARD_MODEL"), generic_model, profile.thinking_model
    )
    thinking_advanced = _first_model(
        env.get("CHAT_THINKING_ADVANCED_MODEL"), profile.advanced_model, fast_advanced
    )
    title_model = _first_model(
        env.get("CHAT_TITLE_MODEL"), profile.title_model, fast_standard
    )
    fallback_model = _first_model(
        env.get("CHAT_CLAUDE_AGENT_FALLBACK_MODEL"), profile.fallback_model
    )
    model = _first_model(generic_model, fast_standard, profile.default_model)
    if not model:
        raise ValueError(
            f"CHAT_MODEL or CHAT_FAST_STANDARD_MODEL is required for provider={provider!r}"
        )

    return ResolvedLLMConfig(
        provider=provider,
        dispatch_provider=profile.dispatch_provider,
        base_url=base_url,
        api_key=SecretStr(api_key) if api_key else None,
        api_key_source=api_key_source,
        model=model,
        fast_standard_model=fast_standard or model,
        fast_advanced_model=fast_advanced or model,
        thinking_standard_model=thinking_standard or model,
        thinking_advanced_model=thinking_advanced or model,
        title_model=title_model or model,
        fallback_model=fallback_model,
        request_timeout_s=_parse_positive_float(
            env.get("CHAT_REQUEST_TIMEOUT_S"), DEFAULT_REQUEST_TIMEOUT_SECONDS
        ),
        max_retries=_parse_retry_count(
            env.get("CHAT_MAX_RETRIES"), DEFAULT_MAX_RETRIES
        ),
        supports_streaming=profile.supports_streaming,
        supports_tool_calling=profile.supports_tool_calling,
        supports_agent_sdk=profile.supports_agent_sdk,
    )


def resolve_llm_request_config(
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    request_timeout_s: float | None = None,
    max_retries: int | None = None,
) -> ResolvedLLMConfig:
    profile = get_provider_profile(provider)
    resolved_base_url = base_url or profile.default_base_url
    if not resolved_base_url:
        raise ValueError(f"base_url is required for provider={profile.provider!r}")
    request_timeout, retry_count = resolve_request_policy()
    return ResolvedLLMConfig(
        provider=profile.provider,
        dispatch_provider=profile.dispatch_provider,
        base_url=resolved_base_url,
        api_key=SecretStr(api_key) if api_key else None,
        api_key_source="call_override",
        model=model,
        fast_standard_model=model,
        fast_advanced_model=model,
        thinking_standard_model=model,
        thinking_advanced_model=model,
        title_model=model,
        fallback_model="",
        request_timeout_s=(
            request_timeout_s if request_timeout_s is not None else request_timeout
        ),
        max_retries=max_retries if max_retries is not None else retry_count,
        supports_streaming=profile.supports_streaming,
        supports_tool_calling=profile.supports_tool_calling,
        supports_agent_sdk=profile.supports_agent_sdk,
    )


def resolve_request_policy(
    environment: Mapping[str, str] | None = None,
) -> tuple[float, int]:
    env = environment if environment is not None else os.environ
    return (
        _parse_positive_float(
            env.get("CHAT_REQUEST_TIMEOUT_S"), DEFAULT_REQUEST_TIMEOUT_SECONDS
        ),
        _parse_retry_count(env.get("CHAT_MAX_RETRIES"), DEFAULT_MAX_RETRIES),
    )


def _normalize_provider(provider: str) -> ChatProvider:
    normalized = provider.strip().lower().replace("-", "_")
    aliases = {"open_router": "openrouter"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in _PROVIDER_PROFILES:
        supported = ", ".join(_PROVIDER_PROFILES)
        raise ValueError(
            f"Unsupported CHAT_PROVIDER={provider!r}; expected one of: {supported}"
        )
    return cast(ChatProvider, normalized)


def _resolve_api_key(env: Mapping[str, str], provider: ChatProvider) -> tuple[str, str]:
    candidates: tuple[str, ...]
    if provider == "openrouter":
        candidates = ("CHAT_API_KEY", "OPEN_ROUTER_API_KEY", "OPENAI_API_KEY")
    elif provider == "openai":
        candidates = ("CHAT_API_KEY", "OPENAI_API_KEY")
    elif provider == "anthropic":
        candidates = (
            "CHAT_API_KEY",
            "CHAT_DIRECT_ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY",
        )
    else:
        candidates = ("CHAT_API_KEY",)
    for name in candidates:
        if value := env.get(name, "").strip():
            return value, name
    return "", "none"


def _first_model(*values: str | None) -> str:
    return next((value.strip() for value in values if value and value.strip()), "")


def _parse_bool(value: str | None, *, default: bool) -> bool:
    parsed = _parse_optional_bool(value)
    return default if parsed is None else parsed


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_positive_float(value: str | None, default: float) -> float:
    parsed = float(value) if value and value.strip() else default
    if parsed <= 0:
        raise ValueError("CHAT_REQUEST_TIMEOUT_S must be greater than zero")
    return parsed


def _parse_retry_count(value: str | None, default: int) -> int:
    parsed = int(value) if value and value.strip() else default
    if not 0 <= parsed <= 10:
        raise ValueError("CHAT_MAX_RETRIES must be between 0 and 10")
    return parsed
