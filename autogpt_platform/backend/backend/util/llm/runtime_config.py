"""Runtime resolution for persisted, environment, and override LLM settings."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from backend.data.llm_provider_settings import PersistedLlmProviderSettings
from backend.util.llm.config import (
    DEFAULT_EMBEDDING_MODEL,
    ChatProvider,
    ResolvedLLMConfig,
    get_provider_profile,
    normalize_chat_provider,
    resolve_chat_config,
    validate_llm_provider_config,
)

logger = logging.getLogger(__name__)

ConfigSource = Literal["db", "env", "default", "override"]
_CACHE_TTL_SECONDS = 5.0
_cached_settings: PersistedLlmProviderSettings | None = None
_cache_loaded = False
_cache_expires_at = 0.0


class LlmRuntimeOverrides(BaseModel):
    provider: ChatProvider | None = None
    use_local: bool | None = None
    base_url: str | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)
    model: str | None = None
    title_model: str | None = None
    fast_standard_model: str | None = None
    fast_advanced_model: str | None = None
    thinking_standard_model: str | None = None
    thinking_advanced_model: str | None = None
    claude_agent_fallback_model: str | None = None
    request_timeout_s: float | None = None
    max_retries: int | None = None
    local_request_timeout_s: float | None = None
    store_embedding_model: str | None = None


class EffectiveLlmConfig(ResolvedLLMConfig):
    model_config = ConfigDict(frozen=True)

    source: ConfigSource
    source_by_field: dict[str, ConfigSource]
    persisted_enabled: bool = False

    @property
    def api_key_configured(self) -> bool:
        return self.api_key is not None

    @property
    def api_key_masked(self) -> str | None:
        return "********" if self.api_key is not None else None

    @property
    def claude_agent_fallback_model(self) -> str:
        return self.fallback_model


def invalidate_runtime_llm_config_cache() -> None:
    global _cache_loaded, _cache_expires_at
    _cache_loaded = False
    _cache_expires_at = 0.0


async def _load_persisted_settings(
    *, force_refresh: bool = False
) -> PersistedLlmProviderSettings | None:
    global _cached_settings, _cache_loaded, _cache_expires_at
    now = time.monotonic()
    if not force_refresh and _cache_loaded and now < _cache_expires_at:
        return _cached_settings

    try:
        from backend.data.llm_provider_settings import get_llm_provider_settings
        from backend.data import db

        if db.is_connected():
            _cached_settings = await get_llm_provider_settings()
        else:
            from backend.util.clients import get_database_manager_async_client

            loaded = (
                await get_database_manager_async_client().get_llm_provider_settings()
            )
            _cached_settings = (
                PersistedLlmProviderSettings.model_validate(loaded)
                if loaded is not None
                else None
            )
    except Exception as exc:
        logger.warning(
            "Persisted LLM settings unavailable; using environment configuration",
            extra={"json_fields": {"error_class": type(exc).__name__}},
        )
        _cached_settings = None
    _cache_loaded = True
    _cache_expires_at = now + _CACHE_TTL_SECONDS
    return _cached_settings


def _environment_sources(
    environment: Mapping[str, str], config: ResolvedLLMConfig
) -> dict[str, ConfigSource]:
    def source(*names: str) -> ConfigSource:
        return (
            "env"
            if any(environment.get(name, "").strip() for name in names)
            else "default"
        )

    return {
        "provider": source("CHAT_PROVIDER", "CHAT_BASE_URL", "CHAT_USE_LOCAL"),
        "use_local": source("CHAT_PROVIDER", "CHAT_USE_LOCAL"),
        "base_url": source("CHAT_BASE_URL"),
        "api_key": "env" if config.api_key_source != "none" else "default",
        "model": source("CHAT_MODEL", "CHAT_FAST_STANDARD_MODEL"),
        "title_model": source("CHAT_TITLE_MODEL"),
        "fast_standard_model": source(
            "CHAT_FAST_STANDARD_MODEL", "CHAT_FAST_MODEL", "CHAT_MODEL"
        ),
        "fast_advanced_model": source("CHAT_FAST_ADVANCED_MODEL"),
        "thinking_standard_model": source("CHAT_THINKING_STANDARD_MODEL", "CHAT_MODEL"),
        "thinking_advanced_model": source(
            "CHAT_THINKING_ADVANCED_MODEL", "CHAT_ADVANCED_MODEL"
        ),
        "claude_agent_fallback_model": source("CHAT_CLAUDE_AGENT_FALLBACK_MODEL"),
        "request_timeout_s": source("CHAT_REQUEST_TIMEOUT_S"),
        "max_retries": source("CHAT_MAX_RETRIES"),
        "local_request_timeout_s": source("CHAT_LOCAL_REQUEST_TIMEOUT_S"),
        "store_embedding_model": source("STORE_EMBEDDING_MODEL"),
    }


def merge_effective_llm_config(
    environment_config: ResolvedLLMConfig,
    *,
    environment: Mapping[str, str],
    persisted: PersistedLlmProviderSettings | None = None,
    persisted_api_key: str | None = None,
    overrides: LlmRuntimeOverrides | None = None,
) -> EffectiveLlmConfig:
    values = environment_config.model_dump()
    api_key = environment_config.api_key
    sources = _environment_sources(environment, environment_config)
    source: ConfigSource = "env" if "env" in sources.values() else "default"
    persisted_enabled = bool(persisted and persisted.enabled)

    if persisted_enabled and persisted is not None:
        provider = normalize_chat_provider(persisted.provider)
        profile = get_provider_profile(provider)
        values.update(
            provider=provider,
            dispatch_provider=profile.dispatch_provider,
            use_local=persisted.use_local or provider == "local",
            base_url=persisted.base_url or profile.default_base_url or "",
            model=persisted.model,
            title_model=persisted.title_model,
            fast_standard_model=persisted.fast_standard_model,
            fast_advanced_model=persisted.fast_advanced_model,
            thinking_standard_model=persisted.thinking_standard_model,
            thinking_advanced_model=persisted.thinking_advanced_model,
            fallback_model=persisted.claude_agent_fallback_model,
            request_timeout_s=persisted.request_timeout_s,
            max_retries=persisted.max_retries,
            local_request_timeout_s=persisted.local_request_timeout_s,
            embedding_model=persisted.store_embedding_model,
            supports_streaming=profile.supports_streaming,
            supports_tool_calling=profile.supports_tool_calling,
            supports_agent_sdk=profile.supports_agent_sdk,
        )
        sources.update(dict.fromkeys(sources, "db"))
        if persisted_api_key:
            api_key = SecretStr(persisted_api_key)
            values["api_key_source"] = "database"
        else:
            sources["api_key"] = _environment_sources(environment, environment_config)[
                "api_key"
            ]
        source = "db"

    if overrides is not None:
        override_values = overrides.model_dump(exclude_none=True)
        override_api_key = overrides.api_key
        has_api_key_override = bool(
            override_api_key is not None and override_api_key.get_secret_value()
        )
        if not override_values and not has_api_key_override:
            override_values = {}
        else:
            source = "override"
        requested_provider = override_values.pop("provider", None)
        if requested_provider:
            provider = normalize_chat_provider(requested_provider)
            profile = get_provider_profile(provider)
            provider_changed = provider != values["provider"]
            values.update(
                provider=provider,
                dispatch_provider=profile.dispatch_provider,
                supports_streaming=profile.supports_streaming,
                supports_tool_calling=profile.supports_tool_calling,
                supports_agent_sdk=profile.supports_agent_sdk,
            )
            if provider_changed:
                profile_model = override_values.get("model") or profile.default_model
                defaults = {
                    "use_local": provider == "local",
                    "base_url": profile.default_base_url,
                    "model": profile_model,
                    "title_model": profile.title_model or profile_model,
                    "fast_standard_model": profile.fast_model or profile_model,
                    "fast_advanced_model": profile.advanced_model or profile_model,
                    "thinking_standard_model": profile.thinking_model or profile_model,
                    "thinking_advanced_model": profile.advanced_model or profile_model,
                    "fallback_model": profile.fallback_model,
                }
                for name, value in defaults.items():
                    if name not in override_values and value is not None:
                        values[name] = value
                        source_name = (
                            "claude_agent_fallback_model"
                            if name == "fallback_model"
                            else name
                        )
                        sources[source_name] = "override"
        supplied_fields = set(override_values)
        if "claude_agent_fallback_model" in override_values:
            override_values["fallback_model"] = override_values.pop(
                "claude_agent_fallback_model"
            )
        if "store_embedding_model" in override_values:
            override_values["embedding_model"] = override_values.pop(
                "store_embedding_model"
            )
        values.update(override_values)
        for name in supplied_fields:
            if name not in {"api_key", "provider"}:
                sources[name] = "override"
        if requested_provider:
            sources["provider"] = "override"
        if has_api_key_override and override_api_key is not None:
            api_key = override_api_key
            values["api_key_source"] = "runtime_override"
            sources["api_key"] = "override"

    provider = normalize_chat_provider(values["provider"])
    base_url = str(values.get("base_url") or "")
    validate_llm_provider_config(
        provider=provider,
        base_url=base_url,
        models=(
            values["model"],
            values["title_model"],
            values["fast_standard_model"],
            values["fast_advanced_model"],
            values["thinking_standard_model"],
            values["thinking_advanced_model"],
            values["fallback_model"],
        ),
        request_timeout_s=float(values["request_timeout_s"]),
        max_retries=int(values["max_retries"]),
        use_local=bool(values["use_local"]),
        local_request_timeout_s=float(values["local_request_timeout_s"]),
    )
    values["api_key"] = api_key
    return EffectiveLlmConfig(
        **values,
        source=source,
        source_by_field=sources,
        persisted_enabled=persisted_enabled,
    )


async def resolve_effective_llm_config(
    environment: Mapping[str, str] | None = None,
    *,
    use_persisted: bool = True,
    overrides: LlmRuntimeOverrides | None = None,
    force_refresh: bool = False,
) -> EffectiveLlmConfig:
    env = environment if environment is not None else os.environ
    environment_config = resolve_chat_config(env)
    persisted = (
        await _load_persisted_settings(force_refresh=force_refresh)
        if use_persisted
        else None
    )
    persisted_api_key: str | None = None
    if persisted and persisted.enabled and persisted.encrypted_api_key:
        try:
            from backend.data.llm_provider_settings import decrypt_llm_api_key

            persisted_api_key = decrypt_llm_api_key(persisted.encrypted_api_key)
        except Exception as exc:
            logger.error(
                "Could not decrypt persisted LLM API key; environment fallback will be used",
                extra={"json_fields": {"error_class": type(exc).__name__}},
            )
    return merge_effective_llm_config(
        environment_config,
        environment=env,
        persisted=persisted,
        persisted_api_key=persisted_api_key,
        overrides=overrides,
    )


async def resolve_effective_request_policy() -> tuple[float, int]:
    config = await resolve_effective_llm_config()
    return config.request_timeout_s, config.max_retries


async def resolve_effective_embedding_model() -> str:
    persisted = await _load_persisted_settings()
    if persisted and persisted.enabled:
        return persisted.store_embedding_model
    return os.getenv("STORE_EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
