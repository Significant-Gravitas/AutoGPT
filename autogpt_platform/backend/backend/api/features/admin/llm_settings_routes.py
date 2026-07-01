"""Admin API for global LLM provider settings and diagnostics."""

import logging
from datetime import datetime

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, Field, SecretStr, field_validator

from backend.data.llm_provider_settings import (
    LlmProviderSettingsWrite,
    PersistedLlmProviderSettings,
    encrypt_llm_api_key,
    get_llm_provider_settings,
    upsert_llm_provider_settings,
)
from backend.util.llm.config import (
    ChatProvider,
    get_provider_profile,
    list_provider_profiles,
)
from backend.util.llm.diagnostics import (
    ProviderDiagnosticResult,
    diagnose_chat_provider,
)
from backend.util.llm.runtime_config import (
    ConfigSource,
    EffectiveLlmConfig,
    LlmRuntimeOverrides,
    invalidate_runtime_llm_config_cache,
    resolve_effective_llm_config,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/llm",
    tags=["llm-settings", "admin"],
    dependencies=[Security(requires_admin_user)],
)

_MODEL_FIELDS = (
    "model",
    "title_model",
    "fast_standard_model",
    "fast_advanced_model",
    "thinking_standard_model",
    "thinking_advanced_model",
    "claude_agent_fallback_model",
)


class LlmSettingsUpdateRequest(BaseModel):
    enabled: bool | None = None
    provider: ChatProvider | None = None
    use_local: bool | None = None
    base_url: str | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)
    remove_api_key: bool = False
    model: str | None = None
    title_model: str | None = None
    fast_standard_model: str | None = None
    fast_advanced_model: str | None = None
    thinking_standard_model: str | None = None
    thinking_advanced_model: str | None = None
    claude_agent_fallback_model: str | None = None
    request_timeout_s: float | None = Field(default=None, ge=5, le=300)
    max_retries: int | None = Field(default=None, ge=0, le=5)
    local_request_timeout_s: float | None = Field(default=None, gt=0)
    store_embedding_model: str | None = None

    @field_validator(*_MODEL_FIELDS, "store_embedding_model")
    @classmethod
    def validate_nonempty_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("model values cannot be empty")
        return value.strip() if value is not None else None

    @field_validator("base_url")
    @classmethod
    def validate_nonempty_base_url(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("base_url cannot be empty")
        return value.strip() if value is not None else None


class LlmSettingsTestRequest(LlmSettingsUpdateRequest):
    enabled: bool | None = Field(default=None, exclude=True)
    remove_api_key: bool = Field(default=False, exclude=True)


class LlmSettingsResponse(BaseModel):
    enabled: bool
    provider: ChatProvider
    use_local: bool
    base_url: str
    api_key_configured: bool
    api_key_masked: str | None
    model: str
    title_model: str
    fast_standard_model: str
    fast_advanced_model: str
    thinking_standard_model: str
    thinking_advanced_model: str
    claude_agent_fallback_model: str
    request_timeout_s: float
    max_retries: int
    local_request_timeout_s: float
    embedding_provider: str
    store_embedding_model: str
    config_source: str
    source_by_field: dict[str, ConfigSource]
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ProviderPresetResponse(BaseModel):
    provider: ChatProvider
    base_url: str | None
    default_model: str
    models: list[str]
    supports_streaming: bool
    supports_tool_calling: bool
    supports_agent_sdk: bool
    pricing_status: str


def _settings_response(
    effective: EffectiveLlmConfig,
    persisted: PersistedLlmProviderSettings | None,
) -> LlmSettingsResponse:
    return LlmSettingsResponse(
        enabled=bool(persisted and persisted.enabled),
        provider=effective.provider,
        use_local=effective.use_local,
        base_url=effective.base_url,
        api_key_configured=effective.api_key_configured,
        api_key_masked=effective.api_key_masked,
        model=effective.model,
        title_model=effective.title_model,
        fast_standard_model=effective.fast_standard_model,
        fast_advanced_model=effective.fast_advanced_model,
        thinking_standard_model=effective.thinking_standard_model,
        thinking_advanced_model=effective.thinking_advanced_model,
        claude_agent_fallback_model=effective.fallback_model,
        request_timeout_s=effective.request_timeout_s,
        max_retries=effective.max_retries,
        local_request_timeout_s=effective.local_request_timeout_s,
        embedding_provider=effective.embedding_provider,
        store_embedding_model=effective.embedding_model,
        config_source=effective.source,
        source_by_field=effective.source_by_field,
        created_at=persisted.created_at if persisted else None,
        updated_at=persisted.updated_at if persisted else None,
    )


def _runtime_overrides(
    request: LlmSettingsUpdateRequest,
    effective: EffectiveLlmConfig,
) -> LlmRuntimeOverrides:
    data = request.model_dump(exclude={"enabled", "remove_api_key"}, exclude_none=True)
    if request.api_key is not None and request.api_key.get_secret_value():
        data["api_key"] = request.api_key

    if request.provider is not None and request.provider != effective.provider:
        profile = get_provider_profile(request.provider)
        if request.provider in {"local", "custom"}:
            if request.base_url is None or request.model is None:
                raise ValueError(
                    "Switching to local or custom requires base_url and model"
                )
            profile_model = request.model
        else:
            profile_model = profile.default_model
        defaults = {
            "use_local": request.provider == "local",
            "base_url": profile.default_base_url,
            "model": profile_model,
            "title_model": profile.title_model or profile_model,
            "fast_standard_model": profile.fast_model or profile_model,
            "fast_advanced_model": profile.advanced_model or profile_model,
            "thinking_standard_model": profile.thinking_model or profile_model,
            "thinking_advanced_model": profile.advanced_model or profile_model,
            "claude_agent_fallback_model": profile.fallback_model,
        }
        for name, value in defaults.items():
            if name not in data and value is not None:
                data[name] = value
    return LlmRuntimeOverrides(**data)


async def _resolve_for_request(
    request: LlmSettingsUpdateRequest,
) -> tuple[EffectiveLlmConfig, LlmRuntimeOverrides]:
    current = await resolve_effective_llm_config(force_refresh=True)
    overrides = _runtime_overrides(request, current)
    effective = await resolve_effective_llm_config(
        overrides=overrides,
        force_refresh=True,
    )
    return effective, overrides


@router.get("/settings", response_model=LlmSettingsResponse)
async def get_llm_settings() -> LlmSettingsResponse:
    persisted = await get_llm_provider_settings()
    effective = await resolve_effective_llm_config(force_refresh=True)
    return _settings_response(effective, persisted)


@router.put("/settings", response_model=LlmSettingsResponse)
async def update_llm_settings(
    request: LlmSettingsUpdateRequest,
    admin_user_id: str = Security(get_user_id),
) -> LlmSettingsResponse:
    existing = await get_llm_provider_settings()
    effective, _ = await _resolve_for_request(request)
    encrypted_api_key = existing.encrypted_api_key if existing else None
    if request.remove_api_key:
        encrypted_api_key = None
    elif request.api_key is not None and request.api_key.get_secret_value():
        try:
            encrypted_api_key = encrypt_llm_api_key(request.api_key.get_secret_value())
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="API key encryption is not configured",
            ) from exc

    write = LlmProviderSettingsWrite(
        enabled=(
            request.enabled
            if request.enabled is not None
            else (existing.enabled if existing else True)
        ),
        provider=effective.provider,
        use_local=effective.use_local,
        base_url=effective.base_url,
        model=effective.model,
        title_model=effective.title_model,
        fast_standard_model=effective.fast_standard_model,
        fast_advanced_model=effective.fast_advanced_model,
        thinking_standard_model=effective.thinking_standard_model,
        thinking_advanced_model=effective.thinking_advanced_model,
        claude_agent_fallback_model=effective.fallback_model,
        request_timeout_s=effective.request_timeout_s,
        max_retries=effective.max_retries,
        local_request_timeout_s=effective.local_request_timeout_s,
        store_embedding_model=effective.embedding_model,
    )
    saved = await upsert_llm_provider_settings(
        write,
        encrypted_api_key=encrypted_api_key,
    )
    invalidate_runtime_llm_config_cache()
    resolved = await resolve_effective_llm_config(force_refresh=True)
    logger.info(
        "Admin updated persisted LLM settings",
        extra={
            "json_fields": {
                "admin_user_id": admin_user_id,
                "provider": resolved.provider,
                "enabled": saved.enabled,
                "api_key_configured": resolved.api_key_configured,
            }
        },
    )
    return _settings_response(resolved, saved)


@router.post("/settings/test", response_model=ProviderDiagnosticResult)
async def test_llm_settings(
    request: LlmSettingsTestRequest,
) -> ProviderDiagnosticResult:
    effective, _ = await _resolve_for_request(request)
    return await diagnose_chat_provider(config=effective)


@router.get("/providers", response_model=list[ProviderPresetResponse])
async def get_llm_providers() -> list[ProviderPresetResponse]:
    responses: list[ProviderPresetResponse] = []
    for profile in list_provider_profiles():
        models = list(
            dict.fromkeys(
                model
                for model in (
                    profile.default_model,
                    profile.fast_model,
                    profile.advanced_model,
                    profile.thinking_model,
                    profile.title_model,
                    profile.fallback_model,
                )
                if model
            )
        )
        responses.append(
            ProviderPresetResponse(
                provider=profile.provider,
                base_url=profile.default_base_url,
                default_model=profile.default_model,
                models=models,
                supports_streaming=profile.supports_streaming,
                supports_tool_calling=profile.supports_tool_calling,
                supports_agent_sdk=profile.supports_agent_sdk,
                pricing_status=profile.pricing_status,
            )
        )
    return responses
