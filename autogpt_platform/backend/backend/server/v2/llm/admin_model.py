"""Request/response models for LLM registry admin API."""

from __future__ import annotations

from typing import Any

import pydantic


class CreateLlmProviderRequest(pydantic.BaseModel):
    name: str
    display_name: str
    description: str | None = None
    default_credential_provider: str | None = None
    default_credential_id: str | None = None
    default_credential_type: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class UpdateLlmProviderRequest(pydantic.BaseModel):
    display_name: str | None = None
    description: str | None = None
    default_credential_provider: str | None = None
    default_credential_id: str | None = None
    default_credential_type: str | None = None
    metadata: dict[str, Any] | None = None


class CreateLlmModelRequest(pydantic.BaseModel):
    slug: str
    display_name: str
    description: str | None = None
    provider_name: str
    creator_id: str | None = None
    context_window: int = pydantic.Field(gt=0)
    max_output_tokens: int | None = pydantic.Field(default=None, gt=0)
    price_tier: int = pydantic.Field(ge=1, le=3)
    is_enabled: bool = True
    is_recommended: bool = False
    supports_tools: bool = False
    supports_json_output: bool = False
    supports_reasoning: bool = False
    supports_parallel_tool_calls: bool = False
    capabilities: dict[str, Any] = pydantic.Field(default_factory=dict)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    costs: list[dict[str, Any]] = pydantic.Field(default_factory=list)


class UpdateLlmModelRequest(pydantic.BaseModel):
    display_name: str | None = None
    description: str | None = None
    creator_id: str | None = None
    context_window: int | None = pydantic.Field(default=None, gt=0)
    max_output_tokens: int | None = pydantic.Field(default=None, gt=0)
    price_tier: int | None = pydantic.Field(default=None, ge=1, le=3)
    is_enabled: bool | None = None
    is_recommended: bool | None = None
    supports_tools: bool | None = None
    supports_json_output: bool | None = None
    supports_reasoning: bool | None = None
    supports_parallel_tool_calls: bool | None = None
    capabilities: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ToggleLlmModelRequest(pydantic.BaseModel):
    is_enabled: bool
    migrate_to_slug: str | None = None
    migration_reason: str | None = None
    custom_credit_cost: int | None = None


class CreateLlmCreatorRequest(pydantic.BaseModel):
    name: str
    display_name: str
    description: str | None = None
    website_url: str | None = None
    logo_url: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class UpdateLlmCreatorRequest(pydantic.BaseModel):
    display_name: str | None = None
    description: str | None = None
    website_url: str | None = None
    logo_url: str | None = None
    metadata: dict[str, Any] | None = None


class LlmCreatorAdminResponse(pydantic.BaseModel):
    id: str
    name: str
    display_name: str
    description: str | None = None
    website_url: str | None = None
    logo_url: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class LlmModelCostAdminResponse(pydantic.BaseModel):
    unit: str
    credit_cost: float
    credential_provider: str
    credential_type: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class LlmProviderAdminResponse(pydantic.BaseModel):
    id: str
    name: str
    display_name: str
    description: str | None = None
    default_credential_provider: str | None = None
    default_credential_id: str | None = None
    default_credential_type: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    model_count: int | None = None


class LlmModelAdminResponse(pydantic.BaseModel):
    id: str
    slug: str
    display_name: str
    description: str | None = None
    provider_id: str
    creator_id: str | None = None
    context_window: int
    max_output_tokens: int | None = None
    price_tier: int
    is_enabled: bool
    is_recommended: bool
    supports_tools: bool
    supports_json_output: bool
    supports_reasoning: bool
    supports_parallel_tool_calls: bool
    capabilities: dict[str, Any] = pydantic.Field(default_factory=dict)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    creator: LlmCreatorAdminResponse | None = None
    costs: list[LlmModelCostAdminResponse] = pydantic.Field(default_factory=list)
