"""Request/response models for the LLM registry admin API."""

from __future__ import annotations

from typing import Any, Literal

import pydantic

LlmVisibility = Literal["GA", "EMPLOYEES", "ADMINS", "HIDDEN"]
LlmSubscriptionTier = Literal[
    "NO_TIER", "BASIC", "PRO", "MAX", "BUSINESS", "ENTERPRISE"
]


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
    kind: Literal["CHAT"] = "CHAT"
    visibility: LlmVisibility = "GA"
    min_subscription_tier: LlmSubscriptionTier | None = None
    fallback_model_slug: str | None = None
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
    kind: Literal["CHAT"] | None = None
    visibility: LlmVisibility | None = None
    min_subscription_tier: LlmSubscriptionTier | None = None
    fallback_model_slug: str | None = None
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


class SetLlmRouteRequest(pydantic.BaseModel):
    surface: str = "copilot"
    mode: str
    tier: str
    model_slug: str | None = None  # None deletes the cell


class LlmRouteResponse(pydantic.BaseModel):
    surface: str
    mode: str
    tier: str
    model_slug: str
    updated_at: str | None = None


class SetLlmRouteResponse(pydantic.BaseModel):
    route: LlmRouteResponse | None = None
    warnings: list[str] = pydantic.Field(default_factory=list)


class LlmRoutesListResponse(pydantic.BaseModel):
    routes: list[LlmRouteResponse] = pydantic.Field(default_factory=list)


class LlmCreatorAdminResponse(pydantic.BaseModel):
    id: str
    name: str
    display_name: str
    description: str | None = None
    website_url: str | None = None
    logo_url: str | None = None
    source: str = "SEED"
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
    source: str = "SEED"
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
    kind: str
    visibility: str
    min_subscription_tier: str | None = None
    fallback_model_slug: str | None = None
    source: str
    catalog_removed_at: str | None = None
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


class LlmMigrationAdminResponse(pydantic.BaseModel):
    id: str
    source_model_slug: str
    target_model_slug: str
    reason: str | None = None
    node_count: int
    custom_credit_cost: int | None = None
    is_reverted: bool
    reverted_at: str | None = None
    created_at: str


class LlmModelsAdminListResponse(pydantic.BaseModel):
    models: list[LlmModelAdminResponse]


class LlmProvidersAdminListResponse(pydantic.BaseModel):
    providers: list[LlmProviderAdminResponse]


class LlmCreatorsAdminListResponse(pydantic.BaseModel):
    creators: list[LlmCreatorAdminResponse]


class LlmMigrationsAdminListResponse(pydantic.BaseModel):
    migrations: list[LlmMigrationAdminResponse]
