"""Pydantic models for LLM registry public API."""

from __future__ import annotations

from typing import Any

import pydantic


class LlmModelCost(pydantic.BaseModel):
    """Cost configuration for an LLM model."""

    unit: str  # "RUN" or "TOKENS"
    credit_cost: int = pydantic.Field(ge=0)
    credential_provider: str
    credential_id: str | None = None
    credential_type: str | None = None
    currency: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class LlmModelCreator(pydantic.BaseModel):
    """Represents the organization that created/trained the model."""

    id: str
    name: str
    display_name: str
    description: str | None = None
    website_url: str | None = None
    logo_url: str | None = None


class LlmModel(pydantic.BaseModel):
    """Public-facing LLM model information."""

    slug: str
    display_name: str
    description: str | None = None
    provider_name: str
    creator: LlmModelCreator | None = None
    context_window: int
    max_output_tokens: int | None = None
    price_tier: int  # 1=cheapest, 2=medium, 3=expensive
    is_enabled: bool = True
    is_recommended: bool = False
    capabilities: dict[str, Any] = pydantic.Field(default_factory=dict)
    costs: list[LlmModelCost] = pydantic.Field(default_factory=list)


class LlmProvider(pydantic.BaseModel):
    """Provider with its enabled models."""

    name: str
    display_name: str
    models: list[LlmModel] = pydantic.Field(default_factory=list)


class LlmModelsResponse(pydantic.BaseModel):
    """Response for GET /llm/models."""

    models: list[LlmModel]
    total: int


class LlmProvidersResponse(pydantic.BaseModel):
    """Response for GET /llm/providers."""

    providers: list[LlmProvider]
