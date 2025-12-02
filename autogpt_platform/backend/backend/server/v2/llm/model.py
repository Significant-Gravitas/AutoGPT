from __future__ import annotations

from typing import Any, Optional

import prisma.enums
import pydantic


class LlmModelCost(pydantic.BaseModel):
    id: str
    unit: prisma.enums.LlmCostUnit = prisma.enums.LlmCostUnit.RUN
    credit_cost: int
    credential_provider: str
    credential_id: Optional[str] = None
    credential_type: Optional[str] = None
    currency: Optional[str] = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class LlmModel(pydantic.BaseModel):
    id: str
    slug: str
    display_name: str
    description: Optional[str] = None
    provider_id: str
    context_window: int
    max_output_tokens: Optional[int] = None
    is_enabled: bool = True
    capabilities: dict[str, Any] = pydantic.Field(default_factory=dict)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    costs: list[LlmModelCost] = pydantic.Field(default_factory=list)


class LlmProvider(pydantic.BaseModel):
    id: str
    name: str
    display_name: str
    description: Optional[str] = None
    default_credential_provider: Optional[str] = None
    default_credential_id: Optional[str] = None
    default_credential_type: Optional[str] = None
    supports_tools: bool = True
    supports_json_output: bool = True
    supports_reasoning: bool = False
    supports_parallel_tool: bool = False
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    models: list[LlmModel] = pydantic.Field(default_factory=list)


class LlmProvidersResponse(pydantic.BaseModel):
    providers: list[LlmProvider]


class LlmModelsResponse(pydantic.BaseModel):
    models: list[LlmModel]


class UpsertLlmProviderRequest(pydantic.BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    default_credential_provider: Optional[str] = None
    default_credential_id: Optional[str] = None
    default_credential_type: Optional[str] = "api_key"
    supports_tools: bool = True
    supports_json_output: bool = True
    supports_reasoning: bool = False
    supports_parallel_tool: bool = False
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class LlmModelCostInput(pydantic.BaseModel):
    unit: prisma.enums.LlmCostUnit = prisma.enums.LlmCostUnit.RUN
    credit_cost: int
    credential_provider: str
    credential_id: Optional[str] = None
    credential_type: Optional[str] = "api_key"
    currency: Optional[str] = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class CreateLlmModelRequest(pydantic.BaseModel):
    slug: str
    display_name: str
    description: Optional[str] = None
    provider_id: str
    context_window: int
    max_output_tokens: Optional[int] = None
    is_enabled: bool = True
    capabilities: dict[str, Any] = pydantic.Field(default_factory=dict)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    costs: list[LlmModelCostInput]


class UpdateLlmModelRequest(pydantic.BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    is_enabled: Optional[bool] = None
    capabilities: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
    provider_id: Optional[str] = None
    costs: Optional[list[LlmModelCostInput]] = None


class ToggleLlmModelRequest(pydantic.BaseModel):
    is_enabled: bool


class DeleteLlmModelResponse(pydantic.BaseModel):
    deleted_model_slug: str
    deleted_model_display_name: str
    replacement_model_slug: str
    nodes_migrated: int
    message: str


class LlmModelUsageResponse(pydantic.BaseModel):
    model_slug: str
    node_count: int

