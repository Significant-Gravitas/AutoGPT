"""Request/response models for LLM registry admin API."""

from typing import Any

from pydantic import BaseModel, Field


class CreateLlmProviderRequest(BaseModel):
    """Request model for creating an LLM provider."""

    name: str = Field(..., description="Provider identifier (e.g., 'openai', 'anthropic')")
    display_name: str = Field(..., description="Human-readable provider name")
    description: str | None = Field(None, description="Provider description")
    default_credential_provider: str | None = Field(
        None, description="Default credential system identifier"
    )
    default_credential_id: str | None = Field(None, description="Default credential ID")
    default_credential_type: str | None = Field(None, description="Default credential type")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class UpdateLlmProviderRequest(BaseModel):
    """Request model for updating an LLM provider."""

    display_name: str | None = Field(None, description="Human-readable provider name")
    description: str | None = Field(None, description="Provider description")
    default_credential_provider: str | None = Field(
        None, description="Default credential system identifier"
    )
    default_credential_id: str | None = Field(None, description="Default credential ID")
    default_credential_type: str | None = Field(None, description="Default credential type")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class CreateLlmModelRequest(BaseModel):
    """Request model for creating an LLM model."""

    slug: str = Field(..., description="Model slug (e.g., 'gpt-4', 'claude-3-opus')")
    display_name: str = Field(..., description="Human-readable model name")
    description: str | None = Field(None, description="Model description")
    provider_id: str = Field(..., description="Provider ID (UUID)")
    creator_id: str | None = Field(None, description="Creator ID (UUID)")
    context_window: int = Field(..., description="Maximum context window in tokens", gt=0)
    max_output_tokens: int | None = Field(
        None, description="Maximum output tokens (None if unlimited)", gt=0
    )
    price_tier: int = Field(..., description="Price tier (1=cheapest, 2=medium, 3=expensive)", ge=1, le=3)
    is_enabled: bool = Field(default=True, description="Whether the model is enabled")
    is_recommended: bool = Field(default=False, description="Whether the model is recommended")
    supports_tools: bool = Field(default=False, description="Supports function calling")
    supports_json_output: bool = Field(default=False, description="Supports JSON output mode")
    supports_reasoning: bool = Field(default=False, description="Supports reasoning mode")
    supports_parallel_tool_calls: bool = Field(default=False, description="Supports parallel tool calls")
    capabilities: dict[str, Any] = Field(
        default_factory=dict, description="Additional capabilities"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class UpdateLlmModelRequest(BaseModel):
    """Request model for updating an LLM model."""

    display_name: str | None = Field(None, description="Human-readable model name")
    description: str | None = Field(None, description="Model description")
    creator_id: str | None = Field(None, description="Creator ID (UUID)")
    context_window: int | None = Field(
        None, description="Maximum context window in tokens", gt=0
    )
    max_output_tokens: int | None = Field(
        None, description="Maximum output tokens (None if unlimited)", gt=0
    )
    price_tier: int | None = Field(
        None, description="Price tier (1=cheapest, 2=medium, 3=expensive)", ge=1, le=3
    )
    is_enabled: bool | None = Field(None, description="Whether the model is enabled")
    is_recommended: bool | None = Field(None, description="Whether the model is recommended")
    supports_tools: bool | None = Field(None, description="Supports function calling")
    supports_json_output: bool | None = Field(None, description="Supports JSON output mode")
    supports_reasoning: bool | None = Field(None, description="Supports reasoning mode")
    supports_parallel_tool_calls: bool | None = Field(
        None, description="Supports parallel tool calls"
    )
    capabilities: dict[str, Any] | None = Field(None, description="Additional capabilities")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")
