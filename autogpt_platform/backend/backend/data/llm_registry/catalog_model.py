"""Catalog payload schema — the single source of truth for the bundled
``catalog.json`` file, the public ``/api/llm/catalog`` endpoint response, and
the remote-sync client's validation.

Carries model FACTS only. Deliberately excluded: ``LlmModelCost`` rows (cloud
credit pricing never leaves the cloud DB), provider credentials, routing
cells (per-install config, not catalog data), and non-GA-visibility models
(in-rollout models are not for distribution).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CATALOG_SCHEMA_VERSION = 1

# Guard against a hostile/corrupt payload expanding into unbounded DB writes.
MAX_CATALOG_MODELS = 2000

_NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,99}$"
# Model slugs are provider-prefixed and may contain "/" (e.g. "openai/gpt-4o").
_SLUG_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9/._:-]{0,199}$"


class CatalogProvider(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=_NAME_PATTERN)
    display_name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    metadata: dict[str, Any] = {}


class CatalogCreator(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=_NAME_PATTERN)
    display_name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    website_url: str | None = None
    logo_url: str | None = None


class CatalogModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str = Field(pattern=_SLUG_PATTERN)
    display_name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    provider: str = Field(pattern=_NAME_PATTERN)  # FK by CatalogProvider.name
    creator: str | None = Field(default=None, pattern=_NAME_PATTERN)
    kind: str = "CHAT"
    context_window: int = Field(gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    price_tier: int = Field(default=1, ge=1, le=3)
    is_enabled: bool = True
    is_recommended: bool = False
    fallback_model_slug: str | None = Field(default=None, pattern=_SLUG_PATTERN)
    supports_tools: bool = False
    supports_json_output: bool = False
    supports_reasoning: bool = False
    supports_parallel_tool_calls: bool = False
    capabilities: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class CatalogPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    generated_at: datetime
    providers: list[CatalogProvider]
    creators: list[CatalogCreator]
    models: list[CatalogModel] = Field(max_length=MAX_CATALOG_MODELS)
