"""Catalog schema — the shape of the canonical LLM catalog file.

The catalog file (``catalog.py`` in this package) IS the model database:
model facts, per-model credit costs, and routing cells, updated by PR and
propagated by deploy (catalog-as-code). This module defines and validates
its shape; ``catalog_test.py`` is the referential-integrity guard.

Deliberately excluded: provider credentials (env/settings as always) and
anything per-install (retirement records live in the DB, see
``LlmModelMigration``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CATALOG_SCHEMA_VERSION = 1

ModelVisibility = Literal["GA", "EMPLOYEES", "ADMINS", "HIDDEN"]

# Guard against a runaway generator expanding into an unreviewable file.
MAX_CATALOG_MODELS = 2000

_NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,99}$"
# Model slugs may be provider-prefixed and contain "/" (e.g. "openai/gpt-4o").
_SLUG_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9/._:-]{0,199}$"


class CatalogProvider(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=_NAME_PATTERN)
    display_name: str = Field(min_length=1, max_length=200)
    description: str | None = None


class CatalogCreator(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=_NAME_PATTERN)
    display_name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    website_url: str | None = None
    logo_url: str | None = None


class CatalogModelCost(BaseModel):
    """Per-model credit pricing.

    ``run_credits`` is the flat credits-per-run tier; the ``*_per_1m``
    fields are credits per 1,000,000 tokens. These ARE the billing
    source: ``backend/data/block_cost_config.py`` derives its
    ``MODEL_COST``/``TOKEN_COST`` lookups from them, and
    ``catalog_test.py`` pins the values billed at cutover via the
    pre-catalog snapshot.
    """

    model_config = ConfigDict(frozen=True)

    run_credits: int | None = Field(default=None, ge=0)
    input_credits_per_1m: float | None = Field(default=None, ge=0)
    output_credits_per_1m: float | None = Field(default=None, ge=0)
    cache_read_credits_per_1m: float | None = Field(default=None, ge=0)
    cache_creation_credits_per_1m: float | None = Field(default=None, ge=0)
    # What the PROVIDER charges us (USD list price per 1M tokens), for
    # in-turn cost estimation — authored per model when it diverges from
    # the transport module's family default (e.g. copilot/moonshot.py).
    # Not the billing source; the credits fields above are what users pay.
    provider_input_usd_per_1m: float | None = Field(default=None, ge=0)
    provider_output_usd_per_1m: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _provider_usd_both_or_neither(self) -> "CatalogModelCost":
        # A half-authored provider price would silently fall back to the
        # transport module's (cheaper) family default — underpricing a
        # premium SKU with no error. Refuse the partial state at authoring.
        if (self.provider_input_usd_per_1m is None) != (
            self.provider_output_usd_per_1m is None
        ):
            raise ValueError(
                "provider_input_usd_per_1m and provider_output_usd_per_1m "
                "must be set together"
            )
        return self


class CatalogModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    slug: str = Field(pattern=_SLUG_PATTERN)
    display_name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    provider: str = Field(pattern=_NAME_PATTERN)  # FK by CatalogProvider.name
    creator: str | None = Field(default=None, pattern=_NAME_PATTERN)
    context_window: int = Field(gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    price_tier: Literal[1, 2, 3] = 1
    is_enabled: bool = True
    is_recommended: bool = False
    # Who can SEE the model. Non-GA models are excluded from block picker
    # metadata today; per-role tiers (EMPLOYEES/ADMINS) get distinct
    # treatment when the catalog-driven picker ships. Orthogonal to
    # is_enabled: is_enabled=False is the kill switch (copilot refuses,
    # picker hides); visibility="HIDDEN" still SERVES when explicitly
    # routed — the pre-launch testing state.
    visibility: ModelVisibility = "GA"
    # Standing replacement pointer: pre-fills the retirement CLI's
    # replacement and is the hook for future runtime failover.
    fallback_model_slug: str | None = Field(default=None, pattern=_SLUG_PATTERN)
    supports_tools: bool = False
    supports_json_output: bool = False
    supports_reasoning: bool = False
    supports_parallel_tool_calls: bool = False
    cost: CatalogModelCost | None = None


class CatalogPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    generated_at: datetime
    providers: list[CatalogProvider]
    creators: list[CatalogCreator]
    models: list[CatalogModel] = Field(max_length=MAX_CATALOG_MODELS)
    # surface -> mode -> tier -> model slug (e.g. routing["copilot"]["fast"]
    # ["standard"]). The admin-set config layer of model resolution: LD
    # per-user override above it, ChatConfig env defaults below it.
    routing: dict[str, dict[str, dict[str, str]]] = {}
