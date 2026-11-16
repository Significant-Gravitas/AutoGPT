"""
Output models for the core Websets blocks (``websets.py``).

These are block outputs, so their field names are part of saved graphs: add
fields, never rename or remove them.
"""

from datetime import datetime
from typing import Annotated, Any

from pydantic import Field

from backend.sdk import BaseModel


class Webset(BaseModel):
    """A webset as returned by Exa, with nested objects kept as plain dicts."""

    id: str
    status: str | None = Field(
        default=None, description="idle, pending, running or paused"
    )
    external_id: Annotated[str | None, Field(alias="externalId")] = None
    title: str | None = None
    searches: list[dict[str, Any]] | None = None
    imports: list[dict[str, Any]] | None = None
    enrichments: list[dict[str, Any]] | None = None
    monitors: list[dict[str, Any]] | None = None
    excludes: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    dashboard_url: Annotated[str | None, Field(alias="dashboardUrl")] = None
    created_at: Annotated[datetime | None, Field(alias="createdAt")] = None
    updated_at: Annotated[datetime | None, Field(alias="updatedAt")] = None


class PreviewCriterionModel(BaseModel):
    """A criterion Exa generated from the previewed query."""

    description: str

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "PreviewCriterionModel":
        return cls(description=data.get("description", ""))


class PreviewEnrichmentModel(BaseModel):
    """An enrichment column Exa detected in the previewed query."""

    description: str
    format: str
    options: list[str]

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "PreviewEnrichmentModel":
        return cls(
            description=data.get("description", ""),
            format=str(data.get("format") or "text"),
            options=[o.get("label", "") for o in data.get("options") or []],
        )


class PreviewSearchModel(BaseModel):
    """How Exa will interpret the previewed search."""

    entity_type: str
    entity_description: str | None
    criteria: list[PreviewCriterionModel]

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "PreviewSearchModel":
        entity = data.get("entity") or {}
        return cls(
            entity_type=str(entity.get("type", "auto")),
            entity_description=entity.get("description"),
            criteria=[
                PreviewCriterionModel.from_api(c) for c in data.get("criteria") or []
            ],
        )


class PreviewWebsetModel(BaseModel):
    """The full preview response."""

    search: PreviewSearchModel
    enrichments: list[PreviewEnrichmentModel]
    items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sample matching items (only when preview items were requested)",
    )

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "PreviewWebsetModel":
        return cls(
            search=PreviewSearchModel.from_api(data.get("search") or {}),
            enrichments=[
                PreviewEnrichmentModel.from_api(e)
                for e in data.get("enrichments") or []
            ],
            items=data.get("items") or [],
        )


class SearchSummaryModel(BaseModel):
    """Summary of searches in a webset."""

    total_searches: int
    completed_searches: int
    total_items_found: int
    queries: list[str]


class EnrichmentSummaryModel(BaseModel):
    """Summary of enrichments in a webset."""

    total_enrichments: int
    completed_enrichments: int
    enrichment_types: list[str]
    titles: list[str]


class MonitorSummaryModel(BaseModel):
    """Summary of monitors in a webset."""

    total_monitors: int
    active_monitors: int
    next_run: datetime | None = None


class WebsetStatisticsModel(BaseModel):
    """Various statistics about a webset."""

    total_operations: int
    is_processing: bool
    has_monitors: bool
    avg_items_per_search: float
