"""
Lenient models of Websets API responses.

Statuses and other enums are plain strings here: Exa adds values over time
(``pending``, ``out_of_credits``, ``failed`` ...) and a block should report a
new value, not crash on it. Only the fields blocks read are modelled; the raw
dict is kept wherever a block passes an object through whole.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class ApiSearchProgress(_ApiModel):
    found: float = 0
    analyzed: float = 0
    completion: float = 0
    time_left: float | None = Field(default=None, alias="timeLeft")


class ApiWebsetSearch(_ApiModel):
    id: str
    webset_id: str = Field(default="", alias="websetId")
    status: str = ""
    query: str = ""
    entity: dict[str, Any] | None = None
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    count: float = 0
    max_people_per_company: int | None = Field(
        default=None, alias="maxPeoplePerCompany"
    )
    behavior: str | None = None
    exclude: list[dict[str, Any]] = Field(default_factory=list)
    scope: list[dict[str, Any]] = Field(default_factory=list)
    progress: ApiSearchProgress | None = None
    recall: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    canceled_at: str | None = Field(default=None, alias="canceledAt")
    canceled_reason: str | None = Field(default=None, alias="canceledReason")
    created_at: str | None = Field(default=None, alias="createdAt")
    updated_at: str | None = Field(default=None, alias="updatedAt")

    @property
    def entity_type(self) -> str:
        return str((self.entity or {}).get("type", "auto"))

    @property
    def found(self) -> int:
        return int(self.progress.found) if self.progress else 0


class ApiEnrichment(_ApiModel):
    id: str
    status: str = ""
    title: str | None = None
    description: str = ""
    format: str | None = None


class ApiMonitorSummary(_ApiModel):
    id: str
    status: str = ""
    next_run_at: str | None = Field(default=None, alias="nextRunAt")


class ApiWebset(_ApiModel):
    id: str
    status: str = ""
    external_id: str | None = Field(default=None, alias="externalId")
    title: str | None = None
    searches: list[ApiWebsetSearch] = Field(default_factory=list)
    enrichments: list[ApiEnrichment] = Field(default_factory=list)
    monitors: list[ApiMonitorSummary] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None
    dashboard_url: str | None = Field(default=None, alias="dashboardUrl")
    created_at: str | None = Field(default=None, alias="createdAt")
    updated_at: str | None = Field(default=None, alias="updatedAt")

    @property
    def items_found(self) -> int:
        """Items found across all searches (Exa reports no webset total)."""
        return sum(search.found for search in self.searches)

    @property
    def is_processing(self) -> bool:
        return self.status in ("running", "pending")

    @property
    def first_entity_type(self) -> str | None:
        if not self.searches or not self.searches[0].entity:
            return None
        return self.searches[0].entity_type


def recall_summary(recall: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten a search ``recall`` object into the keys blocks have always output."""
    if not recall:
        return {}
    expected = recall.get("expected") or {}
    bounds = expected.get("bounds") or {}
    return {
        "expected_total": expected.get("total", 0),
        "confidence": expected.get("confidence", ""),
        "min_expected": bounds.get("min", 0),
        "max_expected": bounds.get("max", 0),
        "reasoning": recall.get("reasoning", ""),
    }
