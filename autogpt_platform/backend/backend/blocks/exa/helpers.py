"""Shared helpers for the Exa blocks.

The content-option models and payload builders live in ``content_settings.py``
and ``content_payload.py``, and the response models in ``result_models.py``;
they are re-exported here because blocks and saved imports use these names.
This module itself holds the cost models and billing helpers.
"""

from typing import Any

from backend.data.model import NodeExecutionStats
from backend.sdk import BaseModel, Block, SchemaField

from .content_payload import (
    process_contents_settings,
    process_context_field,
    process_highlights,
    process_text_field,
)
from .content_settings import (
    ContentSettings,
    ContextAdvanced,
    ContextDisabled,
    ContextEnabled,
    ExtrasSettings,
    HighlightSettings,
    LivecrawlTypes,
    PageSection,
    SummarySettings,
    TextAdvanced,
    TextDisabled,
    TextEnabled,
    TextVerbosity,
)
from .result_models import (
    ExaSearchEntity,
    ExaSearchExtras,
    ExaSearchResults,
    GroundingCitation,
    OutputGrounding,
    RichImageLink,
    SearchNotice,
    SearchSynthesis,
    SearchWarning,
)

__all__ = [
    "ContentSettings",
    "ContextAdvanced",
    "ContextDisabled",
    "ContextEnabled",
    "CostBreakdown",
    "CostBreakdownItem",
    "CostDollars",
    "CostDollarsContents",
    "CostDollarsSearch",
    "ExaSearchEntity",
    "ExaSearchExtras",
    "ExaSearchResults",
    "ExtrasSettings",
    "GroundingCitation",
    "HighlightSettings",
    "LivecrawlTypes",
    "OutputGrounding",
    "PageSection",
    "PerPagePrices",
    "PerRequestPrices",
    "RichImageLink",
    "SearchNotice",
    "SearchSynthesis",
    "SearchWarning",
    "SummarySettings",
    "TextAdvanced",
    "TextDisabled",
    "TextEnabled",
    "TextVerbosity",
    "extract_exa_cost_usd",
    "merge_exa_cost",
    "parse_cost_dollars",
    "process_contents_settings",
    "process_context_field",
    "process_highlights",
    "process_text_field",
]


class CostDollarsSearch(BaseModel):
    neural: float | None = SchemaField(
        default=None, description="Cost of neural search operations"
    )
    keyword: float | None = SchemaField(
        default=None, description="Cost of keyword search operations"
    )


class CostDollarsContents(BaseModel):
    text: float | None = SchemaField(
        default=None, description="Cost of text extraction"
    )
    highlights: float | None = SchemaField(
        default=None, description="Cost of highlight extraction"
    )
    summary: float | None = SchemaField(
        default=None, description="Cost of per-result summaries"
    )


# Legacy breakdown models: Exa no longer returns these shapes; kept so saved
# graphs and the CostDollars output schema stay stable.
class CostBreakdown(BaseModel):
    keywordSearch: float = SchemaField(default=0.0)
    neuralSearch: float = SchemaField(default=0.0)
    contentText: float = SchemaField(default=0.0)
    contentHighlight: float = SchemaField(default=0.0)
    contentSummary: float = SchemaField(default=0.0)


class CostBreakdownItem(BaseModel):
    search: float = SchemaField(default=0.0)
    contents: float = SchemaField(default=0.0)
    breakdown: CostBreakdown = SchemaField(default_factory=CostBreakdown)


class PerRequestPrices(BaseModel):
    neuralSearch_1_25_results: float = SchemaField(default=0.005)
    neuralSearch_26_100_results: float = SchemaField(default=0.025)
    neuralSearch_100_plus_results: float = SchemaField(default=1.0)
    keywordSearch_1_100_results: float = SchemaField(default=0.0025)
    keywordSearch_100_plus_results: float = SchemaField(default=3.0)


class PerPagePrices(BaseModel):
    contentText: float = SchemaField(default=0.001)
    contentHighlight: float = SchemaField(default=0.001)
    contentSummary: float = SchemaField(default=0.001)


class CostDollars(BaseModel):
    total: float = SchemaField(description="Estimated total dollar cost of the request")
    search: CostDollarsSearch | None = SchemaField(
        default=None, description="Search cost by retrieval mode, when reported"
    )
    summary: float | None = SchemaField(
        default=None, description="Cost of synthesized output, when reported"
    )
    contents: CostDollarsContents | None = SchemaField(
        default=None, description="Content retrieval cost breakdown, when reported"
    )
    breakDown: list[CostBreakdownItem] = SchemaField(
        default_factory=list,
        description="Legacy: no longer returned by Exa; use search/contents",
    )
    perRequestPrices: PerRequestPrices = SchemaField(
        default_factory=PerRequestPrices,
        description="Legacy: historical list prices, not returned by Exa",
    )
    perPagePrices: PerPagePrices = SchemaField(
        default_factory=PerPagePrices,
        description="Legacy: historical list prices, not returned by Exa",
    )


def parse_cost_dollars(data: dict[str, Any]) -> CostDollars | None:
    """Parse ``costDollars`` from a raw Exa response, or None when absent."""
    raw = data.get("costDollars")
    if not isinstance(raw, dict) or raw.get("total") is None:
        return None
    return CostDollars.model_validate(raw)


def extract_exa_cost_usd(response: Any) -> float | None:
    """Return ``cost_dollars.total`` (USD) from an Exa SDK response, or None.

    Handles dataclass/pydantic responses (``response.cost_dollars.total``),
    dicts with camelCase keys (``response["costDollars"]["total"]``), dicts
    with snake_case keys, and bare numeric strings. Returns None whenever the
    shape is missing cost info — the caller then skips merge_stats.
    """
    if response is None:
        return None

    # Dataclass / pydantic: response.cost_dollars
    cost_obj = getattr(response, "cost_dollars", None)

    # Dict payloads: try both camelCase and snake_case
    if cost_obj is None and isinstance(response, dict):
        cost_obj = response.get("costDollars") or response.get("cost_dollars")

    if cost_obj is None:
        return None

    # Already a scalar (code_context endpoint returns a string)
    if isinstance(cost_obj, (int, float)):
        return max(0.0, float(cost_obj))
    if isinstance(cost_obj, str):
        try:
            return max(0.0, float(cost_obj))
        except ValueError:
            return None

    # Nested object/dict: grab the `total` field
    total = getattr(cost_obj, "total", None)
    if total is None and isinstance(cost_obj, dict):
        total = cost_obj.get("total")

    if total is None:
        return None

    try:
        return max(0.0, float(total))
    except (TypeError, ValueError):
        return None


def merge_exa_cost(block: Block, response: Any) -> None:
    """Pull ``cost_dollars.total`` off an Exa response and merge it into stats.

    No-op when the response shape has no cost info (e.g. webset CRUD where
    the SDK does not expose per-call pricing) — emission happens only when
    Exa actually reports a USD amount.
    """
    cost_usd = extract_exa_cost_usd(response)
    if cost_usd is None:
        return
    block.merge_stats(
        NodeExecutionStats(
            provider_cost=cost_usd,
            provider_cost_type="cost_usd",
        )
    )
