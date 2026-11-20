"""Response models shared by the Exa Search, Find Similar, Contents and Answer
blocks.

Re-exported from ``helpers.py``. Field names mirror Exa's camelCase response
keys so a raw response item validates directly; unknown keys are ignored so new
Exa fields never break parsing.
"""

from typing import Any, Literal

from pydantic import ConfigDict, model_validator

from backend.sdk import BaseModel, MediaFileType, SchemaField


class RichImageLink(BaseModel):
    url: str = SchemaField(description="The URL of the image")
    alt: str | None = SchemaField(
        default=None, description="The image's alt text, when the page provides one"
    )


class ExaSearchExtras(BaseModel):
    links: list[str] = SchemaField(
        default_factory=list, description="Array of links from the search result"
    )
    imageLinks: list[str] = SchemaField(
        default_factory=list, description="Array of image links from the search result"
    )
    richImageLinks: list[RichImageLink] = SchemaField(
        default_factory=list,
        description="Images with their alt text, in the order they appear on the page",
    )


class ExaSearchEntity(BaseModel):
    """Structured company, person or publication data attached to a result."""

    model_config = ConfigDict(extra="ignore")

    id: str = SchemaField(description="Stable entity identifier")
    type: str = SchemaField(description="Entity kind: company, person or publication")
    version: int = SchemaField(default=1, description="Entity schema version")
    properties: dict[str, Any] = SchemaField(
        default_factory=dict,
        description="Entity fields, e.g. a company's headquarters and workforce "
        "or a publication's authors and DOI",
    )


class ExaSearchResults(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    url: str | None = None
    publishedDate: str | None = None
    author: str | None = None
    id: str = ""
    image: MediaFileType | None = None
    favicon: MediaFileType | None = None
    text: str | None = None
    highlights: list[str] = SchemaField(default_factory=list)
    highlightScores: list[float] = SchemaField(
        default_factory=list,
        description="Legacy: Exa stopped returning highlight scores; always empty",
    )
    summary: str | None = None
    subpages: list[dict] = SchemaField(default_factory=list)
    extras: ExaSearchExtras | None = None
    entities: list[ExaSearchEntity] = SchemaField(
        default_factory=list,
        description="Structured entity data for company, people and publication "
        "results",
    )
    score: float | None = SchemaField(
        default=None,
        description="Similarity to the source URL (Find Similar only; compare "
        "within one response)",
    )

    @model_validator(mode="after")
    def _default_id_to_url(self) -> "ExaSearchResults":
        """Exa's ``id`` is optional; it equals the URL when present."""
        if not self.id and self.url:
            self.id = self.url
        return self


class SearchWarning(BaseModel):
    """A note about the request (e.g. a field with no effect), for the builder
    of the workflow. Exa advises logging these, not passing them to an LLM."""

    code: str = SchemaField(description="Machine-readable code, e.g. IGNORED_FIELD")
    path: str = SchemaField(
        default="", description="JSON Pointer to the request field concerned"
    )
    message: str = SchemaField(description="What to change in the request")


class SearchNotice(BaseModel):
    """A note about how the results were produced, written for the LLM that
    reads them."""

    code: str = SchemaField(
        description="Machine-readable code, e.g. RESTRICTIONS_RELAXED"
    )
    path: str = SchemaField(
        default="", description="JSON Pointer to the request field concerned"
    )
    message: str = SchemaField(description="Explanation to pass to an LLM")
    details: dict[str, Any] = SchemaField(
        default_factory=dict, description="Optional extra context"
    )


class GroundingCitation(BaseModel):
    url: str = SchemaField(description="Source URL")
    title: str = SchemaField(default="", description="Source title")


class OutputGrounding(BaseModel):
    field: str = SchemaField(
        description="Path in the synthesized output, e.g. 'content' or "
        "'companies[0].funding'"
    )
    citations: list[GroundingCitation] = SchemaField(
        default_factory=list, description="Sources supporting this field"
    )
    confidence: Literal["low", "medium", "high"] | str = SchemaField(
        default="", description="Exa's reported reliability for this field"
    )


class SearchSynthesis(BaseModel):
    content: str | dict[str, Any] = ""
    grounding: list[OutputGrounding] = SchemaField(default_factory=list)
