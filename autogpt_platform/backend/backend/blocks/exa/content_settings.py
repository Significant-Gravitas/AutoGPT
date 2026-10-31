"""Content-retrieval option models shared by the Exa Search, Find Similar and
Contents blocks.

Re-exported from ``helpers.py``; saved graphs store these field names, so they
must stay stable. Options Exa has deprecated keep their field and say so in the
description; ``content_payload.py`` translates them to the modern parameters.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import model_validator

from backend.sdk import BaseModel, SchemaField


class LivecrawlTypes(str, Enum):
    """Deprecated by Exa in favour of ``max_age_hours``."""

    NEVER = "never"
    FALLBACK = "fallback"
    ALWAYS = "always"
    PREFERRED = "preferred"


class TextVerbosity(str, Enum):
    COMPACT = "compact"
    STANDARD = "standard"
    FULL = "full"


class PageSection(str, Enum):
    HEADER = "header"
    NAVIGATION = "navigation"
    BANNER = "banner"
    BODY = "body"
    SIDEBAR = "sidebar"
    FOOTER = "footer"
    METADATA = "metadata"


class TextEnabled(BaseModel):
    discriminator: Literal["enabled"] = "enabled"


class TextDisabled(BaseModel):
    discriminator: Literal["disabled"] = "disabled"


class TextAdvanced(BaseModel):
    discriminator: Literal["advanced"] = "advanced"
    max_characters: int | None = SchemaField(
        default=None,
        description="Maximum characters of page text to return per result "
        "(1 to 1,000,000). Leave empty for the full page.",
        placeholder="10000",
        ge=1,
        le=1_000_000,
    )
    include_html_tags: bool = SchemaField(
        default=False,
        description="Keep lightweight HTML tags in the text instead of plain "
        "markdown-style text. Helps an LLM see the page structure.",
    )
    verbosity: TextVerbosity | None = SchemaField(
        default=None,
        description="How much of the page to render: 'compact' (Exa's default) "
        "keeps the main content, 'standard' adds surrounding page context, "
        "'full' returns the most complete text. HTML tags, verbosity and section "
        "filters apply to freshly fetched pages, so pair them with Max Age Hours = 0.",
    )
    include_sections: list[PageSection] = SchemaField(
        default_factory=list,
        description="Only keep text from these page sections (best effort; Exa "
        "cannot classify every page).",
    )
    exclude_sections: list[PageSection] = SchemaField(
        default_factory=list,
        description="Drop text from these page sections, e.g. navigation and "
        "footer (best effort).",
    )


class HighlightSettings(BaseModel):
    max_characters: int | None = SchemaField(
        default=None,
        description="Maximum characters of highlight text per result (1 to "
        "1,000,000). Leave empty to let Exa size highlights to each page, which "
        "Exa recommends.",
        placeholder="2000",
        ge=1,
        le=1_000_000,
    )
    num_sentences: int | None = SchemaField(
        default=None,
        description="Deprecated by Exa: use Max Characters instead. When set (and "
        "Max Characters is empty) it is sent as Max Characters = 1333 per "
        "sentence, Exa's documented mapping.",
        placeholder="1",
        ge=1,
        advanced=True,
    )
    highlights_per_url: int | None = SchemaField(
        default=None,
        description="Deprecated by Exa: ignored by the API and no longer sent.",
        placeholder="1",
        ge=1,
        advanced=True,
    )
    query: str | None = SchemaField(
        default=None,
        description="Custom query that steers which passages Exa picks as highlights",
        placeholder="Key advancements",
    )


class SummarySettings(BaseModel):
    query: str | None = SchemaField(
        default=None,
        description="Custom query for the LLM-generated summary",
        placeholder="Main developments",
    )
    output_schema: dict | None = SchemaField(
        default=None,
        description="JSON schema for structured output from summary",
        advanced=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_backward_compat(cls, data: Any) -> Any:
        """Handle backward compatibility with 'schema' -> 'output_schema' rename."""
        if isinstance(data, dict) and "schema" in data and "output_schema" not in data:
            data = data.copy()
            data["output_schema"] = data.pop("schema")
        return data


class ExtrasSettings(BaseModel):
    links: int = SchemaField(
        default=0,
        description="Number of links to return from each page (0 to 1000)",
        placeholder="1",
        ge=0,
        le=1000,
    )
    image_links: int = SchemaField(
        default=0,
        description="Number of image URLs to return for each result (0 to 1000)",
        placeholder="1",
        ge=0,
        le=1000,
    )
    rich_image_links: int = SchemaField(
        default=0,
        description="Number of images to return with their alt text, in page "
        "order (0 to 1000)",
        placeholder="1",
        ge=0,
        le=1000,
    )


class ContextEnabled(BaseModel):
    discriminator: Literal["enabled"] = "enabled"


class ContextDisabled(BaseModel):
    discriminator: Literal["disabled"] = "disabled"


class ContextAdvanced(BaseModel):
    discriminator: Literal["advanced"] = "advanced"
    max_characters: int | None = SchemaField(
        default=None,
        description="Maximum character limit for context string",
        placeholder="10000",
    )


class ContentSettings(BaseModel):
    text: bool | TextEnabled | TextDisabled | TextAdvanced | None = SchemaField(
        default=None,
        description="Full page text. Boolean for simple enable/disable or object "
        "for length, verbosity and section controls",
    )
    highlights: HighlightSettings | None = SchemaField(
        default=None,
        description="The passages of each page most relevant to the query. Exa's "
        "recommended output for most tasks",
    )
    summary: SummarySettings | None = SchemaField(
        default=None,
        description="LLM-generated summary of each page (billed per page)",
    )
    max_age_hours: int | None = SchemaField(
        default=None,
        description="Maximum age of cached page content, in hours (-1 to 720). "
        "Older pages are fetched fresh. 0 always fetches fresh content, -1 only "
        "uses the cache. Leave empty (recommended) to use the cache and fetch "
        "pages that have no cached copy. Not supported by the 'lite' search type.",
        placeholder="24",
        ge=-1,
        le=720,
        advanced=True,
    )
    snapshot_as_of: str | None = SchemaField(
        default=None,
        description="Return each page as Exa stored it at this time: an ISO 8601 "
        "date-time or a date (midnight UTC). Never fetches live pages, so it "
        "cannot be combined with Max Age Hours, Livecrawl or Subpages.",
        placeholder="2026-01-15",
        advanced=True,
    )
    livecrawl: LivecrawlTypes | None = SchemaField(
        default=None,
        description="Deprecated by Exa: use Max Age Hours instead. Translated as "
        "always -> 0, never -> -1, fallback -> Exa's default; 'preferred' has no "
        "equivalent and is passed through. Ignored when Max Age Hours is set.",
        advanced=True,
    )
    livecrawl_timeout: int | None = SchemaField(
        default=None,
        description="How long a fresh page fetch may take, in milliseconds (1 to "
        "90000, Exa default 10000)",
        placeholder="10000",
        ge=1,
        le=90000,
        advanced=True,
    )
    subpages: int | None = SchemaField(
        default=None,
        description="Number of subpages to crawl per result (0 to 100). Deprecated "
        "by Exa on search: send the result URLs to the Exa Contents block instead.",
        placeholder="0",
        ge=0,
        le=100,
        advanced=True,
    )
    subpage_target: str | list[str] | None = SchemaField(
        default=None,
        description="Keyword(s) to find specific subpages of search results. "
        "Deprecated by Exa on search: use the Exa Contents block instead.",
        advanced=True,
    )
    extras: ExtrasSettings | None = SchemaField(
        default=None,
        description="Extra links and images to return for each result",
        advanced=True,
    )
    context: bool | ContextEnabled | ContextDisabled | ContextAdvanced | None = (
        SchemaField(
            default=None,
            description="Deprecated by Exa: use highlights or text instead. Formats "
            "the results into one context string for LLMs.",
            advanced=True,
        )
    )
