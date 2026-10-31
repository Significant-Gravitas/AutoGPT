from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import exa
from ._exa_rest import EXA_API_BASE, ExaRestMixin
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .content_payload import (
    process_extras,
    process_freshness,
    process_highlights,
    process_summary,
    process_text_advanced,
)
from .contents_models import ContentStatus, ExaContentsResponse
from .helpers import (
    CostDollars,
    ExaSearchResults,
    ExtrasSettings,
    HighlightSettings,
    LivecrawlTypes,
    PageSection,
    SummarySettings,
    TextAdvanced,
    TextVerbosity,
    merge_exa_cost,
    parse_cost_dollars,
)
from .request_checks import checked_user_identifier


class ExaContentsBlock(ExaRestMixin, Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        urls: list[str] = SchemaField(
            description="Up to 100 page URLs to fetch (preferred over 'ids')",
            default_factory=list,
            advanced=False,
        )
        ids: list[str] = SchemaField(
            description="[DEPRECATED - use 'urls' instead] Array of document IDs "
            "obtained from searches",
            default_factory=list,
            advanced=True,
        )
        text: bool = SchemaField(
            description="Return each page's text",
            default=True,
        )
        text_max_characters: int | None = SchemaField(
            description="Maximum characters of text per page (1 to 1,000,000). "
            "Leave empty for the full page.",
            default=None,
            ge=1,
            le=1_000_000,
            placeholder="10000",
            advanced=True,
        )
        text_include_html_tags: bool = SchemaField(
            description="Keep lightweight HTML tags in the text so an LLM can see "
            "the page structure",
            default=True,
            advanced=True,
        )
        text_verbosity: TextVerbosity | None = SchemaField(
            description="How much of each page to render: 'compact' (Exa's "
            "default) keeps the main content, 'standard' adds surrounding "
            "context, 'full' returns the most complete text",
            default=None,
            advanced=True,
        )
        text_include_sections: list[PageSection] = SchemaField(
            description="Only keep text from these page sections (best effort)",
            default_factory=list,
            advanced=True,
        )
        text_exclude_sections: list[PageSection] = SchemaField(
            description="Drop text from these page sections, e.g. navigation and "
            "footer (best effort)",
            default_factory=list,
            advanced=True,
        )
        highlights: HighlightSettings = SchemaField(
            description="The passages of each page most relevant to a query. Sent "
            "only when a query, max characters or (deprecated) sentence count is set",
            default=HighlightSettings(),
        )
        summary: SummarySettings = SchemaField(
            description="LLM-generated summary of each page (billed per page). Sent "
            "only when a query or schema is set",
            default=SummarySettings(),
        )
        max_age_hours: int | None = SchemaField(
            description="Maximum age of cached content, in hours (-1 to 720). Older "
            "pages are fetched fresh; 0 always fetches fresh, -1 only uses the "
            "cache. Leave empty (recommended) to use the cache and fetch pages that "
            "have no cached copy. Overrides Livecrawl.",
            default=None,
            ge=-1,
            le=720,
            placeholder="24",
            advanced=True,
        )
        snapshot_as_of: str | None = SchemaField(
            description="Return each page as Exa stored it at this time: an ISO "
            "8601 date-time or a date (midnight UTC). Never fetches live pages, so "
            "it cannot be combined with Max Age Hours, Livecrawl or Subpages.",
            default=None,
            placeholder="2026-01-15",
            advanced=True,
        )
        livecrawl: LivecrawlTypes | None = SchemaField(
            description="Deprecated by Exa: use Max Age Hours instead. Translated "
            "as always -> 0, never -> -1, fallback (default) -> Exa's default; "
            "'preferred' has no equivalent and is passed through.",
            default=LivecrawlTypes.FALLBACK,
            advanced=True,
        )
        livecrawl_timeout: int | None = SchemaField(
            description="How long a fresh page fetch may take, in milliseconds "
            "(1 to 90000)",
            default=10000,
            ge=1,
            le=90000,
            advanced=True,
        )
        subpages: int | None = SchemaField(
            description="Number of subpages to crawl per page (0 to 100)",
            default=0,
            ge=0,
            le=100,
            advanced=True,
        )
        subpage_target: str | list[str] | None = SchemaField(
            description="Keyword(s) to find specific subpages of search results",
            default=None,
            advanced=True,
        )
        extras: ExtrasSettings = SchemaField(
            description="Extra links and images to return for each page",
            default=ExtrasSettings(),
            advanced=True,
        )
        user_identifier: str | None = SchemaField(
            description="Opaque ID of the end user this request is for (1-64 "
            "letters, digits, '_' or '-', e.g. a UUID or hash). Never an email "
            "or name.",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[ExaSearchResults] = SchemaField(
            description="List of document contents with metadata"
        )
        result: ExaSearchResults = SchemaField(
            description="Single document content result"
        )
        context: str = SchemaField(
            description="Deprecated by Exa: the pages as one string for LLMs. Prefer "
            "the text or highlights of each result."
        )
        request_id: str = SchemaField(description="Unique identifier for the request")
        statuses: list[ContentStatus] = SchemaField(
            description="Status information for each requested URL"
        )
        cost_dollars: CostDollars | None = SchemaField(
            description="Cost breakdown for the request"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="c52be83f-f8cd-4180-b243-af35f986b461",
            description="Fetches clean text, highlights or summaries for known "
            "URLs with Exa Contents, using cached copies or fresh fetches",
            categories={BlockCategory.SEARCH},
            input_schema=ExaContentsBlock.Input,
            output_schema=ExaContentsBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "urls": ["https://example.com"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "results",
                    [ExaSearchResults(id="https://example.com", text="Example")],
                ),
                ("result", ExaSearchResults(id="https://example.com", text="Example")),
                (
                    "statuses",
                    [ContentStatus(id="https://example.com", status="success")],
                ),
                ("request_id", "req-1"),
                ("cost_dollars", CostDollars(total=0.001)),
            ],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    "requestId": "req-1",
                    "results": [{"id": "https://example.com", "text": "Example"}],
                    "statuses": [{"id": "https://example.com", "status": "success"}],
                    "costDollars": {"total": 0.001},
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        payload = build_contents_payload(input_data)
        data = await self.call_exa(
            credentials, "POST", f"{EXA_API_BASE}/contents", body=payload
        )
        response = ExaContentsResponse.model_validate(data)

        yield "results", response.results
        for result in response.results:
            yield "result", result
        if response.context:
            yield "context", response.context
        if response.statuses:
            yield "statuses", response.statuses
        if response.requestId:
            yield "request_id", response.requestId
        cost = parse_cost_dollars(data)
        if cost is not None:
            yield "cost_dollars", cost
        merge_exa_cost(self, data)


def build_contents_payload(input_data: ExaContentsBlock.Input) -> dict[str, Any]:
    """Build the ``/contents`` request body; content options sit at top level.

    Options left at their "off" defaults are omitted, matching what the block
    has always sent. ``context`` is still requested (deprecated by Exa) because
    the ``context`` output depends on it.
    """
    if not input_data.urls and not input_data.ids:
        raise ValueError("Either 'urls' or 'ids' must be provided")
    payload: dict[str, Any] = (
        {"urls": input_data.urls} if input_data.urls else {"ids": input_data.ids}
    )
    if input_data.text:
        payload["text"] = contents_text_options(input_data)
    if highlights_requested(input_data.highlights):
        payload["highlights"] = process_highlights(input_data.highlights)
    summary = process_summary(input_data.summary)
    if summary:
        payload["summary"] = summary
    payload.update(
        process_freshness(
            max_age_hours=input_data.max_age_hours,
            livecrawl=input_data.livecrawl,
            livecrawl_timeout=input_data.livecrawl_timeout,
            snapshot_as_of=input_data.snapshot_as_of,
        )
    )
    if input_data.subpages:
        payload["subpages"] = input_data.subpages
    if input_data.subpage_target is not None:
        payload["subpageTarget"] = input_data.subpage_target
    extras = process_extras(input_data.extras)
    if extras:
        payload["extras"] = extras
    user_identifier = checked_user_identifier(input_data.user_identifier)
    if user_identifier:
        payload["userIdentifier"] = user_identifier
    payload["context"] = True
    return payload


def contents_text_options(input_data: ExaContentsBlock.Input) -> dict[str, Any] | bool:
    options = process_text_advanced(
        TextAdvanced(
            max_characters=input_data.text_max_characters,
            include_html_tags=input_data.text_include_html_tags,
            verbosity=input_data.text_verbosity,
            include_sections=input_data.text_include_sections,
            exclude_sections=input_data.text_exclude_sections,
        )
    )
    return options or True


def highlights_requested(highlights: HighlightSettings) -> bool:
    """Whether highlights differ from the block's "off" default.

    The legacy defaults (1 sentence, 1 per URL) also count as off, as before.
    """
    return bool(
        highlights.query
        or highlights.max_characters is not None
        or (highlights.num_sentences or 1) != 1
        or (highlights.highlights_per_url or 1) != 1
    )
