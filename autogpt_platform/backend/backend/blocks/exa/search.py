import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel

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
from .helpers import (
    ContentSettings,
    CostDollars,
    ExaSearchResults,
    OutputGrounding,
    SearchNotice,
    SearchSynthesis,
    SearchWarning,
    merge_exa_cost,
    parse_cost_dollars,
    process_contents_settings,
)
from .request_checks import checked_country_code, checked_user_identifier
from .search_options import (
    CATEGORY_DESCRIPTION,
    LEGACY_DEFAULT_CONTENTS,
    SEARCH_TYPE_DESCRIPTION,
    ExaSearchCategories,
    ExaSearchTypes,
    category_value,
    check_search_type_rules,
    iso_or_none,
)

logger = logging.getLogger(__name__)


class ExaSearchResponse(BaseModel):
    requestId: str = ""
    results: list[ExaSearchResults] = []
    costDollars: dict[str, Any] | None = None
    warnings: list[SearchWarning] = []
    notices: list[SearchNotice] = []
    output: SearchSynthesis | None = None
    resolvedSearchType: str | None = None
    context: str | None = None


class ExaSearchBlock(ExaRestMixin, Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        query: str = SchemaField(
            description="What to search for, in natural language. Describe the "
            "pages you want rather than listing keywords.",
            placeholder="Recent technical articles comparing hybrid and semantic "
            "retrieval for RAG",
        )
        type: ExaSearchTypes = SchemaField(
            title="Search Type",
            description=SEARCH_TYPE_DESCRIPTION,
            default=ExaSearchTypes.AUTO,
            advanced=True,
        )
        category: ExaSearchCategories | None = SchemaField(
            title="Category",
            description=CATEGORY_DESCRIPTION,
            default=None,
            advanced=True,
        )
        number_of_results: int = SchemaField(
            description="Number of results to return (1 to 100). Each result "
            "above 10 is billed extra.",
            default=10,
            ge=1,
            le=100,
            advanced=True,
        )
        include_domains: list[str] = SchemaField(
            description="Only return results from these domains or paths, e.g. "
            "'arxiv.org', 'anthropic.com/news' or '*.substack.com'",
            default_factory=list,
        )
        exclude_domains: list[str] = SchemaField(
            description="Never return results from these domains or paths",
            default_factory=list,
            advanced=True,
        )
        start_published_date: datetime | None = SchemaField(
            description="Only return pages published after this date",
            advanced=True,
            default=None,
        )
        end_published_date: datetime | None = SchemaField(
            description="Only return pages published before this date",
            advanced=True,
            default=None,
        )
        contents: ContentSettings = SchemaField(
            description="What to return for each result: text, highlights, "
            "summary, freshness. Left empty, each result gets up to 10,000 "
            "characters of page text.",
            default=ContentSettings(),
            advanced=True,
        )
        output_schema: dict[str, Any] | None = SchemaField(
            description="Ask Exa to synthesize an answer from the results. Use "
            '{"type": "text", "description": "..."} for prose or a JSON schema '
            'with "type": "object" (at most 10 properties in total and 2 levels '
            "of nesting) for structured data. The result arrives in Output "
            "Content, with sources in Output Grounding. Adds ~2 s; not supported "
            "by 'lite'; 'deep' types work best.",
            default=None,
            advanced=True,
        )
        system_prompt: str | None = SchemaField(
            description="Instructions for how Exa researches and writes the "
            "synthesized output, e.g. source preferences or what to exclude. Not "
            "supported by 'lite'.",
            placeholder="Prefer primary sources. Exclude duplicate announcements.",
            default=None,
            advanced=True,
        )
        additional_queries: list[str] = SchemaField(
            description="Up to 10 extra query variations for deep search types "
            "only ('deep-lite', 'deep', 'deep-reasoning'). Add ones that cover a "
            "genuinely different angle, not rephrasings.",
            default_factory=list,
            max_length=10,
            advanced=True,
        )
        objective: str | None = SchemaField(
            description="The broader goal this search serves: which documents "
            "should rank first, which to exclude, and which facts or figures to "
            "pull. Most useful when an AI writes the query as one step of a "
            "larger task. Up to 4096 characters.",
            default=None,
            max_length=4096,
            advanced=True,
        )
        user_location: str | None = SchemaField(
            description="The end user's two-letter ISO country code (e.g. 'US') "
            "to localize results",
            placeholder="US",
            default=None,
            advanced=True,
        )
        user_identifier: str | None = SchemaField(
            description="Opaque ID of the end user this search is for (1-64 "
            "letters, digits, '_' or '-', e.g. a UUID or hash). Lets Exa "
            "attribute usage per user. Never put an email or name here.",
            default=None,
            advanced=True,
        )
        moderation: bool = SchemaField(
            description="Filter unsafe content out of the results",
            default=False,
            advanced=True,
        )
        include_text: list[str] = SchemaField(
            description="Deprecated by Exa: put the terms in the query instead. "
            "Strings that must appear in the page text (approximate, word-level "
            "matching). Not supported by 'lite'.",
            default_factory=list,
            advanced=True,
        )
        exclude_text: list[str] = SchemaField(
            description="Deprecated by Exa: put the terms in the query instead. "
            "Strings that must not appear in the page text. Not supported by "
            "'lite'.",
            default_factory=list,
            advanced=True,
        )
        start_crawl_date: datetime | None = SchemaField(
            description="Deprecated by Exa and ignored by the API, so this block "
            "no longer sends it. Use Start Published Date instead.",
            advanced=True,
            default=None,
        )
        end_crawl_date: datetime | None = SchemaField(
            description="Deprecated by Exa and ignored by the API, so this block "
            "no longer sends it. Use End Published Date instead.",
            advanced=True,
            default=None,
        )

    class Output(BlockSchemaOutput):
        results: list[ExaSearchResults] = SchemaField(
            description="List of search results"
        )
        result: ExaSearchResults = SchemaField(description="Single search result")
        output_content: str | dict[str, Any] = SchemaField(
            description="The synthesized answer when Output Schema is set: text, "
            "or an object matching the schema"
        )
        output_grounding: list[OutputGrounding] = SchemaField(
            description="Sources and confidence for each field of Output Content"
        )
        context: str = SchemaField(
            description="Deprecated by Exa: the results as one string for LLMs, "
            "returned only when Contents > Context is enabled."
        )
        search_type: str = SchemaField(description="Legacy: no longer returned by Exa.")
        resolved_search_type: str = SchemaField(
            description="Deprecated by Exa: the search type Exa used; may be empty."
        )
        notices: list[SearchNotice] = SchemaField(
            description="Notes on how the results were produced (e.g. a 'site:' "
            "filter in the query was relaxed). Pass these to an LLM with the results."
        )
        warnings: list[SearchWarning] = SchemaField(
            description="Notes about the request itself, such as a field with no "
            "effect. Meant for whoever builds the workflow; do not pass to an LLM."
        )
        request_id: str = SchemaField(description="Exa's ID for this request")
        cost_dollars: CostDollars | None = SchemaField(
            description="Cost breakdown for the request"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="996cec64-ac40-4dde-982f-b0dc60a5824d",
            description="Searches the web with Exa and returns ranked pages with "
            "their text or highlights. Choose 'lite' for the cheapest, fastest "
            "search or 'deep' for multi-step research, and set an Output Schema "
            "to get a synthesized, cited answer.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaSearchBlock.Input,
            output_schema=ExaSearchBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "retrieval techniques for RAG",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("results", [ExaSearchResults(id="r1", url="https://example.com")]),
                ("result", ExaSearchResults(id="r1", url="https://example.com")),
                ("request_id", "req-1"),
                ("cost_dollars", CostDollars(total=0.007)),
            ],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    "requestId": "req-1",
                    "results": [{"id": "r1", "url": "https://example.com"}],
                    "costDollars": {"total": 0.007},
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        payload = build_search_payload(input_data)
        data = await self.call_exa(
            credentials, "POST", f"{EXA_API_BASE}/search", body=payload
        )
        response = ExaSearchResponse.model_validate(data)
        for warning in response.warnings:
            logger.warning(f"Exa search warning {warning.code}: {warning.message}")

        yield "results", response.results
        for result in response.results:
            yield "result", result
        if response.output is not None:
            yield "output_content", response.output.content
            yield "output_grounding", response.output.grounding
        if response.context:
            yield "context", response.context
        if response.resolvedSearchType:
            yield "resolved_search_type", response.resolvedSearchType
        if response.notices:
            yield "notices", response.notices
        if response.warnings:
            yield "warnings", response.warnings
        if response.requestId:
            yield "request_id", response.requestId
        cost = parse_cost_dollars(data)
        if cost is not None:
            yield "cost_dollars", cost
        merge_exa_cost(self, data)


def build_search_payload(input_data: ExaSearchBlock.Input) -> dict[str, Any]:
    """Build the ``/search`` request body from the block input.

    Unset options are omitted (Exa treats any sent value, even ``false`` or an
    empty list, as set). The crawl dates are never sent: Exa ignores them.
    """
    optional: dict[str, Any] = {
        "category": category_value(input_data.category),
        "userLocation": checked_country_code(input_data.user_location),
        "includeDomains": input_data.include_domains,
        "excludeDomains": input_data.exclude_domains,
        "startPublishedDate": iso_or_none(input_data.start_published_date),
        "endPublishedDate": iso_or_none(input_data.end_published_date),
        "includeText": input_data.include_text,
        "excludeText": input_data.exclude_text,
        "moderation": input_data.moderation or None,
        "outputSchema": input_data.output_schema,
        "systemPrompt": input_data.system_prompt,
        "additionalQueries": input_data.additional_queries,
        "objective": input_data.objective,
        "userIdentifier": checked_user_identifier(input_data.user_identifier),
    }
    payload: dict[str, Any] = {
        "query": input_data.query,
        "type": input_data.type.value,
        "numResults": input_data.number_of_results,
        "contents": process_contents_settings(input_data.contents)
        or LEGACY_DEFAULT_CONTENTS,
        **{key: value for key, value in optional.items() if value},
    }
    check_search_type_rules(payload)
    return payload
