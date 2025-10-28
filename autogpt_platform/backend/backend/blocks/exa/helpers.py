import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, TypeVar, Union

from backend.sdk import BaseModel, MediaFileType, SchemaField


class LivecrawlTypes(str, Enum):
    NEVER = "never"
    FALLBACK = "fallback"
    ALWAYS = "always"
    PREFERRED = "preferred"


# Discriminated union models for text field
class TextEnabled(BaseModel):
    discriminator: Literal["enabled"] = "enabled"


class TextDisabled(BaseModel):
    discriminator: Literal["disabled"] = "disabled"


class TextAdvanced(BaseModel):
    discriminator: Literal["advanced"] = "advanced"
    max_characters: Optional[int] = SchemaField(
        default=None,
        description="Maximum number of characters to return",
        placeholder="1000",
    )
    include_html_tags: bool = SchemaField(
        default=False,
        description="Include HTML tags in the response, helps LLMs understand text structure",
        placeholder="False",
    )


class HighlightSettings(BaseModel):
    num_sentences: int = SchemaField(
        default=1,
        description="Number of sentences per highlight",
        placeholder="1",
        ge=1,
    )
    highlights_per_url: int = SchemaField(
        default=1,
        description="Number of highlights per URL",
        placeholder="1",
        ge=1,
    )
    query: Optional[str] = SchemaField(
        default=None,
        description="Custom query to direct the LLM's selection of highlights",
        placeholder="Key advancements",
    )


class SummarySettings(BaseModel):
    query: Optional[str] = SchemaField(
        default=None,
        description="Custom query for the LLM-generated summary",
        placeholder="Main developments",
    )
    schema: Optional[dict] = SchemaField(  # type: ignore
        default=None,
        description="JSON schema for structured output from summary",
        advanced=True,
    )


class ExtrasSettings(BaseModel):
    links: int = SchemaField(
        default=0,
        description="Number of URLs to return from each webpage",
        placeholder="1",
        ge=0,
    )
    image_links: int = SchemaField(
        default=0,
        description="Number of images to return for each result",
        placeholder="1",
        ge=0,
    )


# Discriminated union models for context field
class ContextEnabled(BaseModel):
    discriminator: Literal["enabled"] = "enabled"


class ContextDisabled(BaseModel):
    discriminator: Literal["disabled"] = "disabled"


class ContextAdvanced(BaseModel):
    discriminator: Literal["advanced"] = "advanced"
    max_characters: Optional[int] = SchemaField(
        default=None,
        description="Maximum character limit for context string",
        placeholder="10000",
    )


class ContentSettings(BaseModel):
    text: Optional[Union[bool, TextEnabled, TextDisabled, TextAdvanced]] = SchemaField(
        default=None,
        description="Text content retrieval. Boolean for simple enable/disable or object for advanced settings",
    )
    highlights: Optional[HighlightSettings] = SchemaField(
        default=None,
        description="Text snippets most relevant from each page",
    )
    summary: Optional[SummarySettings] = SchemaField(
        default=None,
        description="LLM-generated summary of the webpage",
    )
    livecrawl: Optional[LivecrawlTypes] = SchemaField(
        default=None,
        description="Livecrawling options: never, fallback, always, preferred",
        advanced=True,
    )
    livecrawl_timeout: Optional[int] = SchemaField(
        default=None,
        description="Timeout for livecrawling in milliseconds",
        placeholder="10000",
        advanced=True,
    )
    subpages: Optional[int] = SchemaField(
        default=None,
        description="Number of subpages to crawl",
        placeholder="0",
        ge=0,
        advanced=True,
    )
    subpage_target: Optional[Union[str, list[str]]] = SchemaField(
        default=None,
        description="Keyword(s) to find specific subpages of search results",
        advanced=True,
    )
    extras: Optional[ExtrasSettings] = SchemaField(
        default=None,
        description="Extra parameters for additional content",
        advanced=True,
    )
    context: Optional[Union[bool, ContextEnabled, ContextDisabled, ContextAdvanced]] = (
        SchemaField(
            default=None,
            description="Format search results into a context string for LLMs",
            advanced=True,
        )
    )


# Websets Models
class WebsetEntitySettings(BaseModel):
    type: Optional[str] = SchemaField(
        default=None,
        description="Entity type (e.g., 'company', 'person')",
        placeholder="company",
    )


class WebsetCriterion(BaseModel):
    description: str = SchemaField(
        description="Description of the criterion",
        placeholder="Must be based in the US",
    )
    success_rate: Optional[int] = SchemaField(
        default=None,
        description="Success rate percentage",
        ge=0,
        le=100,
    )


class WebsetSearchConfig(BaseModel):
    query: str = SchemaField(
        description="Search query",
        placeholder="Marketing agencies based in the US",
    )
    count: int = SchemaField(
        default=10,
        description="Number of results to return",
        ge=1,
        le=100,
    )
    entity: Optional[WebsetEntitySettings] = SchemaField(
        default=None,
        description="Entity settings for the search",
    )
    criteria: Optional[list[WebsetCriterion]] = SchemaField(
        default=None,
        description="Search criteria",
    )
    behavior: Optional[str] = SchemaField(
        default="override",
        description="Behavior when updating results ('override' or 'append')",
        placeholder="override",
    )


class EnrichmentOption(BaseModel):
    label: str = SchemaField(
        description="Label for the enrichment option",
        placeholder="Option 1",
    )


class WebsetEnrichmentConfig(BaseModel):
    title: str = SchemaField(
        description="Title of the enrichment",
        placeholder="Company Details",
    )
    description: str = SchemaField(
        description="Description of what this enrichment does",
        placeholder="Extract company information",
    )
    format: str = SchemaField(
        default="text",
        description="Format of the enrichment result",
        placeholder="text",
    )
    instructions: Optional[str] = SchemaField(
        default=None,
        description="Instructions for the enrichment",
        placeholder="Extract key company metrics",
    )
    options: Optional[list[EnrichmentOption]] = SchemaField(
        default=None,
        description="Options for the enrichment",
    )


# Shared result models
class ExaSearchExtras(BaseModel):
    links: list[str] = SchemaField(
        default_factory=list, description="Array of links from the search result"
    )
    imageLinks: list[str] = SchemaField(
        default_factory=list, description="Array of image links from the search result"
    )


class ExaSearchResults(BaseModel):
    title: str | None = None
    url: str | None = None
    publishedDate: str | None = None
    author: str | None = None
    id: str
    image: MediaFileType | None = None
    favicon: MediaFileType | None = None
    text: str | None = None
    highlights: list[str] = SchemaField(default_factory=list)
    highlightScores: list[float] = SchemaField(default_factory=list)
    summary: str | None = None
    subpages: list[dict] = SchemaField(default_factory=list)
    extras: ExaSearchExtras | None = None


# Cost tracking models
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
    total: float = SchemaField(description="Total dollar cost for your request")
    breakDown: list[CostBreakdownItem] = SchemaField(
        default_factory=list, description="Breakdown of costs by operation type"
    )
    perRequestPrices: PerRequestPrices = SchemaField(
        default_factory=PerRequestPrices,
        description="Standard price per request for different operations",
    )
    perPagePrices: PerPagePrices = SchemaField(
        default_factory=PerPagePrices,
        description="Standard price per page for different content operations",
    )


# Helper functions for payload processing
def process_text_field(
    text: Union[bool, TextEnabled, TextDisabled, TextAdvanced, None]
) -> Optional[Union[bool, Dict[str, Any]]]:
    """Process text field for API payload."""
    if text is None:
        return None

    # Handle backward compatibility with boolean
    if isinstance(text, bool):
        return text
    elif isinstance(text, TextDisabled):
        return False
    elif isinstance(text, TextEnabled):
        return True
    elif isinstance(text, TextAdvanced):
        text_dict = {}
        if text.max_characters:
            text_dict["maxCharacters"] = text.max_characters
        if text.include_html_tags:
            text_dict["includeHtmlTags"] = text.include_html_tags
        return text_dict if text_dict else True
    return None


def process_contents_settings(contents: Optional[ContentSettings]) -> Dict[str, Any]:
    """Process ContentSettings into API payload format."""
    if not contents:
        return {}

    content_settings = {}

    # Handle text field (can be boolean or object)
    text_value = process_text_field(contents.text)
    if text_value is not None:
        content_settings["text"] = text_value

    # Handle highlights
    if contents.highlights:
        highlights_dict: Dict[str, Any] = {
            "numSentences": contents.highlights.num_sentences,
            "highlightsPerUrl": contents.highlights.highlights_per_url,
        }
        if contents.highlights.query:
            highlights_dict["query"] = contents.highlights.query
        content_settings["highlights"] = highlights_dict

    # Handle summary
    if contents.summary:
        summary_dict = {}
        if contents.summary.query:
            summary_dict["query"] = contents.summary.query
        if contents.summary.schema:
            summary_dict["schema"] = contents.summary.schema
        content_settings["summary"] = summary_dict

    # Handle livecrawl
    if contents.livecrawl:
        content_settings["livecrawl"] = contents.livecrawl.value

    # Handle livecrawl_timeout
    if contents.livecrawl_timeout is not None:
        content_settings["livecrawlTimeout"] = contents.livecrawl_timeout

    # Handle subpages
    if contents.subpages is not None:
        content_settings["subpages"] = contents.subpages

    # Handle subpage_target
    if contents.subpage_target:
        content_settings["subpageTarget"] = contents.subpage_target

    # Handle extras
    if contents.extras:
        extras_dict = {}
        if contents.extras.links:
            extras_dict["links"] = contents.extras.links
        if contents.extras.image_links:
            extras_dict["imageLinks"] = contents.extras.image_links
        content_settings["extras"] = extras_dict

    # Handle context within contents
    context_value = process_context_field(contents.context)
    if context_value is not None:
        content_settings["context"] = context_value

    return content_settings


def process_context_field(
    context: Union[bool, dict, ContextEnabled, ContextDisabled, ContextAdvanced, None]
) -> Optional[Union[bool, Dict[str, int]]]:
    """Process context field for API payload."""
    if context is None:
        return None

    # Handle backward compatibility with boolean
    if isinstance(context, bool):
        return context if context else None
    elif isinstance(context, dict) and "maxCharacters" in context:
        return {"maxCharacters": context["maxCharacters"]}
    elif isinstance(context, ContextDisabled):
        return None  # Don't send context field at all when disabled
    elif isinstance(context, ContextEnabled):
        return True
    elif isinstance(context, ContextAdvanced):
        if context.max_characters:
            return {"maxCharacters": context.max_characters}
        return True
    return None


def format_date_fields(
    input_data: Any, date_field_mapping: Dict[str, str]
) -> Dict[str, str]:
    """Format datetime fields for API payload."""
    formatted_dates = {}
    for input_field, api_field in date_field_mapping.items():
        value = getattr(input_data, input_field, None)
        if value:
            formatted_dates[api_field] = value.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return formatted_dates


def add_optional_fields(
    input_data: Any,
    field_mapping: Dict[str, str],
    payload: Dict[str, Any],
    process_enums: bool = False,
) -> None:
    """Add optional fields to payload if they have values."""
    for input_field, api_field in field_mapping.items():
        value = getattr(input_data, input_field, None)
        if value:  # Only add non-empty values
            if process_enums and hasattr(value, "value"):
                payload[api_field] = value.value
            else:
                payload[api_field] = value


T = TypeVar("T")


async def poll_until_complete(
    check_fn: Callable[[], tuple[bool, T]],
    timeout: int = 300,
    initial_interval: float = 5.0,
    max_interval: float = 30.0,
    backoff_factor: float = 1.5,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> T:
    """
    Generic polling function for async operations.

    Args:
        check_fn: Function that returns (is_complete, result)
        timeout: Maximum time to wait in seconds
        initial_interval: Initial wait interval between polls
        max_interval: Maximum wait interval between polls
        backoff_factor: Factor to increase interval by each iteration
        progress_callback: Optional callback for progress updates

    Returns:
        The result from check_fn when complete

    Raises:
        TimeoutError: If operation doesn't complete within timeout
    """
    start_time = time.time()
    interval = initial_interval
    attempt = 0

    while time.time() - start_time < timeout:
        attempt += 1
        is_complete, result = check_fn()

        if is_complete:
            if progress_callback:
                progress_callback(f"✓ Operation completed after {attempt} attempts")
            return result

        # Calculate remaining time
        elapsed = time.time() - start_time
        remaining = timeout - elapsed

        if progress_callback:
            progress_callback(
                f"⏳ Attempt {attempt}: Operation still in progress. "
                f"Elapsed: {int(elapsed)}s, Remaining: {int(remaining)}s"
            )

        # Wait before next poll
        wait_time = min(interval, remaining)
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        # Exponential backoff
        interval = min(interval * backoff_factor, max_interval)

    raise TimeoutError(f"Operation did not complete within {timeout} seconds")


async def poll_webset_status(
    webset_id: str,
    api_key: str,
    target_status: str = "idle",
    timeout: int = 300,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Poll a webset until it reaches the target status.

    Args:
        webset_id: Webset ID to poll
        api_key: API key for authentication
        target_status: Status to wait for (default: "idle")
        timeout: Maximum time to wait in seconds
        progress_callback: Optional callback for progress updates

    Returns:
        The webset data when target status is reached
    """
    import httpx

    def check_status() -> tuple[bool, Dict[str, Any]]:
        with httpx.Client() as client:
            response = client.get(
                f"https://api.exa.ai/v1alpha/websets/{webset_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            status = data.get("status", {}).get("type")
            is_complete = status == target_status

            if progress_callback and not is_complete:
                items_count = data.get("itemsCount", 0)
                progress_callback(f"Status: {status}, Items: {items_count}")

            return is_complete, data

    return await poll_until_complete(
        check_fn=check_status, timeout=timeout, progress_callback=progress_callback
    )


async def poll_search_completion(
    webset_id: str,
    search_id: str,
    api_key: str,
    timeout: int = 300,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Poll a search until it completes.

    Args:
        webset_id: Webset ID
        search_id: Search ID to poll
        api_key: API key for authentication
        timeout: Maximum time to wait in seconds
        progress_callback: Optional callback for progress updates

    Returns:
        The search data when complete
    """
    import httpx

    def check_search() -> tuple[bool, Dict[str, Any]]:
        with httpx.Client() as client:
            response = client.get(
                f"https://api.exa.ai/v1alpha/websets/{webset_id}/searches/{search_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            status = data.get("status")
            is_complete = status in ["completed", "failed", "cancelled"]

            if progress_callback and not is_complete:
                items_found = data.get("results", {}).get("itemsFound", 0)
                progress_callback(
                    f"Search status: {status}, Items found: {items_found}"
                )

            return is_complete, data

    return await poll_until_complete(
        check_fn=check_search, timeout=timeout, progress_callback=progress_callback
    )


async def poll_enrichment_completion(
    webset_id: str,
    enrichment_id: str,
    api_key: str,
    timeout: int = 300,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Poll an enrichment until it completes.

    Args:
        webset_id: Webset ID
        enrichment_id: Enrichment ID to poll
        api_key: API key for authentication
        timeout: Maximum time to wait in seconds
        progress_callback: Optional callback for progress updates

    Returns:
        The enrichment data when complete
    """
    import httpx

    def check_enrichment() -> tuple[bool, Dict[str, Any]]:
        with httpx.Client() as client:
            response = client.get(
                f"https://api.exa.ai/v1alpha/websets/{webset_id}/enrichments/{enrichment_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()

            status = data.get("status")
            is_complete = status in ["completed", "failed", "cancelled"]

            if progress_callback and not is_complete:
                progress = data.get("progress", {})
                processed = progress.get("processedItems", 0)
                total = progress.get("totalItems", 0)
                progress_callback(
                    f"Enrichment status: {status}, Progress: {processed}/{total}"
                )

            return is_complete, data

    return await poll_until_complete(
        check_fn=check_enrichment, timeout=timeout, progress_callback=progress_callback
    )


def format_progress_message(
    operation_type: str, current_state: str, details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format a progress message for display.

    Args:
        operation_type: Type of operation (webset, search, enrichment)
        current_state: Current state description
        details: Optional details to include

    Returns:
        Formatted progress message
    """
    message_parts = [f"[{operation_type.upper()}]", current_state]

    if details:
        detail_parts = []
        for key, value in details.items():
            detail_parts.append(f"{key}: {value}")
        if detail_parts:
            message_parts.append(f"({', '.join(detail_parts)})")

    return " ".join(message_parts)


def calculate_polling_stats(
    start_time: float, timeout: int, attempts: int
) -> Dict[str, Any]:
    """
    Calculate polling statistics.

    Args:
        start_time: Start time (from time.time())
        timeout: Maximum timeout in seconds
        attempts: Number of polling attempts made

    Returns:
        Dictionary with polling statistics
    """
    elapsed = time.time() - start_time
    remaining = max(0, timeout - elapsed)

    return {
        "elapsed_seconds": int(elapsed),
        "remaining_seconds": int(remaining),
        "attempts": attempts,
        "average_interval": elapsed / attempts if attempts > 0 else 0,
        "progress_percentage": min(100, (elapsed / timeout) * 100),
    }
