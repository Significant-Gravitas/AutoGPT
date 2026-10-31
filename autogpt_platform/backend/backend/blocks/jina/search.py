import json
import logging
import re
from urllib.parse import quote

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.jina._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    JinaCredentials,
    JinaCredentialsField,
    JinaCredentialsInput,
)
from backend.blocks.search import GetRequest
from backend.data.model import NodeExecutionStats, SchemaField
from backend.util.exceptions import BlockExecutionError
from backend.util.request import HTTPClientError, HTTPServerError, validate_url_host

logger = logging.getLogger(__name__)

_NO_RESULTS_MESSAGE_PATTERN = re.compile(
    r"no\s+search\s+results\s+available(?:\s+for\s+query\b.*)?",
    re.IGNORECASE | re.DOTALL,
)


class SearchTheWebBlock(Block, GetRequest):
    class Input(BlockSchemaInput):
        credentials: JinaCredentialsInput = JinaCredentialsField()
        query: str = SchemaField(description="The search query to search the web for")

    class Output(BlockSchemaOutput):
        results: str = SchemaField(
            description="The search results including content from top 5 URLs"
        )

    def __init__(self):
        super().__init__(
            id="87840993-2053-44b7-8da4-187ad4ee518c",
            description="This block searches the internet for the given search query.",
            categories={BlockCategory.SEARCH},
            input_schema=SearchTheWebBlock.Input,
            output_schema=SearchTheWebBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "Artificial Intelligence",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=("results", "search content"),
            test_mock={"get_request": lambda *args, **kwargs: "search content"},
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: JinaCredentials, **kwargs
    ) -> BlockOutput:
        # Encode the search query
        encoded_query = quote(input_data.query)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
        }

        # Prepend the Jina Search URL to the encoded query
        jina_search_url = f"https://s.jina.ai/{encoded_query}"

        try:
            results = await self.get_request(
                jina_search_url, headers=headers, json=False
            )
        except HTTPClientError as e:
            if _is_no_results_error(e):
                yield "results", ""
                return
            raise BlockExecutionError(
                message=f"Search failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e
        except Exception as e:
            raise BlockExecutionError(
                message=f"Search failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        # Jina Reader Search: $0.01/query on the paid tier. Fixed per-query
        # cost; routed through COST_USD so the platform cost log records
        # real USD spend (costMicrodollars) alongside the credit charge.
        self.merge_stats(
            NodeExecutionStats(provider_cost=0.01, provider_cost_type="cost_usd")
        )

        # Output the search results
        yield "results", results


class ExtractWebsiteContentBlock(Block, GetRequest):
    class Input(BlockSchemaInput):
        credentials: JinaCredentialsInput = JinaCredentialsField()
        url: str = SchemaField(description="The URL to scrape the content from")
        raw_content: bool = SchemaField(
            default=False,
            title="Raw Content",
            description="Whether to do a raw scrape of the content or use Jina-ai Reader to scrape the content",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        content: str = SchemaField(description="The scraped content from the given URL")
        error: str = SchemaField(
            description="Error message if the content cannot be retrieved"
        )

    def __init__(self):
        super().__init__(
            id="436c3984-57fd-4b85-8e9a-459b356883bd",
            description="This block scrapes the content from the given web URL.",
            categories={BlockCategory.SEARCH},
            input_schema=ExtractWebsiteContentBlock.Input,
            output_schema=ExtractWebsiteContentBlock.Output,
            test_input={
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=("content", "scraped content"),
            test_mock={"get_request": lambda *args, **kwargs: "scraped content"},
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: JinaCredentials, **kwargs
    ) -> BlockOutput:
        if input_data.raw_content:
            try:
                parsed_url, _, _ = await validate_url_host(input_data.url)
                url = parsed_url.geturl()
            except ValueError as e:
                yield "error", f"Invalid URL: {e}"
                return
            headers = {}
        else:
            url = f"https://r.jina.ai/{input_data.url}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
            }

        try:
            content = await self.get_request(url, json=False, headers=headers)
        except HTTPClientError as e:
            yield (
                "error",
                f"Client error ({e.status_code}) fetching {input_data.url}: {e}",
            )
            return
        except HTTPServerError as e:
            yield (
                "error",
                f"Server error ({e.status_code}) fetching {input_data.url}: {e}",
            )
            return
        except Exception as e:
            yield "error", f"Failed to fetch {input_data.url}: {e}"
            return

        if not content:
            yield "error", f"No content returned for {input_data.url}"
            return

        yield "content", content


def _is_no_results_error(error: HTTPClientError) -> bool:
    if error.status_code != 422:
        return False

    raw = str(error)
    body = _extract_json_body(raw)
    if body is not None:
        for key in ("message", "detail", "error"):
            value = body.get(key)
            if isinstance(value, str) and _NO_RESULTS_MESSAGE_PATTERN.fullmatch(
                value.strip()
            ):
                return True
    else:
        _, marker, message = raw.partition("Body: ")
        if not marker:
            message = raw.removeprefix("HTTP 422 Error: Unprocessable Entity - ")
        if _NO_RESULTS_MESSAGE_PATTERN.fullmatch(message.strip()):
            return True

    logger.warning("Jina returned 422 without a recognized no-results message")
    return False


def _extract_json_body(error_str: str) -> dict[str, object] | None:
    # Requests embeds Jina's {"message": "No search results available for query ..."}.
    _, marker, body = error_str.partition("Body: ")
    if not marker:
        return None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
