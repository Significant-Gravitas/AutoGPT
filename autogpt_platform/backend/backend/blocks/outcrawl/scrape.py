from typing import Any

from backend.data.model import NodeExecutionStats
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
from backend.util.exceptions import BlockExecutionError

from ._api import CREDIT_USD, OutcrawlClient, OutcrawlFormat, credits_from_response
from ._config import outcrawl

# Warning codes Outcrawl attaches to a 200 answer whose content is not the page
# that was asked for: a challenge page, an empty capture, or the site's sign-in
# page after a redirect. Returning that content as the page would mislead the
# blocks downstream, so the block fails instead.
UNUSABLE_WARNING_CODES = frozenset(
    {"blocked_interstitial", "empty_capture", "login_redirect"}
)


class OutcrawlScrapeBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = outcrawl.credentials_field(
            description="The Outcrawl integration requires an API Key."
        )
        url: str = SchemaField(description="The URL to scrape")
        formats: list[OutcrawlFormat] = SchemaField(
            description="The formats to return",
            default=[OutcrawlFormat.MARKDOWN],
        )
        only_main_content: bool = SchemaField(
            description="Only return the main content of the page, without headers, navs and footers",
            default=True,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        url: str = SchemaField(description="Final URL of the scraped page")
        status_code: int = SchemaField(description="HTTP status code of the page")
        markdown: str = SchemaField(description="The page content as markdown")
        html: str = SchemaField(description="The page content as HTML")
        links: list[str] = SchemaField(description="The links found on the page")
        metadata: dict[str, Any] = SchemaField(
            description="Page metadata such as title, description and language"
        )
        error: str = SchemaField(
            description="Error message if the scrape failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="c9f951bb-4c58-4a6b-868c-1c6620e6eb68",
            description="Scrapes a single URL with Outcrawl and returns its content as markdown, HTML or links",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": outcrawl.get_test_credentials().model_dump(),
                "url": "https://agpt.co",
            },
            test_credentials=outcrawl.get_test_credentials(),
            test_output=[
                ("url", "https://agpt.co/"),
                ("status_code", 200),
                ("markdown", lambda x: "AI agents" in x),
                ("metadata", lambda x: x.get("title") == "AutoGPT"),
            ],
            test_mock={
                "_scrape": lambda *args, **kwargs: {
                    "url": "https://agpt.co/",
                    "statusCode": 200,
                    "markdown": "AutoGPT is a platform for building AI agents.",
                    "metadata": {"title": "AutoGPT", "language": "en"},
                    "usage": {"credits": "1"},
                }
            },
            effect=BlockEffect.READ,
        )

    async def _scrape(
        self, credentials: APIKeyCredentials, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /v1/scrape and return the raw response (mockable)."""
        return await OutcrawlClient(credentials).scrape(payload)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        payload: dict[str, Any] = {
            "url": input_data.url,
            "formats": [f.value for f in input_data.formats],
            "onlyMainContent": input_data.only_main_content,
        }
        try:
            response = await self._scrape(credentials, payload)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Scrape failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        # Outcrawl bills 1 credit per scraped page.
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=credits_from_response(response, estimate=1) * CREDIT_USD,
                provider_cost_type="cost_usd",
            )
        )

        unusable = [
            w
            for w in response.get("warnings") or []
            if isinstance(w, dict) and w.get("code") in UNUSABLE_WARNING_CODES
        ]
        if unusable:
            details = "; ".join(
                f"{w['code']}: {w['message']}" if w.get("message") else w["code"]
                for w in unusable
            )
            raise BlockExecutionError(
                message=f"Outcrawl did not return the requested page ({details})",
                block_name=self.name,
                block_id=self.id,
            )

        yield "url", response.get("url") or input_data.url
        if response.get("statusCode") is not None:
            yield "status_code", response["statusCode"]
        for f in input_data.formats:
            if f == OutcrawlFormat.MARKDOWN:
                yield "markdown", response.get("markdown") or ""
            elif f == OutcrawlFormat.HTML:
                yield "html", response.get("html") or ""
            elif f == OutcrawlFormat.LINKS:
                yield "links", response.get("links") or []
        yield "metadata", response.get("metadata") or {}
