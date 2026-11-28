from urllib.parse import urlsplit

from web_router import (
    ExtractedDocument,
    ExtractFailedError,
    ExtractFailure,
    ExtractResponse,
    WebRouterError,
)

from backend.sdk import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError

from ._api import (
    EXTRACT_TIMEOUT,
    WebRouterFailure,
    WebRouterPage,
    keyless_router,
)


class WebRouterExtractBlock(Block):
    class Input(BlockSchemaInput):
        urls: list[str] = SchemaField(
            description="The http or https URLs to read (up to 10 per run)",
            min_length=1,
            max_length=10,
        )

    class Output(BlockSchemaOutput):
        results: list[WebRouterPage] = SchemaField(
            description="The pages that were read"
        )
        result: WebRouterPage = SchemaField(description="Single page that was read")
        failures: list[WebRouterFailure] = SchemaField(
            description="URLs that could not be read, each with the reason"
        )
        provider: str = SchemaField(
            description="The extraction provider that answered this call"
        )

    def __init__(self):
        super().__init__(
            id="d0a003ee-a6a2-482a-ae4b-1cf96b190410",
            description="Reads web pages as markdown with no API key, through WebRouter: the page is fetched by a keyless extraction provider, which falls back to the next if it fails",
            categories={BlockCategory.SEARCH},
            effect=BlockEffect.READ,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"urls": ["https://agpt.co", "https://agpt.co/missing"]},
            test_output=[
                ("results", lambda x: isinstance(x, list) and len(x) == 1),
                ("result", lambda x: x.url == "https://agpt.co"),
                (
                    "failures",
                    lambda x: [f.url for f in x] == ["https://agpt.co/missing"],
                ),
                ("provider", "nimble"),
            ],
            test_mock={
                "_extract": lambda *args, **kwargs: ExtractResponse(
                    provider="nimble",
                    documents=[
                        ExtractedDocument(
                            url="https://agpt.co",
                            title="AutoGPT",
                            content="# AutoGPT\n\nBuild, deploy and run AI agents.",
                        )
                    ],
                    failures=[
                        ExtractFailure(
                            url="https://agpt.co/missing",
                            error="the page came back with status 404",
                        )
                    ],
                )
            },
        )

    async def _extract(self, urls: list[str]) -> ExtractResponse:
        """Read the pages through a keyless WebRouter (mockable)."""
        async with keyless_router("extract") as router:
            return await router.extract(
                urls, format="markdown", timeout=EXTRACT_TIMEOUT
            )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        refused = [u for u in input_data.urls if urlsplit(u).scheme not in _SCHEMES]
        if refused:
            raise BlockInputError(
                message=f"Only http and https URLs can be read: {', '.join(refused)}",
                block_name=self.name,
                block_id=self.id,
            )
        try:
            response = await self._extract(input_data.urls)
        except ExtractFailedError as e:
            # No URL could be read: report each one, as a partial read does.
            response = ExtractResponse(provider=e.provider, failures=e.url_failures)
        except WebRouterError as e:
            raise BlockExecutionError(
                message=f"Extract failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        pages = [
            WebRouterPage(url=d.url, title=d.title, content=d.content)
            for d in response.documents
        ]
        yield "results", pages
        for page in pages:
            yield "result", page
        yield "failures", [
            WebRouterFailure(url=f.url, error=f.error) for f in response.failures
        ]
        yield "provider", response.provider


_SCHEMES = frozenset({"http", "https"})
