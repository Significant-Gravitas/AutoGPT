from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import (
    action_field,
    block_resources_field,
    browser_rendering_field,
    geo_code_field,
    max_age_field,
    max_retries_field,
    scrape_markdown_request,
    text_field,
    timeout_field,
    url_field,
    wait_for_selector_field,
    warm_start_field,
)
from ._base import MrScraperBlockBase
from ._config import mrscraper, mrscraper_credentials_field


class MrScraperScrapeMarkdownBlock(MrScraperBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = mrscraper_credentials_field()
        url: str = url_field()
        browser_rendering: bool | None = browser_rendering_field()
        timeout: int = timeout_field()
        warm_start: bool = warm_start_field()
        block_resources: bool = block_resources_field()
        wait_for_selector: str | None = wait_for_selector_field()
        action: str | None = action_field()
        geo_code: str | None = geo_code_field()
        max_age: int = max_age_field()
        max_retries: int | None = max_retries_field()

    class Output(BlockSchemaOutput):
        markdown: str = SchemaField(description="The page as Markdown.")

    def __init__(self):
        super().__init__(
            id="d0b6898e-a400-4ed9-ab8c-256b77cd3370",
            description="Retrieve a page as clean Markdown. Browser rendering is "
            "optional.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": mrscraper.get_test_credentials().model_dump(),
                "url": "https://example.com",
                "warm_start": True,
                "max_age": 86400000,
            },
            test_credentials=mrscraper.get_test_credentials(),
            test_output=[
                ("markdown", "# Example page\n\nExample Markdown content."),
            ],
            test_mock={
                "_send": lambda *args, **kwargs: (
                    "# Example page\n\nExample Markdown content."
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        request = scrape_markdown_request(
            url=input_data.url,
            browser_rendering=input_data.browser_rendering,
            timeout=input_data.timeout,
            warm_start=input_data.warm_start,
            block_resources=input_data.block_resources,
            wait_for_selector=input_data.wait_for_selector,
            action=input_data.action,
            geo_code=input_data.geo_code,
            max_age=input_data.max_age,
            max_retries=input_data.max_retries,
        )
        yield "markdown", text_field(await self._send(credentials, request), "markdown")
