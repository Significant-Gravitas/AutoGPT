from typing import Any

from backend.data.execution import ExecutionContext
from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    MediaFileType,
    SchemaField,
)

from ._api import (
    DEFAULT_PROMPT,
    MrScraperError,
    block_resources_field,
    browser_rendering_field,
    error_message,
    geo_code_field,
    json_object,
    max_age_field,
    max_retries_field,
    scrape_json_request,
    timeout_field,
    url_field,
    wait_for_selector_field,
    warm_start_field,
)
from ._base import MrScraperBlockBase
from ._config import mrscraper, mrscraper_credentials_field
from ._testdata import TEST_HTML, stored_screenshot


class MrScraperScrapeJSONBlock(MrScraperBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = mrscraper_credentials_field()
        url: str = url_field()
        prompt: str = SchemaField(
            description="Instructions describing the data to extract. Append Json "
            "Schema: and a JSON schema to shape the output.",
            default=DEFAULT_PROMPT,
            advanced=False,
        )
        html: bool = SchemaField(
            description="Whether to include the page HTML in the response.",
            default=True,
        )
        browser_rendering: bool | None = browser_rendering_field()
        timeout: int = timeout_field()
        warm_start: bool = warm_start_field()
        block_resources: bool = block_resources_field()
        wait_for_selector: str | None = wait_for_selector_field()
        geo_code: str | None = geo_code_field()
        max_age: int = max_age_field()
        max_retries: int | None = max_retries_field()
        screenshot: bool = SchemaField(
            description="Whether to include a base64 screenshot in the response.",
            default=False,
        )

    class Output(BlockSchemaOutput):
        data: Any = SchemaField(
            description="Extracted data, shaped by the prompt and any JSON schema."
        )
        html: str = SchemaField(
            description="HTML of the page, when html is true and the API returns it."
        )
        screenshot: MediaFileType = SchemaField(
            description="Screenshot of the page, when screenshot is true and the "
            "API returns it."
        )
        token_usage: int = SchemaField(
            description="MrScraper tokens used by the extraction."
        )

    def __init__(self):
        super().__init__(
            id="43097c74-f421-4539-8a6e-be60b0ad4423",
            description="Extract structured JSON from a page using a "
            "natural-language prompt. Append Json Schema: and a JSON schema to "
            "shape the output.",
            categories={BlockCategory.SEARCH, BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.WORKSPACE,
            test_input={
                "credentials": mrscraper.get_test_credentials().model_dump(),
                "url": "https://example.com",
                "prompt": "Extract the first 2 items with title and price.",
                "browser_rendering": True,
            },
            test_credentials=mrscraper.get_test_credentials(),
            test_output=[
                ("data", [{"title": "Example item", "price": "10.00"}]),
                ("html", TEST_HTML),
                ("token_usage", 1),
            ],
            test_mock={
                "_store_screenshot": stored_screenshot,
                "_send": lambda *args, **kwargs: {
                    "success": True,
                    "data": [{"title": "Example item", "price": "10.00"}],
                    "thread_id": "THREAD_ID",
                    "token_usage": 1,
                    "html": TEST_HTML,
                    "markdown": "",
                    "screenshot": "",
                    "status": "Results ready",
                    "error": "",
                },
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        request = scrape_json_request(
            url=input_data.url,
            prompt=input_data.prompt,
            html=input_data.html,
            browser_rendering=input_data.browser_rendering,
            timeout=input_data.timeout,
            warm_start=input_data.warm_start,
            block_resources=input_data.block_resources,
            wait_for_selector=input_data.wait_for_selector,
            geo_code=input_data.geo_code,
            max_age=input_data.max_age,
            max_retries=input_data.max_retries,
            screenshot=input_data.screenshot,
        )
        body = json_object(await self._send(credentials, request))
        if body.get("success") is False or "data" not in body:
            raise MrScraperError(
                f"MrScraper could not extract data: {error_message(body)}"
            )
        yield "data", body["data"]
        html, screenshot = body.get("html"), body.get("screenshot")
        if input_data.html and isinstance(html, str) and html:
            yield "html", html
        if input_data.screenshot and isinstance(screenshot, str) and screenshot:
            yield "screenshot", await self._store_screenshot(
                screenshot, execution_context
            )
        if isinstance(body.get("token_usage"), int):
            yield "token_usage", body["token_usage"]
