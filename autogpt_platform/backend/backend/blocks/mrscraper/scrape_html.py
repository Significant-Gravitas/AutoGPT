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
    action_field,
    block_resources_field,
    browser_rendering_field,
    geo_code_field,
    max_age_field,
    max_retries_field,
    scrape_html_request,
    text_field,
    timeout_field,
    url_field,
    wait_for_selector_field,
    warm_start_field,
)
from ._base import MrScraperBlockBase
from ._config import mrscraper, mrscraper_credentials_field
from ._testdata import TEST_HTML, TEST_PNG, stored_screenshot


class MrScraperScrapeHTMLBlock(MrScraperBlockBase):
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
        screenshot: str = SchemaField(
            description="Screenshot to capture with the HTML: false for none, full "
            "for the entire page, top for the first viewport, or a WIDTHxHEIGHT "
            "viewport from 100 to 3840 pixels with an optional -ZOOM percent from "
            "25 to 300, such as 1280x800 or 1920x1080-75. Setting a screenshot also "
            "sends browserRendering=true.",
            default="false",
        )

    class Output(BlockSchemaOutput):
        html: str = SchemaField(description="Raw HTML of the page.")
        screenshot: MediaFileType = SchemaField(
            description="PNG screenshot of the page, when a screenshot is set."
        )

    def __init__(self):
        super().__init__(
            id="f8a4ed62-bffe-4cc0-b0f1-24d93508c6a8",
            description="Retrieve the raw HTML of a page. Browser rendering is "
            "optional.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.WORKSPACE,
            test_input=[
                {
                    "credentials": mrscraper.get_test_credentials().model_dump(),
                    "url": "https://example.com",
                },
                {
                    "credentials": mrscraper.get_test_credentials().model_dump(),
                    "url": "https://example.com",
                    "screenshot": "top",
                },
            ],
            test_credentials=mrscraper.get_test_credentials(),
            test_output=[
                ("html", TEST_HTML),
                ("html", TEST_HTML),
                ("screenshot", lambda s: str(s).startswith("data:image/png;base64,")),
            ],
            test_mock={
                "_store_screenshot": stored_screenshot,
                "_send": lambda credentials, request: (
                    {
                        "html": TEST_HTML,
                        "screenshot": TEST_PNG,
                    }
                    if dict(request.query)["screenshot"] != "false"
                    else TEST_HTML
                ),
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
        request = scrape_html_request(
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
            screenshot=input_data.screenshot,
        )
        body = await self._send(credentials, request)
        yield "html", text_field(body, "html")
        if isinstance(body, dict) and body.get("screenshot"):
            yield "screenshot", await self._store_screenshot(
                body["screenshot"], execution_context
            )
