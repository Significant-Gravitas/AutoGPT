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
    MrScraperError,
    geo_code_field,
    json_object,
    max_age_field,
    max_retries_field,
    screenshot_request,
    timeout_field,
    url_field,
    wait_for_selector_field,
)
from ._base import MrScraperBlockBase
from ._config import mrscraper, mrscraper_credentials_field
from ._testdata import TEST_HTML, TEST_PNG, stored_screenshot


class MrScraperGetScreenshotBlock(MrScraperBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = mrscraper_credentials_field()
        url: str = url_field()
        screenshot: str = SchemaField(
            description="Screenshot area to capture: full for the entire scroll "
            "height, top for the first viewport, or a WIDTHxHEIGHT viewport from "
            "100 to 3840 pixels with an optional -ZOOM percent from 25 to 300, such "
            "as 1280x800 or 1920x1080-75.",
            default="full",
            advanced=False,
        )
        timeout: int = timeout_field()
        wait_for_selector: str | None = wait_for_selector_field()
        geo_code: str | None = geo_code_field()
        max_age: int = max_age_field()
        max_retries: int | None = max_retries_field()

    class Output(BlockSchemaOutput):
        screenshot: MediaFileType = SchemaField(
            description="PNG screenshot of the page."
        )
        html: str = SchemaField(
            description="HTML of the page, which the API includes with the "
            "screenshot."
        )

    def __init__(self):
        super().__init__(
            id="489ba375-1bb4-4478-b1c5-0f58aadaf6bd",
            description="Capture a base64 PNG screenshot of the full page, the "
            "first viewport, or a custom viewport. Browser rendering is always "
            "enabled.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.WORKSPACE,
            test_input={
                "credentials": mrscraper.get_test_credentials().model_dump(),
                "url": "https://example.com",
                "screenshot": "top",
                "timeout": 120,
            },
            test_credentials=mrscraper.get_test_credentials(),
            test_output=[
                ("screenshot", lambda s: str(s).startswith("data:image/png;base64,")),
                ("html", TEST_HTML),
            ],
            test_mock={
                "_store_screenshot": stored_screenshot,
                "_send": lambda *args, **kwargs: {
                    "html": TEST_HTML,
                    "screenshot": TEST_PNG,
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
        request = screenshot_request(
            url=input_data.url,
            screenshot=input_data.screenshot,
            timeout=input_data.timeout,
            wait_for_selector=input_data.wait_for_selector,
            geo_code=input_data.geo_code,
            max_age=input_data.max_age,
            max_retries=input_data.max_retries,
        )
        body = json_object(await self._send(credentials, request))
        screenshot = body.get("screenshot")
        if not isinstance(screenshot, str) or not screenshot:
            raise MrScraperError("MrScraper returned no screenshot for this page.")
        yield "screenshot", await self._store_screenshot(screenshot, execution_context)
        if isinstance(body.get("html"), str):
            yield "html", body["html"]
