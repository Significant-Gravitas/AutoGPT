from typing import Any

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
    MrScraperError,
    error_message,
    json_object,
    scrape_brand_request,
    url_field,
)
from ._base import MrScraperBlockBase
from ._config import mrscraper, mrscraper_credentials_field


class MrScraperScrapeBrandBlock(MrScraperBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = mrscraper_credentials_field()
        url: str = url_field()

    class Output(BlockSchemaOutput):
        brand: dict[str, Any] = SchemaField(
            description="Brand details: name, colors, typography, logos, and "
            "social links."
        )
        final_url: str = SchemaField(description="URL of the page after redirects.")
        response: dict[str, Any] = SchemaField(
            description="Raw brand scraping API response, with message and data."
        )

    def __init__(self):
        super().__init__(
            id="8788c018-943f-49af-b12b-9c05d208361e",
            description="Extract a website's colors, typography, logos, and social "
            "links as JSON.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": mrscraper.get_test_credentials().model_dump(),
                "url": "https://example.com",
            },
            test_credentials=mrscraper.get_test_credentials(),
            test_output=[
                ("brand", lambda b: b["colors"]["background"] == "#ffffff"),
                ("final_url", "https://example.com"),
                ("response", lambda r: r["data"]["status"] == "succeeded"),
            ],
            test_mock={
                "_send": lambda *args, **kwargs: {
                    "data": {
                        "job_id": "JOB_ID",
                        "status": "succeeded",
                        "url": "https://example.com",
                        "final_url": "https://example.com",
                        "brand": {
                            "name": "Example brand",
                            "colors": {"background": "#ffffff", "text": "#222222"},
                        },
                        "error": None,
                        "bundle_url": None,
                    },
                    "message": "Example response",
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        request = scrape_brand_request(url=input_data.url)
        body = json_object(await self._send(credentials, request))
        job = body.get("data")
        if not isinstance(job, dict):
            raise MrScraperError(
                f"MrScraper returned no brand data: {error_message(body)}"
            )
        if job.get("error") or job.get("status", "succeeded") != "succeeded":
            raise MrScraperError(
                "MrScraper could not extract the brand: "
                f"{job.get('error') or job.get('status')}"
            )
        brand = job.get("brand")
        if not isinstance(brand, dict):
            raise MrScraperError("MrScraper returned no brand data for this page.")
        yield "brand", brand
        yield "final_url", job.get("final_url") or job.get("url") or input_data.url
        yield "response", body
