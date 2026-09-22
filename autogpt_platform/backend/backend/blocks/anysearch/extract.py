from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._api import AnySearchClient, unwrap_envelope
from ._config import anysearch


class AnySearchExtractBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = anysearch.credentials_field(
            description="The AnySearch integration requires an API Key."
        )
        url: str = SchemaField(description="The URL to extract content from")

    class Output(BlockSchemaOutput):
        url: str = SchemaField(description="URL of the extracted page")
        title: str = SchemaField(description="Title of the extracted page")
        content: str = SchemaField(
            description="Extracted page content, formatted as markdown"
        )
        error: str = SchemaField(
            description="Error message if the extraction failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="dcaff561-4c3e-4669-b841-4392b98cb024",
            description="Extracts the content of a single URL as markdown using "
            "AnySearch, optimized for LLM consumption",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": anysearch.get_test_credentials().model_dump(),
                "url": "https://agpt.co",
            },
            test_credentials=anysearch.get_test_credentials(),
            test_output=[
                ("url", "https://agpt.co"),
                ("title", "AutoGPT"),
                ("content", lambda x: "AI agents" in x),
            ],
            test_mock={
                "_extract": lambda *args, **kwargs: {
                    "code": 0,
                    "message": "success",
                    "data": {
                        "url": "https://agpt.co",
                        "title": "AutoGPT",
                        "content": "AutoGPT is a platform for building AI agents.",
                    },
                }
            },
        )

    async def _extract(
        self, credentials: APIKeyCredentials, url: str
    ) -> dict[str, Any]:
        """POST /v1/extract and return the raw response envelope (mockable)."""
        return await AnySearchClient(credentials).extract(url)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            data = unwrap_envelope(await self._extract(credentials, input_data.url))
        except Exception as e:
            raise BlockExecutionError(
                message=f"Extract failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        yield "url", data.get("url") or input_data.url
        yield "title", data.get("title") or ""
        yield "content", data.get("content") or ""
