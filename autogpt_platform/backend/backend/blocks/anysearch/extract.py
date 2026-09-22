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
            description="Extracted readable page content (cleaned HTML, text, JSON, or markdown; untrusted web content)"
        )

    def __init__(self):
        super().__init__(
            id="dcaff561-4c3e-4669-b841-4392b98cb024",
            description="Extracts readable content from a single URL using "
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
            url = data.get("url") or input_data.url
            title = data.get("title") or ""
            content = data.get("content") or ""
            if not all(isinstance(v, str) for v in (url, title, content)):
                raise ValueError("malformed extract response fields")
        except Exception as e:
            raise BlockExecutionError(
                message=f"Extract failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        yield "url", url
        yield "title", title
        yield "content", content
