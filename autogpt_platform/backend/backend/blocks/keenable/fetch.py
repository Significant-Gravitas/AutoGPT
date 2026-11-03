from typing import Any

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

from ._api import KEENABLE_TRUNCATION_MARKER, KeenableAuth, KeenableClient
from ._config import keenable


class KeenableFetchBlock(Block):
    class Input(BlockSchemaInput):
        auth: KeenableAuth = SchemaField(
            title="Authentication",
            description="Anonymous (keyless, rate-limited per IP) or a Keenable "
            "API key credential",
            default=KeenableAuth.ANONYMOUS,
            advanced=False,
        )
        credentials: CredentialsMetaInput = keenable.credentials_field(
            title="Keenable API key",
            description="Optional: a Keenable API key lifts the anonymous rate limit.",
            discriminator="auth",
            discriminator_mapping={KeenableAuth.API_KEY.value: "keenable"},
            credential_free_discriminator_values={KeenableAuth.ANONYMOUS.value},
            default=None,
        )
        url: str = SchemaField(description="The URL to fetch")
        max_chars: int | None = SchemaField(
            description="Cut the returned markdown at this many characters",
            default=None,
            ge=1,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        url: str = SchemaField(description="URL of the fetched page")
        title: str = SchemaField(description="Title of the fetched page")
        content: str = SchemaField(
            description="Page content as clean markdown (untrusted web content)"
        )
        truncated: bool = SchemaField(
            description="Whether the content was cut to fit max_chars"
        )

    def __init__(self):
        super().__init__(
            id="08dfdebc-f4ac-4877-b281-7fa033244e71",
            description="Fetches a web page as clean markdown using Keenable. "
            "Works without an API key.",
            categories={BlockCategory.SEARCH},
            effect=BlockEffect.READ,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": keenable.get_test_credentials().model_dump(),
                "url": "https://agpt.co",
            },
            test_credentials=keenable.get_test_credentials(),
            test_output=[
                ("url", "https://agpt.co"),
                ("title", "AutoGPT"),
                ("content", lambda x: "AI agents" in x),
                ("truncated", False),
            ],
            test_mock={
                "_fetch": lambda *args, **kwargs: {
                    "url": "https://agpt.co",
                    "title": "AutoGPT",
                    "content": "# AutoGPT\n\nAutoGPT is a platform for building AI agents.",
                }
            },
        )

    async def _fetch(
        self, credentials: APIKeyCredentials | None, url: str, max_chars: int | None
    ) -> dict[str, Any]:
        """GET /v1/fetch and return the raw response (mockable)."""
        return await KeenableClient(credentials).fetch(url, max_chars)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials | None = None,
        **kwargs,
    ) -> BlockOutput:
        if input_data.auth == KeenableAuth.ANONYMOUS:
            credentials = None
        elif credentials is None:
            raise ValueError(
                "Keenable API key credentials are required when auth is api_key."
            )

        try:
            data = await self._fetch(credentials, input_data.url, input_data.max_chars)
            url = data.get("url") or input_data.url
            title = data.get("title") or ""
            content = data.get("content") or ""
            if not all(isinstance(v, str) for v in (url, title, content)):
                raise ValueError("malformed fetch response fields")
        except Exception as e:
            raise BlockExecutionError(
                message=f"Fetch failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        yield "url", url
        yield "title", title
        yield "content", content
        yield "truncated", KEENABLE_TRUNCATION_MARKER in content
