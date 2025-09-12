from __future__ import annotations

from typing import Literal

from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    OAuth2Credentials,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.request import Requests

NOTION_VERSION = "2022-06-28"


class NotionReadPageBlock(Block):
    """Read a Notion page by ID and return its raw JSON."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.NOTION], Literal["oauth2"]
        ] = CredentialsField(
            description="Connect your Notion account. Ensure the page is shared with the integration."
        )
        page_id: str = SchemaField(
            description="Notion page ID. Must be accessible by the connected integration."
        )

    class Output(BlockSchema):
        page: dict = SchemaField(description="Raw Notion page JSON.")
        error: str = SchemaField(description="Error message if the operation failed.")

    def __init__(self):
        super().__init__(
            id="5246cc1d-34b7-452b-8fc5-3fb25fd8f542",  # generated from https://www.uuidgenerator.net/
            description="Read a Notion page by its ID and return its raw JSON.",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=NotionReadPageBlock.Input,
            output_schema=NotionReadPageBlock.Output,
            test_input={
                "page_id": "00000000-0000-0000-0000-000000000000",
                "credentials": {
                    "id": "test-cred-id",
                    "provider": ProviderName.NOTION,
                    "type": "oauth2",
                },
            },
            test_output=[("page", dict)],
            test_credentials={
                "credentials": OAuth2Credentials(
                    id="test-cred-id",
                    provider=ProviderName.NOTION,
                    access_token=SecretStr("test-token"),
                    scopes=[],
                )
            },
            test_mock={
                "_http_get": lambda *args, **kwargs: {"object": "page", "id": "mocked"}
            },
        )

    async def _http_get(self, url: str, headers: dict) -> dict:
        resp = await Requests().get(url, headers=headers)
        return resp.json()

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
        **kwargs,
    ) -> BlockOutput:
        headers = {
            "Authorization": credentials.auth_header(),
            "Notion-Version": NOTION_VERSION,
        }
        url = f"https://api.notion.com/v1/pages/{input_data.page_id}"

        try:
            page = await self._http_get(url, headers=headers)
            yield "page", page
        except Exception as e:
            yield "error", str(e) if str(e) else "Unknown error"
