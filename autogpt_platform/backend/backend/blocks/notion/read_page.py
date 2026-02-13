from __future__ import annotations

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import OAuth2Credentials, SchemaField

from ._api import NotionClient
from ._auth import (
    NOTION_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    NotionCredentialsField,
    NotionCredentialsInput,
)


class NotionReadPageBlock(Block):
    """Read a Notion page by ID and return its raw JSON."""

    class Input(BlockSchemaInput):
        credentials: NotionCredentialsInput = NotionCredentialsField()
        page_id: str = SchemaField(
            description="Notion page ID. Must be accessible by the connected integration. You can get this from the page URL notion.so/A-Page-586edd711467478da59fe3ce29a1ffab would be 586edd711467478da59fe35e29a1ffab",
        )

    class Output(BlockSchemaOutput):
        page: dict = SchemaField(description="Raw Notion page JSON.")

    def __init__(self):
        super().__init__(
            id="5246cc1d-34b7-452b-8fc5-3fb25fd8f542",
            description="Read a Notion page by its ID and return its raw JSON.",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=NotionReadPageBlock.Input,
            output_schema=NotionReadPageBlock.Output,
            disabled=not NOTION_OAUTH_IS_CONFIGURED,
            test_input={
                "page_id": "00000000-0000-0000-0000-000000000000",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[("page", dict)],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "get_page": lambda *args, **kwargs: {"object": "page", "id": "mocked"}
            },
        )

    @staticmethod
    async def get_page(credentials: OAuth2Credentials, page_id: str) -> dict:
        client = NotionClient(credentials)
        return await client.get_page(page_id)

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            page = await self.get_page(credentials, input_data.page_id)
            yield "page", page
        except Exception as e:
            yield "error", str(e) if str(e) else "Unknown error"
