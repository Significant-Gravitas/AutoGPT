from __future__ import annotations

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
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
from .create_page import NotionCreatePageBlock

# Notion refuses an append of more children than this in one request.
MAX_CHILDREN_PER_REQUEST = 100
# Notion refuses a rich-text segment longer than this, counted in UTF-16 units.
MAX_TEXT_LENGTH = 2000


class NotionAppendToPageBlock(Block):
    """Add content to the end of an existing Notion page."""

    class Input(BlockSchemaInput):
        credentials: NotionCredentialsInput = NotionCredentialsField()
        page_id: str = SchemaField(
            description="ID of the page to add content to. Must be accessible by the connected integration.",
        )
        content: str = SchemaField(
            description="Content to add at the end of the page. Can be plain text or markdown - will be converted to Notion blocks.",
        )

    class Output(BlockSchemaOutput):
        block_ids: list[str] = SchemaField(
            description="IDs of the blocks added to the page, in page order."
        )

    def __init__(self):
        super().__init__(
            id="f33a076f-0eb1-46b2-97a3-04ab23ab06f4",
            description="Add content to the end of an existing Notion page. Supports markdown content.",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=NotionAppendToPageBlock.Input,
            output_schema=NotionAppendToPageBlock.Output,
            disabled=not NOTION_OAUTH_IS_CONFIGURED,
            test_input={
                "page_id": "12345678-1234-1234-1234-123456789012",
                "content": "## Update\n\nThis is appended content.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                (
                    "block_ids",
                    [
                        "aaaaaaaa-1234-1234-1234-123456789012",
                        "bbbbbbbb-1234-1234-1234-123456789012",
                    ],
                ),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "append_content": lambda *args, **kwargs: [
                    "aaaaaaaa-1234-1234-1234-123456789012",
                    "bbbbbbbb-1234-1234-1234-123456789012",
                ]
            },
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    @staticmethod
    async def append_content(
        credentials: OAuth2Credentials, page_id: str, content: str
    ) -> list[str]:
        """
        Append markdown content to a Notion page.

        Returns:
            IDs of the appended blocks, in page order.
        """
        client = NotionClient(credentials)
        children = [
            _within_text_limit(block)
            for block in NotionCreatePageBlock._markdown_to_blocks(content)
        ]

        block_ids: list[str] = []
        for start in range(0, len(children), MAX_CHILDREN_PER_REQUEST):
            try:
                result = await client.append_blocks(
                    page_id, children[start : start + MAX_CHILDREN_PER_REQUEST]
                )
            except Exception as e:
                if not block_ids:
                    raise
                raise PartialAppendError(block_ids, len(children), e) from e
            block_ids.extend(block["id"] for block in result.get("results", []))
        return block_ids

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            block_ids = await self.append_content(
                credentials, input_data.page_id, input_data.content
            )
            yield "block_ids", block_ids
        except Exception as e:
            yield "error", str(e) if str(e) else "Unknown error"


class PartialAppendError(Exception):
    """Some blocks are on the page; the error names them instead of an output, which
    would run whatever is wired to `block_ids` on a partial append."""

    def __init__(self, block_ids: list[str], total: int, cause: Exception):
        super().__init__(
            f"Added {len(block_ids)} of {total} blocks, the last with ID "
            f"{block_ids[-1]}, then a request failed; check the page for the rest "
            f"before retrying: {cause}"
        )


def _within_text_limit(block: dict) -> dict:
    """Split each rich-text segment over MAX_TEXT_LENGTH into several, in order."""
    body = block[block["type"]]
    if "rich_text" in body:
        body["rich_text"] = [
            {**segment, "text": {**segment["text"], "content": piece}}
            for segment in body["rich_text"]
            for piece in _split_text(segment["text"]["content"])
        ]
    return block


def _split_text(text: str) -> list[str]:
    pieces: list[str] = []
    piece: list[str] = []
    size = 0
    for char in text:
        width = 2 if ord(char) > 0xFFFF else 1
        if size + width > MAX_TEXT_LENGTH:
            pieces.append("".join(piece))
            piece, size = [], 0
        piece.append(char)
        size += width
    return [*pieces, "".join(piece)]
