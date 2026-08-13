from __future__ import annotations

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import OAuth2Credentials, SchemaField

from ._api import NotionClient, blocks_to_markdown, extract_page_title
from ._auth import (
    NOTION_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    NotionCredentialsField,
    NotionCredentialsInput,
)


class NotionReadPageMarkdownBlock(Block):
    """Read a Notion page and convert it to clean Markdown format."""

    class Input(BlockSchemaInput):
        credentials: NotionCredentialsInput = NotionCredentialsField()
        page_id: str = SchemaField(
            description="Notion page ID. Must be accessible by the connected integration. You can get this from the page URL notion.so/A-Page-586edd711467478da59fe35e29a1ffab would be 586edd711467478da59fe35e29a1ffab",
        )
        include_title: bool = SchemaField(
            description="Whether to include the page title as a header in the markdown",
            default=True,
        )

    class Output(BlockSchemaOutput):
        markdown: str = SchemaField(description="Page content in Markdown format.")
        title: str = SchemaField(description="Page title.")

    def __init__(self):
        super().__init__(
            id="d1312c4d-fae2-4e70-893d-f4d07cce1d4e",
            description="Read a Notion page and convert it to Markdown format with proper formatting for headings, lists, links, and rich text.",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=NotionReadPageMarkdownBlock.Input,
            output_schema=NotionReadPageMarkdownBlock.Output,
            disabled=not NOTION_OAUTH_IS_CONFIGURED,
            test_input={
                "page_id": "00000000-0000-0000-0000-000000000000",
                "include_title": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("markdown", "# Test Page\n\nThis is test content."),
                ("title", "Test Page"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "get_page_markdown": lambda *args, **kwargs: (
                    "# Test Page\n\nThis is test content.",
                    "Test Page",
                )
            },
        )

    @staticmethod
    async def get_page_markdown(
        credentials: OAuth2Credentials, page_id: str, include_title: bool = True
    ) -> tuple[str, str]:
        """
        Get a Notion page and convert it to markdown.

        Args:
            credentials: OAuth2 credentials for Notion.
            page_id: The ID of the page to fetch.
            include_title: Whether to include the page title in the markdown.

        Returns:
            Tuple of (markdown_content, title)
        """
        client = NotionClient(credentials)

        # Get page metadata
        page = await client.get_page(page_id)
        title = extract_page_title(page)

        # Get all blocks from the page
        blocks = await client.get_blocks(page_id, recursive=True)

        # Convert blocks to markdown
        content_markdown = blocks_to_markdown(blocks)

        # Combine title and content if requested
        if include_title and title:
            full_markdown = f"# {title}\n\n{content_markdown}"
        else:
            full_markdown = content_markdown

        return full_markdown, title

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            markdown, title = await self.get_page_markdown(
                credentials, input_data.page_id, input_data.include_title
            )
            yield "markdown", markdown
            yield "title", title
        except Exception as e:
            yield "error", str(e) if str(e) else "Unknown error"
