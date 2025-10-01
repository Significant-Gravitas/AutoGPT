from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import model_validator

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import OAuth2Credentials, SchemaField

from ._api import NotionClient
from ._auth import (
    NOTION_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    NotionCredentialsField,
    NotionCredentialsInput,
)


class NotionCreatePageBlock(Block):
    """Create a new page in Notion with content."""

    class Input(BlockSchema):
        credentials: NotionCredentialsInput = NotionCredentialsField()
        parent_page_id: Optional[str] = SchemaField(
            description="Parent page ID to create the page under. Either this OR parent_database_id is required.",
            default=None,
        )
        parent_database_id: Optional[str] = SchemaField(
            description="Parent database ID to create the page in. Either this OR parent_page_id is required.",
            default=None,
        )
        title: str = SchemaField(
            description="Title of the new page",
        )
        content: Optional[str] = SchemaField(
            description="Content for the page. Can be plain text or markdown - will be converted to Notion blocks.",
            default=None,
        )
        properties: Optional[Dict[str, Any]] = SchemaField(
            description="Additional properties for database pages (e.g., {'Status': 'In Progress', 'Priority': 'High'})",
            default=None,
        )
        icon_emoji: Optional[str] = SchemaField(
            description="Emoji to use as the page icon (e.g., '📄', '🚀')", default=None
        )

        @model_validator(mode="after")
        def validate_parent(self):
            """Ensure either parent_page_id or parent_database_id is provided."""
            if not self.parent_page_id and not self.parent_database_id:
                raise ValueError(
                    "Either parent_page_id or parent_database_id must be provided"
                )
            if self.parent_page_id and self.parent_database_id:
                raise ValueError(
                    "Only one of parent_page_id or parent_database_id should be provided, not both"
                )
            return self

    class Output(BlockSchema):
        page_id: str = SchemaField(description="ID of the created page.")
        page_url: str = SchemaField(description="URL of the created page.")
        error: str = SchemaField(description="Error message if the operation failed.")

    def __init__(self):
        super().__init__(
            id="c15febe0-66ce-4c6f-aebd-5ab351653804",
            description="Create a new page in Notion. Requires EITHER a parent_page_id OR parent_database_id. Supports markdown content.",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=NotionCreatePageBlock.Input,
            output_schema=NotionCreatePageBlock.Output,
            disabled=not NOTION_OAUTH_IS_CONFIGURED,
            test_input={
                "parent_page_id": "00000000-0000-0000-0000-000000000000",
                "title": "Test Page",
                "content": "This is test content.",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("page_id", "12345678-1234-1234-1234-123456789012"),
                (
                    "page_url",
                    "https://notion.so/Test-Page-12345678123412341234123456789012",
                ),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "create_page": lambda *args, **kwargs: (
                    "12345678-1234-1234-1234-123456789012",
                    "https://notion.so/Test-Page-12345678123412341234123456789012",
                )
            },
        )

    @staticmethod
    def _markdown_to_blocks(content: str) -> List[dict]:
        """Convert markdown content to Notion block objects."""
        if not content:
            return []

        blocks = []
        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Skip empty lines
            if not line.strip():
                i += 1
                continue

            # Headings
            if line.startswith("### "):
                blocks.append(
                    {
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[4:].strip()}}
                            ]
                        },
                    }
                )
            elif line.startswith("## "):
                blocks.append(
                    {
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[3:].strip()}}
                            ]
                        },
                    }
                )
            elif line.startswith("# "):
                blocks.append(
                    {
                        "type": "heading_1",
                        "heading_1": {
                            "rich_text": [
                                {"type": "text", "text": {"content": line[2:].strip()}}
                            ]
                        },
                    }
                )
            # Bullet points
            elif line.strip().startswith("- "):
                blocks.append(
                    {
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": line.strip()[2:].strip()},
                                }
                            ]
                        },
                    }
                )
            # Numbered list
            elif line.strip() and line.strip()[0].isdigit() and ". " in line:
                content_start = line.find(". ") + 2
                blocks.append(
                    {
                        "type": "numbered_list_item",
                        "numbered_list_item": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": line[content_start:].strip()},
                                }
                            ]
                        },
                    }
                )
            # Code block
            elif line.strip().startswith("```"):
                code_lines = []
                language = line[3:].strip() or "plain text"
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                blocks.append(
                    {
                        "type": "code",
                        "code": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": "\n".join(code_lines)},
                                }
                            ],
                            "language": language,
                        },
                    }
                )
            # Quote
            elif line.strip().startswith("> "):
                blocks.append(
                    {
                        "type": "quote",
                        "quote": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": line.strip()[2:].strip()},
                                }
                            ]
                        },
                    }
                )
            # Horizontal rule
            elif line.strip() in ["---", "***", "___"]:
                blocks.append({"type": "divider", "divider": {}})
            # Regular paragraph
            else:
                # Parse for basic markdown formatting
                text_content = line.strip()
                rich_text = []

                # Simple bold/italic parsing (this is simplified)
                if "**" in text_content or "*" in text_content:
                    # For now, just pass as plain text
                    # A full implementation would parse and create proper annotations
                    rich_text = [{"type": "text", "text": {"content": text_content}}]
                else:
                    rich_text = [{"type": "text", "text": {"content": text_content}}]

                blocks.append(
                    {"type": "paragraph", "paragraph": {"rich_text": rich_text}}
                )

            i += 1

        return blocks

    @staticmethod
    def _build_properties(
        title: str, additional_properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build properties object for page creation."""
        properties: Dict[str, Any] = {
            "title": {"title": [{"type": "text", "text": {"content": title}}]}
        }

        if additional_properties:
            for key, value in additional_properties.items():
                if key.lower() == "title":
                    continue  # Skip title as we already have it

                # Try to intelligently map property types
                if isinstance(value, bool):
                    properties[key] = {"checkbox": value}
                elif isinstance(value, (int, float)):
                    properties[key] = {"number": value}
                elif isinstance(value, list):
                    # Assume multi-select
                    properties[key] = {
                        "multi_select": [{"name": str(item)} for item in value]
                    }
                elif isinstance(value, str):
                    # Could be select, rich_text, or other types
                    # For simplicity, try common patterns
                    if key.lower() in ["status", "priority", "type", "category"]:
                        properties[key] = {"select": {"name": value}}
                    elif key.lower() in ["url", "link"]:
                        properties[key] = {"url": value}
                    elif key.lower() in ["email"]:
                        properties[key] = {"email": value}
                    else:
                        properties[key] = {
                            "rich_text": [{"type": "text", "text": {"content": value}}]
                        }

        return properties

    @staticmethod
    async def create_page(
        credentials: OAuth2Credentials,
        title: str,
        parent_page_id: Optional[str] = None,
        parent_database_id: Optional[str] = None,
        content: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        icon_emoji: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Create a new Notion page.

        Returns:
            Tuple of (page_id, page_url)
        """
        if not parent_page_id and not parent_database_id:
            raise ValueError(
                "Either parent_page_id or parent_database_id must be provided"
            )
        if parent_page_id and parent_database_id:
            raise ValueError(
                "Only one of parent_page_id or parent_database_id should be provided, not both"
            )

        client = NotionClient(credentials)

        # Build parent object
        if parent_page_id:
            parent = {"type": "page_id", "page_id": parent_page_id}
        else:
            parent = {"type": "database_id", "database_id": parent_database_id}

        # Build properties
        page_properties = NotionCreatePageBlock._build_properties(title, properties)

        # Convert content to blocks if provided
        children = None
        if content:
            children = NotionCreatePageBlock._markdown_to_blocks(content)

        # Build icon if provided
        icon = None
        if icon_emoji:
            icon = {"type": "emoji", "emoji": icon_emoji}

        # Create the page
        result = await client.create_page(
            parent=parent, properties=page_properties, children=children, icon=icon
        )

        page_id = result.get("id", "")
        page_url = result.get("url", "")

        if not page_id or not page_url:
            raise ValueError("Failed to get page ID or URL from Notion response")

        return page_id, page_url

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            page_id, page_url = await self.create_page(
                credentials,
                input_data.title,
                input_data.parent_page_id,
                input_data.parent_database_id,
                input_data.content,
                input_data.properties,
                input_data.icon_emoji,
            )
            yield "page_id", page_id
            yield "page_url", page_url
        except Exception as e:
            yield "error", str(e) if str(e) else "Unknown error"
