from __future__ import annotations

from typing import Any

from pydantic import model_validator

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


class NotionUpdatePagePropertiesBlock(Block):
    """Set an existing Notion page's properties, icon, cover or trash state."""

    class Input(BlockSchemaInput):
        credentials: NotionCredentialsInput = NotionCredentialsField()
        page_id: str = SchemaField(
            description="ID of the page to update. Must be accessible by the connected integration.",
        )
        title: str | None = SchemaField(
            description="New title for the page", default=None
        )
        properties: dict[str, Any] | None = SchemaField(
            description="Properties to set by name (e.g., {'Status': 'Done', 'Tags': ['urgent']}). Each value is written as the type the property already has; null clears it, or resets a status to its default option.",
            default=None,
        )
        icon_emoji: str | None = SchemaField(
            description="Emoji to use as the page icon (e.g., '📄', '🚀')", default=None
        )
        cover_url: str | None = SchemaField(
            description="URL of an image to use as the page cover", default=None
        )
        in_trash: bool | None = SchemaField(
            description="True moves the page to the trash, False restores it. Leave empty to keep it where it is.",
            default=None,
        )

        @model_validator(mode="after")
        def validate_change(self):
            """Ensure the update changes something."""
            if not (
                self.title
                or self.properties
                or self.icon_emoji
                or self.cover_url
                or self.in_trash is not None
            ):
                raise ValueError(
                    "Provide at least one of title, properties, icon_emoji, cover_url or in_trash"
                )
            return self

    class Output(BlockSchemaOutput):
        page_id: str = SchemaField(description="ID of the updated page.")
        page_url: str = SchemaField(description="URL of the updated page.")

    def __init__(self):
        super().__init__(
            id="c80f7468-58e5-4638-a82e-bddf3bf2d383",
            description="Update an existing Notion page's properties, its title among them, its icon or cover, or whether it is in the trash. The page's content is left as it is.",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=NotionUpdatePagePropertiesBlock.Input,
            output_schema=NotionUpdatePagePropertiesBlock.Output,
            disabled=not NOTION_OAUTH_IS_CONFIGURED,
            test_input={
                "page_id": "12345678-1234-1234-1234-123456789012",
                "title": "Renamed Page",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_output=[
                ("page_id", "12345678-1234-1234-1234-123456789012"),
                (
                    "page_url",
                    "https://notion.so/Renamed-Page-12345678123412341234123456789012",
                ),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "update_page": lambda *args, **kwargs: (
                    "12345678-1234-1234-1234-123456789012",
                    "https://notion.so/Renamed-Page-12345678123412341234123456789012",
                )
            },
            effect=BlockEffect.EXTERNAL_JUDGED,
        )

    @staticmethod
    async def update_page(
        credentials: OAuth2Credentials,
        page_id: str,
        title: str | None = None,
        properties: dict[str, Any] | None = None,
        icon_emoji: str | None = None,
        cover_url: str | None = None,
        in_trash: bool | None = None,
    ) -> tuple[str, str]:
        """
        Update a Notion page.

        Returns:
            Tuple of (page_id, page_url)
        """
        client = NotionClient(credentials)

        page_properties: dict[str, Any] = {}
        if properties:
            page = await client.get_page(page_id)
            page_properties = _property_values(page.get("properties", {}), properties)
        if title:
            page_properties["title"] = {
                "title": [{"type": "text", "text": {"content": title}}]
            }

        result = await client.update_page(
            page_id,
            properties=page_properties,
            icon={"type": "emoji", "emoji": icon_emoji} if icon_emoji else None,
            cover=(
                {"type": "external", "external": {"url": cover_url}}
                if cover_url
                else None
            ),
            # `archived` is the trash flag's name in the pinned NOTION_VERSION.
            archived=in_trash,
        )
        return result["id"], result["url"]

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            page_id, page_url = await self.update_page(
                credentials,
                input_data.page_id,
                input_data.title,
                input_data.properties,
                input_data.icon_emoji,
                input_data.cover_url,
                input_data.in_trash,
            )
            yield "page_id", page_id
            yield "page_url", page_url
        except Exception as e:
            yield "error", str(e) if str(e) else "Unknown error"


# What clears each type this block writes; read-only types such as formula are absent.
_CLEARED_VALUES: dict[str, Any] = {
    "checkbox": False,
    "select": None,
    "status": None,
    "date": None,
    "number": None,
    "url": None,
    "email": None,
    "phone_number": None,
    "title": [],
    "rich_text": [],
    "multi_select": [],
    "people": [],
    "relation": [],
    "files": [],
}
_SCALAR_TYPES = {"number", "checkbox", "url", "email", "phone_number"}


def _property_values(
    page_properties: dict[str, Any], values: dict[str, Any]
) -> dict[str, Any]:
    """Shape each value as the type the page's property of that name already has."""
    shaped: dict[str, Any] = {}
    for name, value in values.items():
        if name not in page_properties:
            raise ValueError(f"The page has no property named '{name}'")
        shaped[name] = _property_value(name, page_properties[name]["type"], value)
    return shaped


def _property_value(name: str, prop_type: str, value: Any) -> dict:
    if prop_type not in _CLEARED_VALUES:
        raise ValueError(f"Writing a {prop_type} property ('{name}') is not supported")
    if value is None:
        return {prop_type: _CLEARED_VALUES[prop_type]}
    if prop_type in ("title", "rich_text"):
        return {prop_type: [{"type": "text", "text": {"content": str(value)}}]}
    if prop_type in ("select", "status"):
        return {prop_type: {"name": str(value)}}
    if prop_type == "multi_select":
        names = value if isinstance(value, list) else [value]
        return {"multi_select": [{"name": str(option)} for option in names]}
    if prop_type == "date":
        return {"date": {"start": str(value)}}
    if prop_type in _SCALAR_TYPES:
        return {prop_type: value}
    raise ValueError(f"Writing a {prop_type} property ('{name}') is not supported")
