import pytest

from backend.blocks.notion._auth import TEST_CREDENTIALS
from backend.blocks.notion.update_page import NotionUpdatePageBlock

PAGE_ID = "12345678-1234-1234-1234-123456789012"
PAGE_URL = f"https://api.notion.com/v1/pages/{PAGE_ID}"
UPDATED = {"id": PAGE_ID, "url": "https://notion.so/Renamed-12345678"}


def _text(content: str) -> list[dict]:
    return [{"type": "text", "text": {"content": content}}]


async def test_properties_are_written_as_the_types_the_page_already_has(notion_api):
    schema = {
        "properties": {
            "Name": {"id": "title", "type": "title"},
            "Status": {"id": "a", "type": "status"},
            "Stage": {"id": "b", "type": "select"},
            "Tags": {"id": "c", "type": "multi_select"},
            "Notes": {"id": "d", "type": "rich_text"},
            "Due": {"id": "e", "type": "date"},
            "Done": {"id": "f", "type": "checkbox"},
            "Owner": {"id": "g", "type": "people"},
        }
    }
    sent = notion_api(schema, UPDATED)

    result = await NotionUpdatePageBlock.update_page(
        TEST_CREDENTIALS,
        PAGE_ID,
        title="Renamed",
        properties={
            "Status": "Done",
            "Stage": "Review",
            "Tags": ["urgent", "q4"],
            "Notes": "Shipped",
            "Due": "2026-10-31",
            "Done": True,
            "Owner": None,
        },
        icon_emoji="🚀",
        cover_url="https://example.com/cover.png",
    )

    assert result == (PAGE_ID, UPDATED["url"])
    assert sent == [
        ("GET", PAGE_URL, None),
        (
            "PATCH",
            PAGE_URL,
            {
                "properties": {
                    "Status": {"status": {"name": "Done"}},
                    "Stage": {"select": {"name": "Review"}},
                    "Tags": {"multi_select": [{"name": "urgent"}, {"name": "q4"}]},
                    "Notes": {"rich_text": _text("Shipped")},
                    "Due": {"date": {"start": "2026-10-31"}},
                    "Done": {"checkbox": True},
                    "Owner": {"people": []},
                    "title": {"title": _text("Renamed")},
                },
                "icon": {"type": "emoji", "emoji": "🚀"},
                "cover": {
                    "type": "external",
                    "external": {"url": "https://example.com/cover.png"},
                },
            },
        ),
    ]


@pytest.mark.parametrize("in_trash", [True, False, None])
async def test_trash_state_is_sent_only_when_asked(notion_api, in_trash: bool | None):
    sent = notion_api(UPDATED)

    await NotionUpdatePageBlock.update_page(
        TEST_CREDENTIALS, PAGE_ID, title="Renamed", in_trash=in_trash
    )

    expected: dict = {"properties": {"title": {"title": _text("Renamed")}}}
    if in_trash is not None:
        expected["archived"] = in_trash
    assert sent == [("PATCH", PAGE_URL, expected)]


async def test_an_unknown_property_is_refused_before_anything_is_written(notion_api):
    sent = notion_api({"properties": {"Name": {"id": "title", "type": "title"}}})

    with pytest.raises(ValueError, match="no property named 'Stauts'"):
        await NotionUpdatePageBlock.update_page(
            TEST_CREDENTIALS, PAGE_ID, properties={"Stauts": "Done"}
        )

    assert [method for method, _, _ in sent] == ["GET"]
