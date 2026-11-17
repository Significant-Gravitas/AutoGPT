import pytest
from pydantic import ValidationError

from backend.blocks.notion._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.notion.update_page_properties import NotionUpdatePagePropertiesBlock

PAGE_ID = "12345678-1234-1234-1234-123456789012"
PAGE_URL = f"https://api.notion.com/v1/pages/{PAGE_ID}"
UPDATED = {"id": PAGE_ID, "url": "https://notion.so/Renamed-12345678"}
NOT_FOUND = {"object": "error", "status": 404, "code": "object_not_found"}


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
        }
    }
    sent = notion_api(schema, UPDATED)

    result = await NotionUpdatePagePropertiesBlock.update_page(
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

    await NotionUpdatePagePropertiesBlock.update_page(
        TEST_CREDENTIALS, PAGE_ID, title="Renamed", in_trash=in_trash
    )

    expected: dict = {"properties": {"title": {"title": _text("Renamed")}}}
    if in_trash is not None:
        expected["archived"] = in_trash
    assert sent == [("PATCH", PAGE_URL, expected)]


async def test_a_trash_only_update_sends_only_the_trash_flag(notion_api):
    sent = notion_api(UPDATED)

    await NotionUpdatePagePropertiesBlock.update_page(
        TEST_CREDENTIALS, PAGE_ID, in_trash=True
    )

    assert sent == [("PATCH", PAGE_URL, {"archived": True})]


@pytest.mark.parametrize(
    "change, valid",
    [
        ({}, False),
        ({"in_trash": False}, True),
        ({"cover_url": "https://x.io/c.png"}, True),
    ],
)
def test_an_update_must_change_something(change: dict, valid: bool):
    def build():
        return NotionUpdatePagePropertiesBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT,
            page_id=PAGE_ID,
            **change,
        )

    if valid:
        build()
    else:
        with pytest.raises(ValidationError, match="Provide at least one of"):
            build()


async def test_a_notion_error_is_the_blocks_only_output(notion_api):
    notion_api(NOT_FOUND, statuses=[404])
    block = NotionUpdatePagePropertiesBlock()
    input_data = block.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        page_id=PAGE_ID,
        title="Renamed",
    )

    outputs = [out async for out in block.run(input_data, credentials=TEST_CREDENTIALS)]

    assert [name for name, _ in outputs] == ["error"]
    assert "HTTP 404 Error: Not Found" in outputs[0][1]
    assert "object_not_found" in outputs[0][1]


async def test_null_clears_each_property_the_way_its_type_allows(notion_api):
    schema = {
        "properties": {
            "Done": {"id": "a", "type": "checkbox"},
            "Stage": {"id": "b", "type": "select"},
            "Owner": {"id": "c", "type": "people"},
        }
    }
    sent = notion_api(schema, UPDATED)

    await NotionUpdatePagePropertiesBlock.update_page(
        TEST_CREDENTIALS,
        PAGE_ID,
        properties={"Done": None, "Stage": None, "Owner": None},
    )

    assert sent[-1][2] == {
        "properties": {
            "Done": {"checkbox": False},
            "Stage": {"select": None},
            "Owner": {"people": []},
        }
    }


@pytest.mark.parametrize(
    "properties, error",
    [
        ({"Stauts": "Done"}, "no property named 'Stauts'"),
        ({"Total": None}, r"Writing a formula property \('Total'\) is not supported"),
        ({"Owner": ["Ada"]}, r"Writing a people property \('Owner'\) is not supported"),
    ],
)
async def test_a_property_it_cannot_write_is_refused_before_anything_is_written(
    notion_api, properties: dict, error: str
):
    schema = {
        "properties": {
            "Name": {"id": "title", "type": "title"},
            "Total": {"id": "a", "type": "formula"},
            "Owner": {"id": "b", "type": "people"},
        }
    }
    sent = notion_api(schema)

    with pytest.raises(ValueError, match=error):
        await NotionUpdatePagePropertiesBlock.update_page(
            TEST_CREDENTIALS, PAGE_ID, properties=properties
        )

    assert [method for method, _, _ in sent] == ["GET"]
