from backend.blocks.notion._auth import TEST_CREDENTIALS
from backend.blocks.notion.append_to_page import NotionAppendToPageBlock

PAGE_ID = "12345678-1234-1234-1234-123456789012"
CHILDREN_URL = f"https://api.notion.com/v1/blocks/{PAGE_ID}/children"


def _paragraph(content: str) -> dict:
    return {
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": content}}]},
    }


async def test_markdown_is_appended_as_children_of_the_page(notion_api):
    sent = notion_api({"results": [{"id": "h"}, {"id": "p"}]})

    block_ids = await NotionAppendToPageBlock.append_content(
        TEST_CREDENTIALS, PAGE_ID, "## Update\n\nShipped today."
    )

    assert block_ids == ["h", "p"]
    heading = {
        "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Update"}}]},
    }
    assert sent == [
        ("PATCH", CHILDREN_URL, {"children": [heading, _paragraph("Shipped today.")]})
    ]


async def test_more_than_a_hundred_blocks_go_in_order_over_several_requests(
    notion_api,
):
    lines = [f"line {n}" for n in range(250)]
    sent = notion_api(
        *(
            {"results": [{"id": line} for line in chunk]}
            for chunk in (lines[:100], lines[100:200], lines[200:])
        )
    )

    block_ids = await NotionAppendToPageBlock.append_content(
        TEST_CREDENTIALS, PAGE_ID, "\n".join(lines)
    )

    assert block_ids == lines
    assert [(method, url) for method, url, _ in sent] == [("PATCH", CHILDREN_URL)] * 3
    assert [body["children"] for _, _, body in sent] == [
        [_paragraph(line) for line in chunk]
        for chunk in (lines[:100], lines[100:200], lines[200:])
    ]
