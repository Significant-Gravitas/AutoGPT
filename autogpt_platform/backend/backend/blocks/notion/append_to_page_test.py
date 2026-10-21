from backend.blocks.notion._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
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


async def test_a_notion_error_is_the_blocks_only_output(notion_api):
    notion_api({"object": "error", "code": "object_not_found"}, statuses=[404])
    block = NotionAppendToPageBlock()
    input_data = block.Input(
        credentials=TEST_CREDENTIALS_INPUT, page_id=PAGE_ID, content="Shipped today."
    )

    outputs = [out async for out in block.run(input_data, credentials=TEST_CREDENTIALS)]

    assert [name for name, _ in outputs] == ["error"]
    assert "HTTP 404 Error: Not Found" in outputs[0][1]
    assert "object_not_found" in outputs[0][1]


async def test_a_failure_after_the_first_request_still_reports_the_blocks_added(
    notion_api,
):
    lines = [f"line {n}" for n in range(150)]
    notion_api(
        {"results": [{"id": line} for line in lines[:100]]},
        {"object": "error", "code": "rate_limited"},
        statuses=[200, 429],
    )
    block = NotionAppendToPageBlock()
    input_data = block.Input(
        credentials=TEST_CREDENTIALS_INPUT, page_id=PAGE_ID, content="\n".join(lines)
    )

    outputs = [out async for out in block.run(input_data, credentials=TEST_CREDENTIALS)]

    assert [name for name, _ in outputs] == ["block_ids", "error"]
    assert outputs[0][1] == lines[:100]
    assert outputs[1][1].startswith("Added 100 of 150 blocks before Notion refused")
    assert "rate_limited" in outputs[1][1]
