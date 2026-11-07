import pytest

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


async def test_a_failure_after_the_first_request_names_the_blocks_added_in_the_error(
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

    assert [name for name, _ in outputs] == ["error"]
    assert outputs[0][1].startswith(
        "Added 100 of 150 blocks, the last with ID line 99, then a request failed"
    )
    assert "rate_limited" in outputs[0][1]


@pytest.mark.parametrize(
    "line, widths",
    [("a" * 4500, [2000, 2000, 500]), ("😀" * 1500, [2000, 1000])],
    ids=["ascii", "emoji"],
)
async def test_a_line_over_the_text_limit_is_split_within_one_block(
    notion_api, line: str, widths: list[int]
):
    sent = notion_api({"results": [{"id": "p"}]})

    await NotionAppendToPageBlock.append_content(TEST_CREDENTIALS, PAGE_ID, line)

    [paragraph] = sent[0][2]["children"]
    pieces = [seg["text"]["content"] for seg in paragraph["paragraph"]["rich_text"]]
    assert [len(piece.encode("utf-16-le")) // 2 for piece in pieces] == widths
    assert "".join(pieces) == line
