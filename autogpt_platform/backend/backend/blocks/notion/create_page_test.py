import pytest

from backend.blocks.notion.create_page import NotionCreatePageBlock


def _text(block: dict, block_type: str) -> str:
    return block[block_type]["rich_text"][0]["text"]["content"]


@pytest.mark.parametrize(
    "content",
    [
        "2024 was strong. Revenue grew 20%.",
        "3.5 stars. Would buy again.",
        "9am. Meeting",
    ],
)
def test_sentences_starting_with_number_stay_paragraphs(content):
    blocks = NotionCreatePageBlock._markdown_to_blocks(content)

    assert len(blocks) == 1
    assert blocks[0]["type"] == "paragraph"
    assert _text(blocks[0], "paragraph") == content


def test_consecutive_numbered_items_are_parsed():
    blocks = NotionCreatePageBlock._markdown_to_blocks("1. First\n2. Second")

    assert [block["type"] for block in blocks] == ["numbered_list_item"] * 2
    assert [_text(block, "numbered_list_item") for block in blocks] == [
        "First",
        "Second",
    ]


def test_multi_digit_numbered_item_is_parsed():
    blocks = NotionCreatePageBlock._markdown_to_blocks("10. Tenth")

    assert len(blocks) == 1
    assert blocks[0]["type"] == "numbered_list_item"
    assert _text(blocks[0], "numbered_list_item") == "Tenth"


def test_indented_numbered_item_is_parsed():
    blocks = NotionCreatePageBlock._markdown_to_blocks("   1. Indented")

    assert len(blocks) == 1
    assert blocks[0]["type"] == "numbered_list_item"
    assert _text(blocks[0], "numbered_list_item") == "Indented"


def test_bulleted_item_is_still_parsed():
    blocks = NotionCreatePageBlock._markdown_to_blocks("- Bullet")

    assert len(blocks) == 1
    assert blocks[0]["type"] == "bulleted_list_item"
    assert _text(blocks[0], "bulleted_list_item") == "Bullet"
