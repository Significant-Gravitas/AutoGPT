"""Regression tests for #15295: digit-led sentences were turned into numbered
list items and lost their prefix."""

import pytest

from backend.blocks.notion.create_page import NotionCreatePageBlock


def _text(block: dict) -> str:
    return block[block["type"]]["rich_text"][0]["text"]["content"]


@pytest.mark.parametrize(
    "line",
    [
        "2024 was strong. Revenue grew 20%.",
        "3.5 stars. Would buy again.",
        "9am. Meeting",
    ],
)
def test_digit_led_sentence_stays_a_paragraph(line):
    blocks = NotionCreatePageBlock._markdown_to_blocks(line)
    assert [b["type"] for b in blocks] == ["paragraph"]
    assert _text(blocks[0]) == line


def test_numbered_list_items_are_detected():
    blocks = NotionCreatePageBlock._markdown_to_blocks("1. First\n2. Second\n10. Ten")
    assert [b["type"] for b in blocks] == ["numbered_list_item"] * 3
    assert [_text(b) for b in blocks] == ["First", "Second", "Ten"]
