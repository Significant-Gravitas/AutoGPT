"""A first-turn block renders as the first user message carries it."""

from backend.copilot.context_blocks import ContextBlock, render_context_blocks


def test_a_block_renders_its_head_items_and_tail_inside_its_tags():
    block = ContextBlock(
        tag="routines",
        head="Standing work:",
        items=("- weekly report", "- inbox sweep\nwith a second line"),
        tail="Ask before switching one on.",
    )

    assert block.text == (
        "<routines>\n"
        "Standing work:\n"
        "- weekly report\n"
        "- inbox sweep\nwith a second line\n"
        "Ask before switching one on.\n"
        "</routines>\n\n"
    )


def test_an_empty_head_or_tail_adds_no_line():
    assert ContextBlock(tag="t", items=("- a",)).text == "<t>\n- a\n</t>\n\n"
    assert ContextBlock(tag="t", head="only").text == "<t>\nonly\n</t>\n\n"


def test_blocks_render_in_order():
    first = ContextBlock(tag="a", head="1")
    second = ContextBlock(tag="b", head="2")

    assert render_context_blocks([first, second]) == first.text + second.text
