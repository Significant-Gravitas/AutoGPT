import pytest

from backend.api.features.builder import db as builder_db
from backend.api.features.builder.model import BlockTypeFilter
from backend.blocks import load_all_blocks
from backend.blocks._base import AnyBlockSchema, BlockType


def _menu_block_ids(*block_types: BlockType) -> set[str]:
    ids: set[str] = set()
    for block_cls in load_all_blocks().values():
        block: AnyBlockSchema = block_cls()
        if block.disabled or block.id in builder_db.EXCLUDED_BLOCK_IDS:
            continue
        if block.block_type in block_types:
            ids.add(block.id)
    return ids


def _filtered_block_ids(type_filter: BlockTypeFilter) -> set[str]:
    response = builder_db.get_blocks(type=type_filter, page_size=1_000_000)
    assert response.pagination.total_items == len(
        response.blocks
    ), "page_size too small — results were truncated"
    return {b.id for b in response.blocks}


def test_block_menu_type_classifies_triggers_as_input():
    assert builder_db._block_menu_type(BlockType.INPUT) == "input"
    assert builder_db._block_menu_type(BlockType.WEBHOOK) == "input"
    assert builder_db._block_menu_type(BlockType.WEBHOOK_MANUAL) == "input"


def test_block_menu_type_classifies_output_and_action():
    assert builder_db._block_menu_type(BlockType.OUTPUT) == "output"
    assert builder_db._block_menu_type(BlockType.STANDARD) == "action"
    assert builder_db._block_menu_type(BlockType.AI) == "action"
    assert builder_db._block_menu_type(BlockType.AGENT) == "action"


def test_every_block_type_has_a_menu_category():
    # Guards against a future BlockType being classified as something other than
    # one of the three menu categories the frontend knows how to render.
    for block_type in BlockType:
        assert builder_db._block_menu_type(block_type) in ("input", "output", "action")


def test_trigger_blocks_appear_under_input_blocks():
    trigger_ids = _menu_block_ids(BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
    if not trigger_ids:
        pytest.skip("no trigger blocks loaded in this environment")
    assert trigger_ids <= _filtered_block_ids("input")


def test_trigger_blocks_do_not_appear_under_action_blocks():
    trigger_ids = _menu_block_ids(BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
    if not trigger_ids:
        pytest.skip("no trigger blocks loaded in this environment")
    assert not trigger_ids & _filtered_block_ids("action")


def test_input_output_action_partition_all_blocks():
    all_ids = _filtered_block_ids("all")
    input_ids = _filtered_block_ids("input")
    output_ids = _filtered_block_ids("output")
    action_ids = _filtered_block_ids("action")

    assert input_ids | output_ids | action_ids == all_ids
    assert not input_ids & output_ids
    assert not input_ids & action_ids
    assert not output_ids & action_ids


def test_filtered_counts_match_block_menu_type_classification():
    # The get_blocks filter and the _get_static_counts badges both classify via
    # _block_menu_type, so the filtered set sizes must match a direct count.
    expected = {"input": 0, "output": 0, "action": 0}
    for block_cls in load_all_blocks().values():
        block: AnyBlockSchema = block_cls()
        if block.disabled or block.id in builder_db.EXCLUDED_BLOCK_IDS:
            continue
        expected[builder_db._block_menu_type(block.block_type)] += 1

    assert len(_filtered_block_ids("input")) == expected["input"]
    assert len(_filtered_block_ids("output")) == expected["output"]
    assert len(_filtered_block_ids("action")) == expected["action"]
