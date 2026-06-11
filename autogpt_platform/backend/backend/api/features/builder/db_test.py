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


def _filtered_block_ids(type: BlockTypeFilter) -> set[str]:
    response = builder_db.get_blocks(type=type, page_size=10_000)
    return {b.id for b in response.blocks}


def test_trigger_blocks_appear_under_input_blocks():
    trigger_ids = _menu_block_ids(BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
    assert trigger_ids, "expected at least one trigger block to be loaded"
    assert trigger_ids <= _filtered_block_ids("input")


def test_trigger_blocks_do_not_appear_under_action_blocks():
    trigger_ids = _menu_block_ids(BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
    assert trigger_ids, "expected at least one trigger block to be loaded"
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
