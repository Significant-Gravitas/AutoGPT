import json
from datetime import datetime

import pytest
from prisma.models import BlocksRegistry

from backend.blocks.basic import (
    FileStoreBlock,
    PrintToConsoleBlock,
    ReverseListOrderBlock,
    StoreValueBlock,
)
from backend.data.block import (
    check_block_same,
    find_delta_blocks,
    recursive_json_compare,
)


@pytest.mark.asyncio
async def test_recursive_json_compare():
    db_block_definition = {
        "a": 1,
        "b": 2,
        "c": 3,
    }
    local_block_definition = {
        "a": 1,
        "b": 2,
        "c": 3,
    }
    assert recursive_json_compare(db_block_definition, local_block_definition)
    assert not recursive_json_compare(
        db_block_definition, {**local_block_definition, "d": 4}
    )
    assert not recursive_json_compare(
        db_block_definition, {**local_block_definition, "a": 2}
    )
    assert not recursive_json_compare(
        db_block_definition, {**local_block_definition, "b": 3}
    )
    assert not recursive_json_compare(
        db_block_definition, {**local_block_definition, "c": 4}
    )
    assert not recursive_json_compare(
        db_block_definition, {**local_block_definition, "a": 1, "b": 2, "c": 3, "d": 4}
    )
    assert recursive_json_compare({}, {})
    assert recursive_json_compare({"a": 1}, {"a": 1})
    assert not recursive_json_compare({"a": 1}, {"b": 1})
    assert not recursive_json_compare({"a": 1}, {"a": 2})
    assert not recursive_json_compare({"a": 1}, {"a": [1, 2]})
    assert not recursive_json_compare({"a": 1}, {"a": {"b": 1}})
    assert not recursive_json_compare({"a": 1}, {"a": {"b": 2}})
    assert not recursive_json_compare({"a": 1}, {"a": {"b": [1, 2]}})
    assert not recursive_json_compare({"a": 1}, {"a": {"b": {"c": 1}}})
    assert not recursive_json_compare({"a": 1}, {"a": {"b": {"c": 2}}})


@pytest.mark.asyncio
async def test_check_block_same():
    local_block = PrintToConsoleBlock()
    db_block = BlocksRegistry(
        id="f3b1c1b2-4c4f-4f0d-8d2f-4c4f0d8d2f4c",
        name=local_block.__class__.__name__,
        definition=json.dumps(local_block.to_dict()),  # type: ignore To much type magic going on here
        updatedAt=datetime.now(),
    )
    assert check_block_same(db_block, local_block)


@pytest.mark.asyncio
async def test_check_block_not_same():
    local_block = PrintToConsoleBlock()
    local_block_data = local_block.to_dict()
    local_block_data["description"] = "Hello, World!"

    db_block = BlocksRegistry(
        id="f3b1c1b2-4c4f-4f0d-8d2f-4c4f0d8d2f4c",
        name=local_block.__class__.__name__,
        definition=json.dumps(local_block_data),  # type: ignore To much type magic going on here
        updatedAt=datetime.now(),
    )
    assert not check_block_same(db_block, local_block)


@pytest.mark.asyncio
async def test_find_delta_blocks():
    now = datetime.now()
    store_value_block = StoreValueBlock()
    local_blocks = {
        PrintToConsoleBlock().id: PrintToConsoleBlock(),
        ReverseListOrderBlock().id: ReverseListOrderBlock(),
        FileStoreBlock().id: FileStoreBlock(),
        store_value_block.id: store_value_block,
    }
    db_blocks = {
        PrintToConsoleBlock().id: BlocksRegistry(
            id=PrintToConsoleBlock().id,
            name=PrintToConsoleBlock().__class__.__name__,
            definition=json.dumps(PrintToConsoleBlock().to_dict()),  # type: ignore To much type magic going on here
            updatedAt=now,
        ),
        ReverseListOrderBlock().id: BlocksRegistry(
            id=ReverseListOrderBlock().id,
            name=ReverseListOrderBlock().__class__.__name__,
            definition=json.dumps(ReverseListOrderBlock().to_dict()),  # type: ignore To much type magic going on here
            updatedAt=now,
        ),
        FileStoreBlock().id: BlocksRegistry(
            id=FileStoreBlock().id,
            name=FileStoreBlock().__class__.__name__,
            definition=json.dumps(FileStoreBlock().to_dict()),  # type: ignore To much type magic going on here
            updatedAt=now,
        ),
    }
    delta_blocks = find_delta_blocks(db_blocks, local_blocks)
    assert len(delta_blocks) == 1
    assert store_value_block.id in delta_blocks.keys()
    assert delta_blocks[store_value_block.id] == store_value_block


@pytest.mark.asyncio
async def test_find_delta_blocks_block_updated():
    now = datetime.now()
    store_value_block = StoreValueBlock()
    print_to_console_block_definition = PrintToConsoleBlock().to_dict()
    print_to_console_block_definition["description"] = "Hello, World!"
    local_blocks = {
        PrintToConsoleBlock().id: PrintToConsoleBlock(),
        ReverseListOrderBlock().id: ReverseListOrderBlock(),
        FileStoreBlock().id: FileStoreBlock(),
        store_value_block.id: store_value_block,
    }
    db_blocks = {
        PrintToConsoleBlock().id: BlocksRegistry(
            id=PrintToConsoleBlock().id,
            name=PrintToConsoleBlock().__class__.__name__,
            definition=json.dumps(print_to_console_block_definition),  # type: ignore To much type magic going on here
            updatedAt=now,
        ),
        ReverseListOrderBlock().id: BlocksRegistry(
            id=ReverseListOrderBlock().id,
            name=ReverseListOrderBlock().__class__.__name__,
            definition=json.dumps(ReverseListOrderBlock().to_dict()),  # type: ignore To much type magic going on here
            updatedAt=now,
        ),
        FileStoreBlock().id: BlocksRegistry(
            id=FileStoreBlock().id,
            name=FileStoreBlock().__class__.__name__,
            definition=json.dumps(FileStoreBlock().to_dict()),  # type: ignore To much type magic going on here
            updatedAt=now,
        ),
    }
    delta_blocks = find_delta_blocks(db_blocks, local_blocks)
    assert len(delta_blocks) == 2
    assert store_value_block.id in delta_blocks.keys()
    assert delta_blocks[store_value_block.id] == store_value_block
    assert PrintToConsoleBlock().id in delta_blocks.keys()


@pytest.mark.asyncio
async def test_find_delta_block_no_diff():
    now = datetime.now()
    local_blocks = {
        PrintToConsoleBlock().id: PrintToConsoleBlock(),
        ReverseListOrderBlock().id: ReverseListOrderBlock(),
        FileStoreBlock().id: FileStoreBlock(),
    }
    db_blocks = {
        PrintToConsoleBlock().id: BlocksRegistry(
            id=PrintToConsoleBlock().id,
            name=PrintToConsoleBlock().__class__.__name__,
            definition=json.dumps(PrintToConsoleBlock().to_dict()),  # type: ignore To much type magic going on here
            updatedAt=now,
        ),
        ReverseListOrderBlock().id: BlocksRegistry(
            id=ReverseListOrderBlock().id,
            name=ReverseListOrderBlock().__class__.__name__,
            definition=json.dumps(ReverseListOrderBlock().to_dict()),  # type: ignore To much type magic going on here
            updatedAt=now,
        ),
        FileStoreBlock().id: BlocksRegistry(
            id=FileStoreBlock().id,
            name=FileStoreBlock().__class__.__name__,
            definition=json.dumps(FileStoreBlock().to_dict()),  # type: ignore To much type magic going on here
            updatedAt=now,
        ),
    }
    delta_blocks = find_delta_blocks(db_blocks, local_blocks)
    assert len(delta_blocks) == 0
