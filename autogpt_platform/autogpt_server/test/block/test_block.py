from autogpt_server.data.block import get_blocks
from autogpt_server.util.test import execute_block_test


def test_available_blocks():
    for block in get_blocks().values():
        execute_block_test(type(block)())
