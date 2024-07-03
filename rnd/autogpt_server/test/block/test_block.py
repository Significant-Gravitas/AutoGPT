from autogpt_server.blocks import AVAILABLE_BLOCKS


def test_available_blocks():
    for name, block in AVAILABLE_BLOCKS.items():
        block.execute_block_test()
