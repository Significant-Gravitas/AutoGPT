import logging

from autogpt_server.data.block import Block, get_blocks

logger = logging.getLogger(__name__)
log = print


def execute_block_test(block: Block):
    prefix = f"[Test-{block.name}]"

    if not block.test_input or not block.test_output:
        log(f"{prefix} No test data provided")
        return
    if not isinstance(block.test_input, list):
        block.test_input = [block.test_input]
    if not isinstance(block.test_output, list):
        block.test_output = [block.test_output]

    output_index = 0
    log(f"{prefix} Executing {len(block.test_input)} tests...")
    prefix = " " * 4 + prefix

    for mock_name, mock_obj in (block.test_mock or {}).items():
        log(f"{prefix} mocking {mock_name}...")
        setattr(block, mock_name, mock_obj)

    for input_data in block.test_input:
        log(f"{prefix} in: {input_data}")

        for output_name, output_data in block.execute(input_data):
            if output_index >= len(block.test_output):
                raise ValueError(f"{prefix} produced output more than expected")
            ex_output_name, ex_output_data = block.test_output[output_index]

            def compare(data, expected_data):
                if isinstance(expected_data, type):
                    is_matching = isinstance(data, expected_data)
                else:
                    is_matching = data == expected_data

                mark = "✅" if is_matching else "❌"
                log(f"{prefix} {mark} comparing `{data}` vs `{expected_data}`")
                if not is_matching:
                    raise ValueError(f"{prefix}: wrong output {data} vs {expected_data}")

            compare(output_name, ex_output_name)
            compare(output_data, ex_output_data)
            output_index += 1


def test_available_blocks():
    for block in get_blocks().values():
        execute_block_test(type(block)())
