import os
import re
from typing import Type

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.util.test import execute_block_test


class BlockInstallationBlock(Block):
    """
    This block allows the verification and installation of other blocks in the system.

    NOTE:
        This block allows remote code execution on the server, and it should be used
        for development purposes only.
    """

    class Input(BlockSchema):
        code: str

    class Output(BlockSchema):
        success: str
        error: str

    def __init__(self):
        super().__init__(
            id="45e78db5-03e9-447f-9395-308d712f5f08",
            description="Given a code string, this block allows the verification and installation of a block code into the system.",
            categories={BlockCategory.BASIC},
            input_schema=BlockInstallationBlock.Input,
            output_schema=BlockInstallationBlock.Output,
            disabled=True,
        )

    def run(self, input_data: Input) -> BlockOutput:
        code = input_data.code

        if search := re.search(r"class (\w+)\(Block\):", code):
            class_name = search.group(1)
        else:
            yield "error", "No class found in the code."
            return

        if search := re.search(r"id=\"(\w+-\w+-\w+-\w+-\w+)\"", code):
            file_name = search.group(1)
        else:
            yield "error", "No UUID found in the code."
            return

        block_dir = os.path.dirname(__file__)
        file_path = f"{block_dir}/{file_name}.py"
        module_name = f"autogpt_server.blocks.{file_name}"
        with open(file_path, "w") as f:
            f.write(code)

        try:
            module = __import__(module_name, fromlist=[class_name])
            block_class: Type[Block] = getattr(module, class_name)
            block = block_class()
            execute_block_test(block)
            yield "success", "Block installed successfully."
        except Exception as e:
            os.remove(file_path)
            yield "error", f"[Code]\n{code}\n\n[Error]\n{str(e)}"
