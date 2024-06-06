import inspect
import json

from abc import ABC, abstractmethod
from prisma.models import AgentBlock
from pydantic import BaseModel
from typing import ClassVar


class Block(ABC, BaseModel):
    @property
    @abstractmethod
    def input_schema(self) -> dict[str, str]:
        """
        The schema for the block input data. The keys are the names of the input,
        and the values are the types of the input.
        Example:
        {
            "system_prompt": "str",
            "user_prompt": "str",
            "max_tokens": "int",
        }
        """
        pass

    @property
    @abstractmethod
    def output_schema(self) -> dict[str, str]:
        """
        The schema for the block possible output. The keys are the names of the output,
        and the values are the types of the output.
        Example:
        {
            "on_completion": "str",
            "on_failure": "str",
        }
        """
        pass

    @abstractmethod
    def run(self, input_data: dict[str, str]) -> (str, str):
        """
        Run the block with the given input data.
        Args:
            input_data: The input data with the structure of input_schema.
        Returns:
            The output data, with the structure of one entry of the output_schema.
        """
        pass

    async def execute(self, input_data: dict[str, str]) -> dict[str, str]:
        result = self.run(input_data)
        if inspect.isawaitable(result):
            return await result
        return result


# ===================== Inline-Block Implementations ===================== #


class ParrotBlock(Block):
    input_schema: ClassVar[dict[str, str]] = {
        "input": "str",
    }
    output_schema: ClassVar[dict[str, str]] = {
        "output": "str",
    }

    def run(self, input_data: dict[str, str]) -> (str, str):
        return "output", input_data["input"]


class TextCombinerBlock(Block):
    input_schema: ClassVar[dict[str, str]] = {
        "text1": "str",
        "text2": "str",
        "format": "str",
    }
    output_schema: ClassVar[dict[str, str]] = {
        "combined_text": "str",
    }

    def run(self, input_data: dict[str, str]) -> (str, str):
        return "combined_text", input_data["format"].format(
            text1=input_data["text1"],
            text2=input_data["text2"],
        )


class PrintingBlock(Block):
    input_schema: ClassVar[dict[str, str]] = {
        "text": "str",
    }
    output_schema: ClassVar[dict[str, str]] = {}

    async def run(self, input_data: dict[str, str]) -> (str, str):
        print(input_data["text"])
        return "text", input_data["text"]


# ======================= Block Helper Functions ======================= #

AVAILABLE_BLOCKS: dict[str, Block] = {}


async def initialize_blocks() -> None:
    global AVAILABLE_BLOCKS

    AVAILABLE_BLOCKS = {block.__name__: block() for block in Block.__subclasses__()}

    for block_name, block in AVAILABLE_BLOCKS.items():
        existing_block = await AgentBlock.prisma().find_first(
            where={"name": block_name}
        )
        if existing_block:
            continue

        await AgentBlock.prisma().create(
            data={
                "name": block_name,
                "inputSchema": json.dumps(block.input_schema),
                "outputSchema": json.dumps(block.output_schema),
            }
        )


async def get_block(name: str) -> Block:
    if not AVAILABLE_BLOCKS:
        await initialize_blocks()
    return AVAILABLE_BLOCKS[name]
