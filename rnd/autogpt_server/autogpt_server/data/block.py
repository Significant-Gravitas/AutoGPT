import json
import jsonschema

from abc import ABC, abstractmethod
from prisma.models import AgentBlock
from pydantic import BaseModel
from typing import Any, ClassVar

BlockData = dict[str, Any]


class BlockSchema(BaseModel):
    """
    A schema for the block input and output data.
    The dictionary structure is an object-typed `jsonschema`.
    The top-level properties are the block input/output names.

    You can initialize this class by providing a dictionary of properties.
    The key is the string of the property name, and the value is either
    a string of the type or a dictionary of the jsonschema.

    You can also provide additional keyword arguments for additional properties.
    Like `name`, `required` (by default all properties are required), etc.

    Example:
    input_schema = BlockSchema({
        "system_prompt": "string",
        "user_prompt": "string",
        "max_tokens": "integer",
        "user_info": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        },
    }, required=["system_prompt", "user_prompt"])

    output_schema = BlockSchema({
        "on_complete": "string",
        "on_failures": "string",
    })
    """
    jsonschema: dict[str, Any]

    def __init__(
            self,
            properties: dict[str, str | dict],
            required: list[str] | None = None,
            **kwargs: Any
    ):
        schema = {
            "type": "object",
            "properties": {
                key: {"type": value} if isinstance(value, str) else value
                for key, value in properties.items()
            },
            "required": required or list(properties.keys()),
            **kwargs,
        }
        super().__init__(jsonschema=schema)

    def __str__(self) -> str:
        return json.dumps(self.jsonschema)

    def validate_data(self, data: BlockData) -> str | None:
        """
        Validate the data against the schema.
        Returns the validation error message if the data does not match the schema.
        """
        try:
            jsonschema.validate(data, self.jsonschema)
            return None
        except jsonschema.ValidationError as e:
            return str(e)

    def validate_field(self, field_name: str, data: BlockData) -> str | None:
        """
        Validate the data against a specific property (one of the input/output name).
        Returns the validation error message if the data does not match the schema.
        """
        property_schema = self.jsonschema["properties"].get(field_name)
        if not property_schema:
            return f"Invalid property name {field_name}"

        try:
            jsonschema.validate(data, property_schema)
            return None
        except jsonschema.ValidationError as e:
            return str(e)


class Block(ABC, BaseModel):
    @classmethod
    @property
    @abstractmethod
    def input_schema(cls) -> BlockSchema:
        """
        The schema for the block input data.
        The top-level properties are the possible input name expected by the block.
        """
        pass

    @classmethod
    @property
    @abstractmethod
    def output_schema(cls) -> BlockSchema:
        """
        The schema for the block output.
        The top-level properties are the possible output name produced by the block.
        """
        pass

    @abstractmethod
    async def run(self, input_data: BlockData) -> tuple[str, Any]:
        """
        Run the block with the given input data.
        Args:
            input_data: The input data with the structure of input_schema.
        Returns:
            The (output name, output data), matching the type in output_schema.
        """
        pass

    async def execute(self, input_data: BlockData) -> tuple[str, Any]:
        if error := self.input_schema.validate_data(input_data):
            raise ValueError(
                f"Unable to execute block with invalid input data: {error}"
            )

        output_name, output_data = await self.run(input_data)

        if error := self.output_schema.validate_field(output_name, output_data):
            raise ValueError(
                f"Unable to execute block with invalid output data: {error}"
            )

        return output_name, output_data


# ===================== Inline-Block Implementations ===================== #


class ParrotBlock(Block):
    input_schema: ClassVar[BlockSchema] = BlockSchema({  # type: ignore
        "input": "string",
    })
    output_schema: ClassVar[BlockSchema] = BlockSchema({  # type: ignore
        "output": "string",
    })

    async def run(self, input_data: BlockData) -> tuple[str, Any]:
        return "output", input_data["input"]


class TextCombinerBlock(Block):
    input_schema: ClassVar[BlockSchema] = BlockSchema({  # type: ignore
        "text1": "string",
        "text2": "string",
        "format": "string",
    })
    output_schema: ClassVar[BlockSchema] = BlockSchema({  # type: ignore
        "combined_text": "string",
    })

    async def run(self, input_data: BlockData) -> tuple[str, Any]:
        return "combined_text", input_data["format"].format(
            text1=input_data["text1"],
            text2=input_data["text2"],
        )


class PrintingBlock(Block):
    input_schema: ClassVar[BlockSchema] = BlockSchema({  # type: ignore
        "text": "string",
    })
    output_schema: ClassVar[BlockSchema] = BlockSchema({  # type: ignore
        "status": "string",
    })

    async def run(self, input_data: BlockData) -> tuple[str, Any]:
        print(input_data["text"])
        return "status", "printed"


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
                "inputSchema": str(block.input_schema),
                "outputSchema": str(block.output_schema),
            }
        )


async def get_block(name: str) -> Block:
    if not AVAILABLE_BLOCKS:
        await initialize_blocks()
    return AVAILABLE_BLOCKS[name]
