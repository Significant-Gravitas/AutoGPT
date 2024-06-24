import json
from abc import ABC, abstractmethod
from typing import Any, Generator, ClassVar

import jsonschema
from prisma.models import AgentBlock
from pydantic import BaseModel

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
            **kwargs: Any,
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

    def get_fields(self) -> set[str]:
        return set(self.jsonschema["properties"].keys())


BlockOutput = Generator[tuple[str, Any], None, None]


class Block(ABC, BaseModel):
    @classmethod
    @property
    @abstractmethod
    def id(cls) -> str:
        """
        The unique identifier for the block, this value will be persisted in the DB.
        So it should be a unique and constant across the application run.
        Use the UUID format for the ID.
        """
        pass

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
    def run(self, input_data: BlockData) -> BlockOutput:
        """
        Run the block with the given input data.
        Args:
            input_data: The input data with the structure of input_schema.
        Returns:
            A Generator that yields (output_name, output_data).
            output_name: One of the output name defined in Block's output_schema.
            output_data: The data for the output_name, matching the defined schema.
        """
        pass

    @classmethod
    @property
    def name(cls):
        return cls.__name__

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "inputSchema": self.input_schema.jsonschema,
            "outputSchema": self.output_schema.jsonschema,
        }

    def execute(self, input_data: BlockData) -> BlockOutput:
        if error := self.input_schema.validate_data(input_data):
            raise ValueError(
                f"Unable to execute block with invalid input data: {error}"
            )

        for output_name, output_data in self.run(input_data):
            if error := self.output_schema.validate_field(output_name, output_data):
                raise ValueError(
                    f"Unable to execute block with invalid output data: {error}"
                )
            yield output_name, output_data


# ===================== Inline-Block Implementations ===================== #


class ParrotBlock(Block):
    id: ClassVar[str] = "1ff065e9-88e8-4358-9d82-8dc91f622ba9"  # type: ignore
    input_schema: ClassVar[BlockSchema] = BlockSchema(  # type: ignore
        {
            "input": "string",
        }
    )
    output_schema: ClassVar[BlockSchema] = BlockSchema(  # type: ignore
        {
            "output": "string",
        }
    )

    def run(self, input_data: BlockData) -> BlockOutput:
        yield "output", input_data["input"]


class TextCombinerBlock(Block):
    id: ClassVar[str] = "db7d8f02-2f44-4c55-ab7a-eae0941f0c30"  # type: ignore
    input_schema: ClassVar[BlockSchema] = BlockSchema(  # type: ignore
        {
            "text1": "string",
            "text2": "string",
            "format": "string",
        }
    )
    output_schema: ClassVar[BlockSchema] = BlockSchema(  # type: ignore
        {
            "combined_text": "string",
        }
    )

    def run(self, input_data: BlockData) -> BlockOutput:
        yield "combined_text", input_data["format"].format(
            text1=input_data["text1"],
            text2=input_data["text2"],
        )


class PrintingBlock(Block):
    id: ClassVar[str] = "f3b1c1b2-4c4f-4f0d-8d2f-4c4f0d8d2f4c"  # type: ignore
    input_schema: ClassVar[BlockSchema] = BlockSchema(  # type: ignore
        {
            "text": "string",
        }
    )
    output_schema: ClassVar[BlockSchema] = BlockSchema(  # type: ignore
        {
            "status": "string",
        }
    )

    def run(self, input_data: BlockData) -> BlockOutput:
        yield "status", "printed"


# ======================= Block Helper Functions ======================= #

AVAILABLE_BLOCKS: dict[str, Block] = {}


async def initialize_blocks() -> None:
    global AVAILABLE_BLOCKS

    AVAILABLE_BLOCKS = {block.id: block() for block in Block.__subclasses__()}

    for block in AVAILABLE_BLOCKS.values():
        if await AgentBlock.prisma().find_unique(where={"id": block.id}):
            continue

        await AgentBlock.prisma().create(
            data={
                "id": block.id,
                "name": block.name,
                "inputSchema": str(block.input_schema),
                "outputSchema": str(block.output_schema),
            }
        )


async def get_blocks() -> list[Block]:
    if not AVAILABLE_BLOCKS:
        await initialize_blocks()
    return list(AVAILABLE_BLOCKS.values())


async def get_block(block_id: str) -> Block | None:
    if not AVAILABLE_BLOCKS:
        await initialize_blocks()
    return AVAILABLE_BLOCKS.get(block_id)
