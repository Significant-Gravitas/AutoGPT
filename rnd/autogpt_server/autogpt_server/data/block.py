from abc import ABC, abstractmethod
from typing import Any, Generator, Generic, TypeVar, Type

import jsonschema
from prisma.models import AgentBlock
from pydantic import BaseModel

BlockData = dict[str, Any]


class BlockSchema(BaseModel):
    """
    A schema for the block input and output data.
    The dictionary structure is an object-typed `jsonschema`.
    The top-level properties are the block input/output names.

    BlockSchema is inherently a pydantic schema with some helper methods.
    """

    @classmethod
    def validate_data(cls, data: BlockData) -> str | None:
        """
        Validate the data against the schema.
        Returns the validation error message if the data does not match the schema.
        """
        try:
            jsonschema.validate(data, cls.model_json_schema())
            return None
        except jsonschema.ValidationError as e:
            return str(e)

    @classmethod
    def validate_field(cls, field_name: str, data: BlockData) -> str | None:
        """
        Validate the data against a specific property (one of the input/output name).
        Returns the validation error message if the data does not match the schema.
        """
        model_schema = cls.model_json_schema().get("properties", {})
        if not model_schema:
            return f"Invalid model schema {cls}"

        property_schema = model_schema.get(field_name)
        if not property_schema:
            return f"Invalid property name {field_name}"

        try:
            jsonschema.validate(data, property_schema)
            return None
        except jsonschema.ValidationError as e:
            return str(e)

    @classmethod
    def get_fields(cls) -> set[str]:
        return set(cls.model_json_schema().get("properties", {}).keys())

    @classmethod
    def get_required_fields(cls) -> set[str]:
        return set(cls.model_json_schema().get("required", {}))


BlockOutput = Generator[tuple[str, Any], None, None]
BlockSchemaInputType = TypeVar('BlockSchemaInputType', bound=BlockSchema)
BlockSchemaOutputType = TypeVar('BlockSchemaOutputType', bound=BlockSchema)


class Block(ABC, Generic[BlockSchemaInputType, BlockSchemaOutputType]):
    def __init__(
            self,
            input_schema: Type[BlockSchemaInputType] = None,
            output_schema: Type[BlockSchemaOutputType] = None,
    ):
        self.input_schema = input_schema
        self.output_schema = output_schema

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

    @abstractmethod
    def run(self, input_data: BlockSchemaInputType) -> BlockOutput:
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
            "inputSchema": self.input_schema.model_json_schema(),
            "outputSchema": self.output_schema.model_json_schema(),
        }

    def execute(self, input_data: BlockData) -> BlockOutput:
        if error := self.input_schema.validate_data(input_data):
            raise ValueError(
                f"Unable to execute block with invalid input data: {error}"
            )

        for output_name, output_data in self.run(self.input_schema(**input_data)):
            if error := self.output_schema.validate_field(output_name, output_data):
                raise ValueError(
                    f"Unable to execute block with invalid output data: {error}"
                )
            yield output_name, output_data


# ======================= Block Helper Functions ======================= #

from autogpt_server.blocks import AVAILABLE_BLOCKS  # noqa: E402


async def initialize_blocks() -> None:
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


def get_blocks() -> list[Block]:
    return list(AVAILABLE_BLOCKS.values())


def get_block(block_id: str) -> Block | None:
    return AVAILABLE_BLOCKS.get(block_id)
