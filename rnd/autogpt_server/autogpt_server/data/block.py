from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generator, Generic, Type, TypeVar, cast

import jsonref
import jsonschema
from prisma.models import AgentBlock
from pydantic import BaseModel

from autogpt_server.util import json

BlockInput = dict[str, Any]
BlockData = tuple[str, Any]
BlockOutput = Generator[BlockData, None, None]


class BlockSchema(BaseModel):
    cached_jsonschema: ClassVar[dict[str, Any]] = {}

    @classmethod
    def jsonschema(cls) -> dict[str, Any]:
        if cls.cached_jsonschema:
            return cls.cached_jsonschema

        model = jsonref.replace_refs(cls.model_json_schema())

        def ref_to_dict(obj):
            if isinstance(obj, dict):
                return {
                    key: ref_to_dict(value)
                    for key, value in obj.items()
                    if not key.startswith("$")
                }
            elif isinstance(obj, list):
                return [ref_to_dict(item) for item in obj]
            return obj

        cls.cached_jsonschema = cast(dict[str, Any], ref_to_dict(model))
        return cls.cached_jsonschema

    @classmethod
    def validate_data(cls, data: BlockInput) -> str | None:
        """
        Validate the data against the schema.
        Returns the validation error message if the data does not match the schema.
        """
        try:
            jsonschema.validate(data, cls.jsonschema())
            return None
        except jsonschema.ValidationError as e:
            return str(e)

    @classmethod
    def validate_field(cls, field_name: str, data: BlockInput) -> str | None:
        """
        Validate the data against a specific property (one of the input/output name).
        Returns the validation error message if the data does not match the schema.
        """
        model_schema = cls.jsonschema().get("properties", {})
        if not model_schema:
            return f"Invalid model schema {cls}"

        property_schema = model_schema.get(field_name)
        if not property_schema:
            return f"Invalid property name {field_name}"

        try:
            jsonschema.validate(json.to_dict(data), property_schema)
            return None
        except jsonschema.ValidationError as e:
            return str(e)

    @classmethod
    def get_fields(cls) -> set[str]:
        return set(cls.model_fields.keys())

    @classmethod
    def get_required_fields(cls) -> set[str]:
        return {
            field
            for field, field_info in cls.model_fields.items()
            if field_info.is_required()
        }


BlockSchemaInputType = TypeVar("BlockSchemaInputType", bound=BlockSchema)
BlockSchemaOutputType = TypeVar("BlockSchemaOutputType", bound=BlockSchema)


class EmptySchema(BlockSchema):
    pass


class Block(ABC, Generic[BlockSchemaInputType, BlockSchemaOutputType]):
    def __init__(
        self,
        id: str = "",
        input_schema: Type[BlockSchemaInputType] = EmptySchema,
        output_schema: Type[BlockSchemaOutputType] = EmptySchema,
            test_input: BlockInput | list[BlockInput] | None = None,
            test_output: BlockData | list[BlockData] | None = None,
            test_mock: dict[str, Any] | None = None,
    ):
        """
        Initialize the block with the given schema.
        
        Args:
            id: The unique identifier for the block, this value will be persisted in the
                DB. So it should be a unique and constant across the application run.
                Use the UUID format for the ID.
            input_schema: The schema, defined as a Pydantic model, for the input data.
            output_schema: The schema, defined as a Pydantic model, for the output data.
            test_input: The list or single sample input data for the block, for testing.
            test_output: The list or single expected output if the test_input is run.
            test_mock: function names on the block implementation to mock on test run.
        """
        self.id = id
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.test_input = test_input
        self.test_output = test_output
        self.test_mock = test_mock

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

    @property
    def name(self):
        return self.__class__.__name__

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "inputSchema": self.input_schema.jsonschema(),
            "outputSchema": self.output_schema.jsonschema(),
        }

    def execute(self, input_data: BlockInput) -> BlockOutput:
        if error := self.input_schema.validate_data(input_data):
            raise ValueError(
                f"Unable to execute block with invalid input data: {error}"
            )

        for output_name, output_data in self.run(self.input_schema(**input_data)):
            if error := self.output_schema.validate_field(output_name, output_data):
                raise ValueError(f"Block produced an invalid output data: {error}")
            yield output_name, output_data


# ======================= Block Helper Functions ======================= #

def get_blocks() -> dict[str, Block]:
    from autogpt_server.blocks import AVAILABLE_BLOCKS  # noqa: E402
    return AVAILABLE_BLOCKS


async def initialize_blocks() -> None:
    for block in get_blocks().values():
        if await AgentBlock.prisma().find_unique(where={"id": block.id}):
            continue

        await AgentBlock.prisma().create(
            data={
                "id": block.id,
                "name": block.name,
                "inputSchema": json.dumps(block.input_schema.jsonschema()),
                "outputSchema": json.dumps(block.output_schema.jsonschema()),
            }
        )


def get_block(block_id: str) -> Block | None:
    return get_blocks().get(block_id)
