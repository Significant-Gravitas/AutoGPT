import inspect
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Generator,
    Generic,
    Optional,
    Type,
    TypeVar,
    cast,
    get_origin,
)

import jsonref
import jsonschema
from autogpt_libs.supabase_integration_credentials_store.types import Credentials
from prisma.models import AgentBlock
from pydantic import BaseModel

from backend.util import json

from .model import CREDENTIALS_FIELD_NAME, ContributorDetails, CredentialsMetaInput

BlockData = tuple[str, Any]  # Input & Output data should be a tuple of (name, data).
BlockInput = dict[str, Any]  # Input: 1 input pin consumes 1 data.
BlockOutput = Generator[BlockData, None, None]  # Output: 1 output pin produces n data.
CompletedBlockOutput = dict[str, list[Any]]  # Completed stream, collected as a dict.


class BlockType(Enum):
    STANDARD = "Standard"
    INPUT = "Input"
    OUTPUT = "Output"
    NOTE = "Note"


class BlockCategory(Enum):
    AI = "Block that leverages AI to perform a task."
    SOCIAL = "Block that interacts with social media platforms."
    TEXT = "Block that processes text data."
    SEARCH = "Block that searches or extracts information from the internet."
    BASIC = "Block that performs basic operations."
    INPUT = "Block that interacts with input of the graph."
    OUTPUT = "Block that interacts with output of the graph."
    LOGIC = "Programming logic to control the flow of your agent"
    COMMUNICATION = "Block that interacts with communication platforms."
    DEVELOPER_TOOLS = "Developer tools such as GitHub blocks."
    DATA = "Block that interacts with structured data."

    def dict(self) -> dict[str, str]:
        return {"category": self.name, "description": self.value}


class BlockSchema(BaseModel):
    cached_jsonschema: ClassVar[dict[str, Any]] = {}

    @classmethod
    def jsonschema(cls) -> dict[str, Any]:
        if cls.cached_jsonschema:
            return cls.cached_jsonschema

        model = jsonref.replace_refs(cls.model_json_schema(), merge_props=True)

        def ref_to_dict(obj):
            if isinstance(obj, dict):
                # OpenAPI <3.1 does not support sibling fields that has a $ref key
                # So sometimes, the schema has an "allOf"/"anyOf"/"oneOf" with 1 item.
                keys = {"allOf", "anyOf", "oneOf"}
                one_key = next((k for k in keys if k in obj and len(obj[k]) == 1), None)
                if one_key:
                    obj.update(obj[one_key][0])

                return {
                    key: ref_to_dict(value)
                    for key, value in obj.items()
                    if not key.startswith("$") and key != one_key
                }
            elif isinstance(obj, list):
                return [ref_to_dict(item) for item in obj]
            return obj

        cls.cached_jsonschema = cast(dict[str, Any], ref_to_dict(model))

        # Set default properties values
        for field in cls.cached_jsonschema.get("properties", {}).values():
            if isinstance(field, dict) and "advanced" not in field:
                field["advanced"] = True

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

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Validates the schema definition. Rules:
        - Only one `CredentialsMetaInput` field may be present.
          - This field MUST be called `credentials`.
        - A field that is called `credentials` MUST be a `CredentialsMetaInput`.
        """
        super().__pydantic_init_subclass__(**kwargs)
        credentials_fields = [
            field_name
            for field_name, info in cls.model_fields.items()
            if (
                inspect.isclass(info.annotation)
                and issubclass(
                    get_origin(info.annotation) or info.annotation,
                    CredentialsMetaInput,
                )
            )
        ]
        if len(credentials_fields) > 1:
            raise ValueError(
                f"{cls.__qualname__} can only have one CredentialsMetaInput field"
            )
        elif (
            len(credentials_fields) == 1
            and credentials_fields[0] != CREDENTIALS_FIELD_NAME
        ):
            raise ValueError(
                f"CredentialsMetaInput field on {cls.__qualname__} "
                "must be named 'credentials'"
            )
        elif (
            len(credentials_fields) == 0
            and CREDENTIALS_FIELD_NAME in cls.model_fields.keys()
        ):
            raise TypeError(
                f"Field 'credentials' on {cls.__qualname__} "
                f"must be of type {CredentialsMetaInput.__name__}"
            )


BlockSchemaInputType = TypeVar("BlockSchemaInputType", bound=BlockSchema)
BlockSchemaOutputType = TypeVar("BlockSchemaOutputType", bound=BlockSchema)


class EmptySchema(BlockSchema):
    pass


class Block(ABC, Generic[BlockSchemaInputType, BlockSchemaOutputType]):
    def __init__(
        self,
        id: str = "",
        description: str = "",
        contributors: list[ContributorDetails] = [],
        categories: set[BlockCategory] | None = None,
        input_schema: Type[BlockSchemaInputType] = EmptySchema,
        output_schema: Type[BlockSchemaOutputType] = EmptySchema,
        test_input: BlockInput | list[BlockInput] | None = None,
        test_output: BlockData | list[BlockData] | None = None,
        test_mock: dict[str, Any] | None = None,
        test_credentials: Optional[Credentials] = None,
        disabled: bool = False,
        static_output: bool = False,
        block_type: BlockType = BlockType.STANDARD,
    ):
        """
        Initialize the block with the given schema.

        Args:
            id: The unique identifier for the block, this value will be persisted in the
                DB. So it should be a unique and constant across the application run.
                Use the UUID format for the ID.
            description: The description of the block, explaining what the block does.
            contributors: The list of contributors who contributed to the block.
            input_schema: The schema, defined as a Pydantic model, for the input data.
            output_schema: The schema, defined as a Pydantic model, for the output data.
            test_input: The list or single sample input data for the block, for testing.
            test_output: The list or single expected output if the test_input is run.
            test_mock: function names on the block implementation to mock on test run.
            disabled: If the block is disabled, it will not be available for execution.
            static_output: Whether the output links of the block are static by default.
        """
        self.id = id
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.test_input = test_input
        self.test_output = test_output
        self.test_mock = test_mock
        self.test_credentials = test_credentials
        self.description = description
        self.categories = categories or set()
        self.contributors = contributors or set()
        self.disabled = disabled
        self.static_output = static_output
        self.block_type = block_type

    @abstractmethod
    def run(self, input_data: BlockSchemaInputType, **kwargs) -> BlockOutput:
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
            "description": self.description,
            "categories": [category.dict() for category in self.categories],
            "contributors": [
                contributor.model_dump() for contributor in self.contributors
            ],
            "staticOutput": self.static_output,
            "uiType": self.block_type.value,
        }

    def execute(self, input_data: BlockInput, **kwargs) -> BlockOutput:
        if error := self.input_schema.validate_data(input_data):
            raise ValueError(
                f"Unable to execute block with invalid input data: {error}"
            )

        for output_name, output_data in self.run(
            self.input_schema(**input_data), **kwargs
        ):
            if output_name == "error":
                raise RuntimeError(output_data)
            if error := self.output_schema.validate_field(output_name, output_data):
                raise ValueError(f"Block produced an invalid output data: {error}")
            yield output_name, output_data


# ======================= Block Helper Functions ======================= #


def get_blocks() -> dict[str, Block]:
    from backend.blocks import AVAILABLE_BLOCKS  # noqa: E402

    return AVAILABLE_BLOCKS


async def initialize_blocks() -> None:
    for block in get_blocks().values():
        existing_block = await AgentBlock.prisma().find_first(
            where={"OR": [{"id": block.id}, {"name": block.name}]}
        )
        if not existing_block:
            await AgentBlock.prisma().create(
                data={
                    "id": block.id,
                    "name": block.name,
                    "inputSchema": json.dumps(block.input_schema.jsonschema()),
                    "outputSchema": json.dumps(block.output_schema.jsonschema()),
                }
            )
            continue

        input_schema = json.dumps(block.input_schema.jsonschema())
        output_schema = json.dumps(block.output_schema.jsonschema())
        if (
            block.id != existing_block.id
            or block.name != existing_block.name
            or input_schema != existing_block.inputSchema
            or output_schema != existing_block.outputSchema
        ):
            await AgentBlock.prisma().update(
                where={"id": existing_block.id},
                data={
                    "id": block.id,
                    "name": block.name,
                    "inputSchema": input_schema,
                    "outputSchema": output_schema,
                },
            )


def get_block(block_id: str) -> Block | None:
    return get_blocks().get(block_id)
