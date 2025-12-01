import inspect
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator as AsyncGen
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Optional,
    Sequence,
    Type,
    TypeAlias,
    TypeVar,
    cast,
    get_origin,
)

import jsonref
import jsonschema
from prisma.models import AgentBlock
from prisma.types import AgentBlockCreateInput
from pydantic import BaseModel

from backend.data.model import NodeExecutionStats
from backend.integrations.providers import ProviderName
from backend.util import json
from backend.util.cache import cached
from backend.util.exceptions import (
    BlockError,
    BlockExecutionError,
    BlockInputError,
    BlockOutputError,
    BlockUnknownError,
)
from backend.util.settings import Config

from .model import (
    ContributorDetails,
    Credentials,
    CredentialsFieldInfo,
    CredentialsMetaInput,
    SchemaField,
    is_credentials_field_name,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .graph import Link

app_config = Config()

BlockInput = dict[str, Any]  # Input: 1 input pin consumes 1 data.
BlockOutputEntry = tuple[str, Any]  # Output data should be a tuple of (name, value).
BlockOutput = AsyncGen[BlockOutputEntry, None]  # Output: 1 output pin produces n data.
BlockTestOutput = BlockOutputEntry | tuple[str, Callable[[Any], bool]]
CompletedBlockOutput = dict[str, list[Any]]  # Completed stream, collected as a dict.


class BlockType(Enum):
    STANDARD = "Standard"
    INPUT = "Input"
    OUTPUT = "Output"
    NOTE = "Note"
    WEBHOOK = "Webhook"
    WEBHOOK_MANUAL = "Webhook (manual)"
    AGENT = "Agent"
    AI = "AI"
    AYRSHARE = "Ayrshare"


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
    HARDWARE = "Block that interacts with hardware."
    AGENT = "Block that interacts with other agents."
    CRM = "Block that interacts with CRM services."
    SAFETY = (
        "Block that provides AI safety mechanisms such as detecting harmful content"
    )
    PRODUCTIVITY = "Block that helps with productivity"
    ISSUE_TRACKING = "Block that helps with issue tracking"
    MULTIMEDIA = "Block that interacts with multimedia content"
    MARKETING = "Block that helps with marketing"

    def dict(self) -> dict[str, str]:
        return {"category": self.name, "description": self.value}


class BlockCostType(str, Enum):
    RUN = "run"  # cost X credits per run
    BYTE = "byte"  # cost X credits per byte
    SECOND = "second"  # cost X credits per second


class BlockCost(BaseModel):
    cost_amount: int
    cost_filter: BlockInput
    cost_type: BlockCostType

    def __init__(
        self,
        cost_amount: int,
        cost_type: BlockCostType = BlockCostType.RUN,
        cost_filter: Optional[BlockInput] = None,
        **data: Any,
    ) -> None:
        super().__init__(
            cost_amount=cost_amount,
            cost_filter=cost_filter or {},
            cost_type=cost_type,
            **data,
        )


class BlockInfo(BaseModel):
    id: str
    name: str
    inputSchema: dict[str, Any]
    outputSchema: dict[str, Any]
    costs: list[BlockCost]
    description: str
    categories: list[dict[str, str]]
    contributors: list[dict[str, Any]]
    staticOutput: bool
    uiType: str


class BlockSchema(BaseModel):
    cached_jsonschema: ClassVar[dict[str, Any] | None] = None

    @classmethod
    def clear_schema_cache(cls) -> None:
        """Clear the cached JSON schema for this class."""
        # Use None instead of {} because {} is truthy and would prevent regeneration
        cls.cached_jsonschema = None  # type: ignore

    @staticmethod
    def clear_all_schema_caches() -> None:
        """Clear cached JSON schemas for all BlockSchema subclasses."""
        def clear_recursive(cls: type) -> None:
            """Recursively clear cache for class and all subclasses."""
            if hasattr(cls, 'clear_schema_cache'):
                cls.clear_schema_cache()
            for subclass in cls.__subclasses__():
                clear_recursive(subclass)
        
        clear_recursive(BlockSchema)

    @classmethod
    def jsonschema(cls) -> dict[str, Any]:
        # Generate schema if not cached
        if not cls.cached_jsonschema:
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
        
        # Always post-process to ensure discriminator is present for multi-provider credentials fields
        # and refresh LLM model options to get latest enabled/disabled status
        # This handles cases where the schema was generated before the registry was loaded or updated
        # Note: We mutate the cached schema directly, which is safe since post-processing is idempotent
        # IMPORTANT: We always refresh options even if schema was cached, to ensure disabled models are excluded
        cls._ensure_discriminator_in_schema(cls.cached_jsonschema, cls)
        
        return cls.cached_jsonschema
    
    @staticmethod
    def _ensure_discriminator_in_schema(schema: dict[str, Any], model_class: type | None = None) -> None:
        """Ensure discriminator is present in multi-provider credentials fields and refresh LLM model options."""
        properties = schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                continue
            
            # Check if this is an LLM model field by checking the field definition
            is_llm_model_field = False
            if model_class and hasattr(model_class, "model_fields") and field_name in model_class.model_fields:
                try:
                    field_info = model_class.model_fields[field_name]
                    # Check if json_schema_extra has "options" (set by llm_model_schema_extra)
                    if hasattr(field_info, "json_schema_extra") and isinstance(field_info.json_schema_extra, dict):
                        if "options" in field_info.json_schema_extra:
                            is_llm_model_field = True
                    # Also check if the field type is LlmModel
                    if not is_llm_model_field and hasattr(field_info, "annotation"):
                        from backend.blocks.llm import LlmModel
                        from typing import get_origin, get_args
                        annotation = field_info.annotation
                        if annotation == LlmModel:
                            is_llm_model_field = True
                        else:
                            # Check for Optional[LlmModel] or Union types
                            origin = get_origin(annotation)
                            if origin:
                                args = get_args(annotation)
                                if LlmModel in args:
                                    is_llm_model_field = True
                except Exception:
                    pass
            
            # Only refresh LLM model options for LLM model fields
            # This prevents filtering other enum fields that aren't LLM models
            if is_llm_model_field:
                def refresh_options_in_schema(schema_part: dict[str, Any], path: str = "") -> bool:
                    """Recursively refresh options in schema part. Returns True if options were found and refreshed."""
                    import logging
                    logger = logging.getLogger(__name__)
                    
                    # Check for "options" key (used by frontend for select dropdowns)
                    has_options = "options" in schema_part and isinstance(schema_part.get("options"), list)
                    # Check for "enum" key (Pydantic generates this for Enum fields)
                    has_enum = "enum" in schema_part and isinstance(schema_part.get("enum"), list)
                    
                    if has_options or has_enum:
                        try:
                            from backend.data import llm_registry
                            # Always refresh options from registry to get latest enabled/disabled status
                            fresh_options = llm_registry.get_llm_model_schema_options()
                            if fresh_options:
                                # Get enabled model slugs from fresh options
                                enabled_slugs = {opt.get("value") for opt in fresh_options if isinstance(opt, dict) and "value" in opt}
                                
                                # Update "options" if present
                                if has_options:
                                    old_count = len(schema_part["options"])
                                    old_slugs = {opt.get("value") for opt in schema_part["options"] if isinstance(opt, dict) and "value" in opt}
                                    schema_part["options"] = fresh_options
                                    new_count = len(fresh_options)
                                    new_slugs = enabled_slugs
                                    
                                    # Log if there's a difference (models added/removed)
                                    if old_count != new_count or old_slugs != new_slugs:
                                        removed = old_slugs - new_slugs
                                        added = new_slugs - old_slugs
                                        if removed or added:
                                            logger.info(
                                                "Refreshed LLM model options for field %s%s: %d -> %d models. "
                                                "Removed: %s, Added: %s",
                                                field_name, f".{path}" if path else "", old_count, new_count, removed, added
                                            )
                                
                                # Update "enum" if present - filter to only enabled models
                                if has_enum:
                                    old_enum = schema_part.get("enum", [])
                                    # Filter enum values to only include enabled models
                                    filtered_enum = [val for val in old_enum if val in enabled_slugs]
                                    schema_part["enum"] = filtered_enum
                                    
                                    if len(old_enum) != len(filtered_enum):
                                        removed_enum = set(old_enum) - enabled_slugs
                                        logger.info(
                                            "Filtered LLM model enum for field %s%s: %d -> %d models. "
                                            "Removed disabled: %s",
                                            field_name, f".{path}" if path else "", len(old_enum), len(filtered_enum), removed_enum
                                        )
                                
                                return True
                        except Exception as e:
                            logger.warning("Failed to refresh LLM model options for field %s%s: %s", field_name, f".{path}" if path else "", e)
                    
                    # Check nested structures
                    for key in ["anyOf", "oneOf", "allOf"]:
                        if key in schema_part and isinstance(schema_part[key], list):
                            for idx, item in enumerate(schema_part[key]):
                                if isinstance(item, dict) and refresh_options_in_schema(item, f"{path}.{key}[{idx}]" if path else f"{key}[{idx}]"):
                                    return True
                    
                    return False
                
                refresh_options_in_schema(field_schema)
            
            # Check if this is a credentials field - look for credentials_provider or credentials_types
            has_credentials_provider = "credentials_provider" in field_schema
            has_credentials_types = "credentials_types" in field_schema
            
            if not (has_credentials_provider or has_credentials_types):
                continue
            
            # This is a credentials field
            providers = field_schema.get("credentials_provider", [])
            
            # If providers not in field schema yet, try to get from model class
            if not providers and model_class and hasattr(model_class, "model_fields"):
                try:
                    if field_name in model_class.model_fields:
                        field_info = model_class.model_fields[field_name]
                        # Check if this is a CredentialsMetaInput field
                        from backend.data.model import CredentialsMetaInput
                        if (hasattr(field_info, "annotation") and 
                            inspect.isclass(field_info.annotation) and
                            issubclass(get_origin(field_info.annotation) or field_info.annotation, CredentialsMetaInput)):
                            # Get providers from the annotation
                            providers_list = CredentialsMetaInput.allowed_providers.__func__(field_info.annotation)
                            if providers_list:
                                providers = list(providers_list)
                                field_schema["credentials_provider"] = providers
                except Exception:
                    pass
            
            # Check if this is a multi-provider field
            if isinstance(providers, list) and len(providers) > 1:
                # Multi-provider field - ensure discriminator is set
                if "discriminator" not in field_schema:
                    # Try to get discriminator from model field definition
                    discriminator_found = False
                    if model_class and hasattr(model_class, "model_fields") and field_name in model_class.model_fields:
                        try:
                            field_info = model_class.model_fields[field_name]
                            if hasattr(field_info, "json_schema_extra") and isinstance(field_info.json_schema_extra, dict):
                                discriminator = field_info.json_schema_extra.get("discriminator")
                                if discriminator:
                                    field_schema["discriminator"] = discriminator
                                    discriminator_found = True
                        except Exception:
                            pass
                    
                    # If not found, check if this looks like an LLM field and default to "model"
                    if not discriminator_found:
                        llm_providers = {"openai", "anthropic", "groq", "open_router", "llama_api", "aiml_api", "v0", "ollama"}
                        if any(p in llm_providers for p in providers):
                            field_schema["discriminator"] = "model"
                
                # If discriminator is "model", ensure discriminator_mapping is populated
                if field_schema.get("discriminator") == "model":
                    mapping = field_schema.get("discriminator_mapping")
                    # If mapping is empty, missing, or None, refresh from registry
                    if not mapping or (isinstance(mapping, dict) and len(mapping) == 0):
                        try:
                            from backend.data import llm_registry
                            refreshed_mapping = llm_registry.get_llm_discriminator_mapping()
                            if refreshed_mapping:
                                field_schema["discriminator_mapping"] = refreshed_mapping
                            else:
                                # Ensure at least an empty dict is present
                                field_schema["discriminator_mapping"] = {}
                        except Exception:
                            if "discriminator_mapping" not in field_schema:
                                field_schema["discriminator_mapping"] = {}

    @classmethod
    def validate_data(cls, data: BlockInput) -> str | None:
        return json.validate_with_jsonschema(
            schema=cls.jsonschema(),
            data={k: v for k, v in data.items() if v is not None},
        )

    @classmethod
    def get_mismatch_error(cls, data: BlockInput) -> str | None:
        return cls.validate_data(data)

    @classmethod
    def get_field_schema(cls, field_name: str) -> dict[str, Any]:
        model_schema = cls.jsonschema().get("properties", {})
        if not model_schema:
            raise ValueError(f"Invalid model schema {cls}")

        property_schema = model_schema.get(field_name)
        if not property_schema:
            raise ValueError(f"Invalid property name {field_name}")

        return property_schema

    @classmethod
    def validate_field(cls, field_name: str, data: BlockInput) -> str | None:
        """
        Validate the data against a specific property (one of the input/output name).
        Returns the validation error message if the data does not match the schema.
        """
        try:
            property_schema = cls.get_field_schema(field_name)
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
        - Fields with annotation `CredentialsMetaInput` MUST be
          named `credentials` or `*_credentials`
        - Fields named `credentials` or `*_credentials` MUST be
          of type `CredentialsMetaInput`
        """
        super().__pydantic_init_subclass__(**kwargs)

        # Reset cached JSON schema to prevent inheriting it from parent class
        cls.cached_jsonschema = {}

        credentials_fields = cls.get_credentials_fields()

        for field_name in cls.get_fields():
            if is_credentials_field_name(field_name):
                if field_name not in credentials_fields:
                    raise TypeError(
                        f"Credentials field '{field_name}' on {cls.__qualname__} "
                        f"is not of type {CredentialsMetaInput.__name__}"
                    )

                credentials_fields[field_name].validate_credentials_field_schema(cls)

            elif field_name in credentials_fields:
                raise KeyError(
                    f"Credentials field '{field_name}' on {cls.__qualname__} "
                    "has invalid name: must be 'credentials' or *_credentials"
                )

    @classmethod
    def get_credentials_fields(cls) -> dict[str, type[CredentialsMetaInput]]:
        return {
            field_name: info.annotation
            for field_name, info in cls.model_fields.items()
            if (
                inspect.isclass(info.annotation)
                and issubclass(
                    get_origin(info.annotation) or info.annotation,
                    CredentialsMetaInput,
                )
            )
        }

    @classmethod
    def get_credentials_fields_info(cls) -> dict[str, CredentialsFieldInfo]:
        return {
            field_name: CredentialsFieldInfo.model_validate(
                cls.get_field_schema(field_name), by_alias=True
            )
            for field_name in cls.get_credentials_fields().keys()
        }

    @classmethod
    def get_input_defaults(cls, data: BlockInput) -> BlockInput:
        return data  # Return as is, by default.

    @classmethod
    def get_missing_links(cls, data: BlockInput, links: list["Link"]) -> set[str]:
        input_fields_from_nodes = {link.sink_name for link in links}
        return input_fields_from_nodes - set(data)

    @classmethod
    def get_missing_input(cls, data: BlockInput) -> set[str]:
        return cls.get_required_fields() - set(data)


class BlockSchemaInput(BlockSchema):
    """
    Base schema class for block inputs.
    All block input schemas should extend this class for consistency.
    """

    pass


class BlockSchemaOutput(BlockSchema):
    """
    Base schema class for block outputs that includes a standard error field.
    All block output schemas should extend this class to ensure consistent error handling.
    """

    error: str = SchemaField(
        description="Error message if the operation failed", default=""
    )


BlockSchemaInputType = TypeVar("BlockSchemaInputType", bound=BlockSchemaInput)
BlockSchemaOutputType = TypeVar("BlockSchemaOutputType", bound=BlockSchemaOutput)


class EmptyInputSchema(BlockSchemaInput):
    pass


class EmptyOutputSchema(BlockSchemaOutput):
    pass


# For backward compatibility - will be deprecated
EmptySchema = EmptyOutputSchema


# --8<-- [start:BlockWebhookConfig]
class BlockManualWebhookConfig(BaseModel):
    """
    Configuration model for webhook-triggered blocks on which
    the user has to manually set up the webhook at the provider.
    """

    provider: ProviderName
    """The service provider that the webhook connects to"""

    webhook_type: str
    """
    Identifier for the webhook type. E.g. GitHub has repo and organization level hooks.

    Only for use in the corresponding `WebhooksManager`.
    """

    event_filter_input: str = ""
    """
    Name of the block's event filter input.
    Leave empty if the corresponding webhook doesn't have distinct event/payload types.
    """

    event_format: str = "{event}"
    """
    Template string for the event(s) that a block instance subscribes to.
    Applied individually to each event selected in the event filter input.

    Example: `"pull_request.{event}"` -> `"pull_request.opened"`
    """


class BlockWebhookConfig(BlockManualWebhookConfig):
    """
    Configuration model for webhook-triggered blocks for which
    the webhook can be automatically set up through the provider's API.
    """

    resource_format: str
    """
    Template string for the resource that a block instance subscribes to.
    Fields will be filled from the block's inputs (except `payload`).

    Example: `f"{repo}/pull_requests"` (note: not how it's actually implemented)

    Only for use in the corresponding `WebhooksManager`.
    """
    # --8<-- [end:BlockWebhookConfig]


class Block(ABC, Generic[BlockSchemaInputType, BlockSchemaOutputType]):
    def __init__(
        self,
        id: str = "",
        description: str = "",
        contributors: list[ContributorDetails] = [],
        categories: set[BlockCategory] | None = None,
        input_schema: Type[BlockSchemaInputType] = EmptyInputSchema,
        output_schema: Type[BlockSchemaOutputType] = EmptyOutputSchema,
        test_input: BlockInput | list[BlockInput] | None = None,
        test_output: BlockTestOutput | list[BlockTestOutput] | None = None,
        test_mock: dict[str, Any] | None = None,
        test_credentials: Optional[Credentials | dict[str, Credentials]] = None,
        disabled: bool = False,
        static_output: bool = False,
        block_type: BlockType = BlockType.STANDARD,
        webhook_config: Optional[BlockWebhookConfig | BlockManualWebhookConfig] = None,
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
        self.webhook_config = webhook_config
        self.execution_stats: NodeExecutionStats = NodeExecutionStats()

        if self.webhook_config:
            if isinstance(self.webhook_config, BlockWebhookConfig):
                # Enforce presence of credentials field on auto-setup webhook blocks
                if not (cred_fields := self.input_schema.get_credentials_fields()):
                    raise TypeError(
                        "credentials field is required on auto-setup webhook blocks"
                    )
                # Disallow multiple credentials inputs on webhook blocks
                elif len(cred_fields) > 1:
                    raise ValueError(
                        "Multiple credentials inputs not supported on webhook blocks"
                    )

                self.block_type = BlockType.WEBHOOK
            else:
                self.block_type = BlockType.WEBHOOK_MANUAL

            # Enforce shape of webhook event filter, if present
            if self.webhook_config.event_filter_input:
                event_filter_field = self.input_schema.model_fields[
                    self.webhook_config.event_filter_input
                ]
                if not (
                    isinstance(event_filter_field.annotation, type)
                    and issubclass(event_filter_field.annotation, BaseModel)
                    and all(
                        field.annotation is bool
                        for field in event_filter_field.annotation.model_fields.values()
                    )
                ):
                    raise NotImplementedError(
                        f"{self.name} has an invalid webhook event selector: "
                        "field must be a BaseModel and all its fields must be boolean"
                    )

            # Enforce presence of 'payload' input
            if "payload" not in self.input_schema.model_fields:
                raise TypeError(
                    f"{self.name} is webhook-triggered but has no 'payload' input"
                )

            # Disable webhook-triggered block if webhook functionality not available
            if not app_config.platform_base_url:
                self.disabled = True

    @classmethod
    def create(cls: Type["Block"]) -> "Block":
        return cls()

    @abstractmethod
    async def run(self, input_data: BlockSchemaInputType, **kwargs) -> BlockOutput:
        """
        Run the block with the given input data.
        Args:
            input_data: The input data with the structure of input_schema.

        Kwargs: Currently 14/02/2025 these include
            graph_id: The ID of the graph.
            node_id: The ID of the node.
            graph_exec_id: The ID of the graph execution.
            node_exec_id: The ID of the node execution.
            user_id: The ID of the user.

        Returns:
            A Generator that yields (output_name, output_data).
            output_name: One of the output name defined in Block's output_schema.
            output_data: The data for the output_name, matching the defined schema.
        """
        # --- satisfy the type checker, never executed -------------
        if False:  # noqa: SIM115
            yield "name", "value"  # pyright: ignore[reportMissingYield]
        raise NotImplementedError(f"{self.name} does not implement the run method.")

    async def run_once(
        self, input_data: BlockSchemaInputType, output: str, **kwargs
    ) -> Any:
        async for item in self.run(input_data, **kwargs):
            name, data = item
            if name == output:
                return data
        raise ValueError(f"{self.name} did not produce any output for {output}")

    def merge_stats(self, stats: NodeExecutionStats) -> NodeExecutionStats:
        self.execution_stats += stats
        return self.execution_stats

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

    def get_info(self) -> BlockInfo:
        from backend.data.credit import get_block_cost

        return BlockInfo(
            id=self.id,
            name=self.name,
            inputSchema=self.input_schema.jsonschema(),
            outputSchema=self.output_schema.jsonschema(),
            costs=get_block_cost(self),
            description=self.description,
            categories=[category.dict() for category in self.categories],
            contributors=[
                contributor.model_dump() for contributor in self.contributors
            ],
            staticOutput=self.static_output,
            uiType=self.block_type.value,
        )

    async def execute(self, input_data: BlockInput, **kwargs) -> BlockOutput:
        try:
            async for output_name, output_data in self._execute(input_data, **kwargs):
                yield output_name, output_data
        except Exception as ex:
            if not isinstance(ex, BlockError):
                raise BlockUnknownError(
                    message=str(ex),
                    block_name=self.name,
                    block_id=self.id,
                ) from ex
            else:
                raise ex

    async def _execute(self, input_data: BlockInput, **kwargs) -> BlockOutput:
        if error := self.input_schema.validate_data(input_data):
            raise BlockInputError(
                message=f"Unable to execute block with invalid input data: {error}",
                block_name=self.name,
                block_id=self.id,
            )

        async for output_name, output_data in self.run(
            self.input_schema(**{k: v for k, v in input_data.items() if v is not None}),
            **kwargs,
        ):
            if output_name == "error":
                raise BlockExecutionError(
                    message=output_data, block_name=self.name, block_id=self.id
                )
            if self.block_type == BlockType.STANDARD and (
                error := self.output_schema.validate_field(output_name, output_data)
            ):
                raise BlockOutputError(
                    message=f"Block produced an invalid output data: {error}",
                    block_name=self.name,
                    block_id=self.id,
                )
            yield output_name, output_data

    def is_triggered_by_event_type(
        self, trigger_config: dict[str, Any], event_type: str
    ) -> bool:
        if not self.webhook_config:
            raise TypeError("This method can't be used on non-trigger blocks")
        if not self.webhook_config.event_filter_input:
            return True
        event_filter = trigger_config.get(self.webhook_config.event_filter_input)
        if not event_filter:
            raise ValueError("Event filter is not configured on trigger")
        return event_type in [
            self.webhook_config.event_format.format(event=k)
            for k in event_filter
            if event_filter[k] is True
        ]


# Type alias for any block with standard input/output schemas
AnyBlockSchema: TypeAlias = Block[BlockSchemaInput, BlockSchemaOutput]


# ======================= Block Helper Functions ======================= #


def get_blocks() -> dict[str, Type[Block]]:
    from backend.blocks import load_all_blocks

    return load_all_blocks()


def is_block_auth_configured(
    block_cls: type[AnyBlockSchema],
) -> bool:
    """
    Check if a block has a valid authentication method configured at runtime.

    For example if a block is an OAuth-only block and there env vars are not set,
    do not show it in the UI.

    """
    from backend.sdk.registry import AutoRegistry

    # Create an instance to access input_schema
    try:
        block = block_cls()
    except Exception as e:
        # If we can't create a block instance, assume it's not OAuth-only
        logger.error(f"Error creating block instance for {block_cls.__name__}: {e}")
        return True
    logger.debug(
        f"Checking if block {block_cls.__name__} has a valid provider configured"
    )

    # Get all credential inputs from input schema
    credential_inputs = block.input_schema.get_credentials_fields_info()
    required_inputs = block.input_schema.get_required_fields()
    if not credential_inputs:
        logger.debug(
            f"Block {block_cls.__name__} has no credential inputs - Treating as valid"
        )
        return True

    # Check credential inputs
    if len(required_inputs.intersection(credential_inputs.keys())) == 0:
        logger.debug(
            f"Block {block_cls.__name__} has only optional credential inputs"
            " - will work without credentials configured"
        )

    # Check if the credential inputs for this block are correctly configured
    for field_name, field_info in credential_inputs.items():
        provider_names = field_info.provider
        if not provider_names:
            logger.warning(
                f"Block {block_cls.__name__} "
                f"has credential input '{field_name}' with no provider options"
                " - Disabling"
            )
            return False

        # If a field has multiple possible providers, each one needs to be usable to
        # prevent breaking the UX
        for _provider_name in provider_names:
            provider_name = _provider_name.value
            if provider_name in ProviderName.__members__.values():
                logger.debug(
                    f"Block {block_cls.__name__} credential input '{field_name}' "
                    f"provider '{provider_name}' is part of the legacy provider system"
                    " - Treating as valid"
                )
                break

            provider = AutoRegistry.get_provider(provider_name)
            if not provider:
                logger.warning(
                    f"Block {block_cls.__name__} credential input '{field_name}' "
                    f"refers to unknown provider '{provider_name}' - Disabling"
                )
                return False

            # Check the provider's supported auth types
            if field_info.supported_types != provider.supported_auth_types:
                logger.warning(
                    f"Block {block_cls.__name__} credential input '{field_name}' "
                    f"has mismatched supported auth types (field <> Provider): "
                    f"{field_info.supported_types} != {provider.supported_auth_types}"
                )

            if not (supported_auth_types := provider.supported_auth_types):
                # No auth methods are been configured for this provider
                logger.warning(
                    f"Block {block_cls.__name__} credential input '{field_name}' "
                    f"provider '{provider_name}' "
                    "has no authentication methods configured - Disabling"
                )
                return False

            # Check if provider supports OAuth
            if "oauth2" in supported_auth_types:
                # Check if OAuth environment variables are set
                if (oauth_config := provider.oauth_config) and bool(
                    os.getenv(oauth_config.client_id_env_var)
                    and os.getenv(oauth_config.client_secret_env_var)
                ):
                    logger.debug(
                        f"Block {block_cls.__name__} credential input '{field_name}' "
                        f"provider '{provider_name}' is configured for OAuth"
                    )
                else:
                    logger.error(
                        f"Block {block_cls.__name__} credential input '{field_name}' "
                        f"provider '{provider_name}' "
                        "is missing OAuth client ID or secret - Disabling"
                    )
                    return False

        logger.debug(
            f"Block {block_cls.__name__} credential input '{field_name}' is valid; "
            f"supported credential types: {', '.join(field_info.supported_types)}"
        )

    return True


async def initialize_blocks() -> None:
    # Refresh LLM registry before initializing blocks so blocks can use registry data
    # This ensures the registry cache is populated even in executor context
    try:
        from backend.data import llm_registry
        from backend.data.block_cost_config import refresh_llm_costs
        
        # Only refresh if we have DB access (check if Prisma is connected)
        from backend.data.db import is_connected
        if is_connected():
            await llm_registry.refresh_llm_registry()
            refresh_llm_costs()
            logger.info("LLM registry refreshed during block initialization")
        else:
            logger.warning("Prisma not connected, skipping LLM registry refresh during block initialization")
    except Exception as exc:
        logger.warning("Failed to refresh LLM registry during block initialization: %s", exc)
    
    # First, sync all provider costs to blocks
    # Imported here to avoid circular import
    from backend.sdk.cost_integration import sync_all_provider_costs

    sync_all_provider_costs()

    for cls in get_blocks().values():
        block = cls()
        existing_block = await AgentBlock.prisma().find_first(
            where={"OR": [{"id": block.id}, {"name": block.name}]}
        )
        if not existing_block:
            await AgentBlock.prisma().create(
                data=AgentBlockCreateInput(
                    id=block.id,
                    name=block.name,
                    inputSchema=json.dumps(block.input_schema.jsonschema()),
                    outputSchema=json.dumps(block.output_schema.jsonschema()),
                )
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


# Note on the return type annotation: https://github.com/microsoft/pyright/issues/10281
def get_block(block_id: str) -> AnyBlockSchema | None:
    cls = get_blocks().get(block_id)
    return cls() if cls else None


@cached(ttl_seconds=3600)
def get_webhook_block_ids() -> Sequence[str]:
    return [
        id
        for id, B in get_blocks().items()
        if B().block_type in (BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
    ]


@cached(ttl_seconds=3600)
def get_io_block_ids() -> Sequence[str]:
    return [
        id
        for id, B in get_blocks().items()
        if B().block_type in (BlockType.INPUT, BlockType.OUTPUT)
    ]
