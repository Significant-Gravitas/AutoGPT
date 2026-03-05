import asyncio
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Self, cast

from prisma.enums import SubmissionStatus
from prisma.models import (
    AgentGraph,
    AgentNode,
    AgentNodeLink,
    LibraryAgent,
    StoreListingVersion,
)
from prisma.types import (
    AgentGraphCreateInput,
    AgentGraphWhereInput,
    AgentNodeCreateInput,
    AgentNodeLinkCreateInput,
    StoreListingVersionWhereInput,
)
from pydantic import BaseModel, BeforeValidator, Field
from pydantic.fields import computed_field

from backend.blocks import get_block, get_blocks
from backend.blocks._base import Block, BlockType, EmptySchema
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.blocks.llm import LlmModel
from backend.integrations.providers import ProviderName
from backend.util import type as type_utils
from backend.util.exceptions import GraphNotAccessibleError, GraphNotInLibraryError
from backend.util.json import SafeJson
from backend.util.models import Pagination
from backend.util.request import parse_url

from .block import BlockInput
from .db import BaseDbModel
from .db import prisma as db
from .db import query_raw_with_schema, transaction
from .dynamic_fields import is_tool_pin, sanitize_pin_name
from .includes import AGENT_GRAPH_INCLUDE, AGENT_NODE_INCLUDE, MAX_GRAPH_VERSIONS_FETCH
from .model import CredentialsFieldInfo, CredentialsMetaInput, is_credentials_field_name

if TYPE_CHECKING:
    from backend.blocks._base import AnyBlockSchema

    from .execution import NodesInputMasks

logger = logging.getLogger(__name__)


class GraphSettings(BaseModel):
    # Use Annotated with BeforeValidator to coerce None to default values.
    # This handles cases where the database has null values for these fields.
    model_config = {"extra": "ignore"}

    human_in_the_loop_safe_mode: Annotated[
        bool, BeforeValidator(lambda v: v if v is not None else True)
    ] = True
    sensitive_action_safe_mode: Annotated[
        bool, BeforeValidator(lambda v: v if v is not None else False)
    ] = False

    @classmethod
    def from_graph(
        cls,
        graph: "GraphModel",
        hitl_safe_mode: bool | None = None,
        sensitive_action_safe_mode: bool = False,
    ) -> "GraphSettings":
        # Default to True if not explicitly set
        if hitl_safe_mode is None:
            hitl_safe_mode = True
        return cls(
            human_in_the_loop_safe_mode=hitl_safe_mode,
            sensitive_action_safe_mode=sensitive_action_safe_mode,
        )


class Link(BaseDbModel):
    source_id: str
    sink_id: str
    source_name: str
    sink_name: str
    is_static: bool = False

    @staticmethod
    def from_db(link: AgentNodeLink):
        return Link(
            id=link.id,
            source_name=link.sourceName,
            source_id=link.agentNodeSourceId,
            sink_name=link.sinkName,
            sink_id=link.agentNodeSinkId,
            is_static=link.isStatic,
        )

    def __hash__(self):
        return hash((self.source_id, self.sink_id, self.source_name, self.sink_name))


class Node(BaseDbModel):
    block_id: str
    input_default: BlockInput = Field(  # dict[input_name, default_value]
        default_factory=dict
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    input_links: list[Link] = Field(default_factory=list)
    output_links: list[Link] = Field(default_factory=list)

    @property
    def credentials_optional(self) -> bool:
        """
        Whether credentials are optional for this node.
        When True and credentials are not configured, the node will be skipped
        during execution rather than causing a validation error.
        """
        return self.metadata.get("credentials_optional", False)

    @property
    def block(self) -> "AnyBlockSchema | _UnknownBlockBase":
        """Get the block for this node. Returns UnknownBlock if block is deleted/missing."""
        block = get_block(self.block_id)
        if not block:
            # Log warning but don't raise exception - return a placeholder block for deleted blocks
            logger.warning(
                f"Block #{self.block_id} does not exist for Node #{self.id} (deleted/missing block), using UnknownBlock"
            )
            return _UnknownBlockBase(self.block_id)
        return block


class NodeModel(Node):
    graph_id: str
    graph_version: int

    webhook_id: Optional[str] = None
    # webhook: Optional["Webhook"] = None  # deprecated

    @staticmethod
    def from_db(node: AgentNode, for_export: bool = False) -> "NodeModel":
        obj = NodeModel(
            id=node.id,
            block_id=node.agentBlockId,
            input_default=type_utils.convert(node.constantInput, BlockInput),
            metadata=type_utils.convert(node.metadata, dict[str, Any]),
            graph_id=node.agentGraphId,
            graph_version=node.agentGraphVersion,
            webhook_id=node.webhookId,
        )
        obj.input_links = [Link.from_db(link) for link in node.Input or []]
        obj.output_links = [Link.from_db(link) for link in node.Output or []]
        if for_export:
            return obj.stripped_for_export()
        return obj

    def is_triggered_by_event_type(self, event_type: str) -> bool:
        return self.block.is_triggered_by_event_type(self.input_default, event_type)

    def stripped_for_export(self) -> "NodeModel":
        """
        Returns a copy of the node model, stripped of any non-transferable properties
        """
        stripped_node = self.model_copy(deep=True)

        # Remove credentials and other (possible) secrets from node input
        if stripped_node.input_default:
            stripped_node.input_default = NodeModel._filter_secrets_from_node_input(
                stripped_node.input_default, self.block.input_schema.jsonschema()
            )

        # Remove default secret value from secret input nodes
        if (
            stripped_node.block.block_type == BlockType.INPUT
            and stripped_node.input_default.get("secret", False) is True
            and "value" in stripped_node.input_default
        ):
            del stripped_node.input_default["value"]

        # Remove webhook info
        stripped_node.webhook_id = None

        return stripped_node

    @staticmethod
    def _filter_secrets_from_node_input(
        input_data: BlockInput, schema: dict[str, Any] | None
    ) -> BlockInput:
        sensitive_keys = ["credentials", "api_key", "password", "token", "secret"]
        field_schemas = schema.get("properties", {}) if schema else {}
        result = {}
        for key, value in input_data.items():
            field_schema: dict | None = field_schemas.get(key)
            if (field_schema and field_schema.get("secret", False)) or (
                any(sensitive_key in key.lower() for sensitive_key in sensitive_keys)
                # Prevent removing `secret` flag on input nodes
                and type(value) is not bool
            ):
                # This is a secret value -> filter this key-value pair out
                continue
            elif isinstance(value, dict):
                result[key] = NodeModel._filter_secrets_from_node_input(
                    value, field_schema
                )
            else:
                result[key] = value
        return result


class GraphBaseMeta(BaseDbModel):
    """
    Shared base for `GraphMeta` and `BaseGraph`, with core graph metadata fields.
    """

    version: int = 1
    is_active: bool = True
    name: str
    description: str
    instructions: str | None = None
    recommended_schedule_cron: str | None = None
    forked_from_id: str | None = None
    forked_from_version: int | None = None


class BaseGraph(GraphBaseMeta):
    """
    Graph with nodes, links, and computed I/O schema fields.

    Used to represent sub-graphs within a `Graph`. Contains the full graph
    structure including nodes and links, plus computed fields for schemas
    and trigger info. Does NOT include user_id or created_at (see GraphModel).
    """

    nodes: list[Node] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)

    @computed_field
    @property
    def input_schema(self) -> dict[str, Any]:
        return self._generate_schema(
            *(
                (block.input_schema, node.input_default)
                for node in self.nodes
                if (block := node.block).block_type == BlockType.INPUT
                and issubclass(block.input_schema, AgentInputBlock.Input)
            )
        )

    @computed_field
    @property
    def output_schema(self) -> dict[str, Any]:
        return self._generate_schema(
            *(
                (block.input_schema, node.input_default)
                for node in self.nodes
                if (block := node.block).block_type == BlockType.OUTPUT
                and issubclass(block.input_schema, AgentOutputBlock.Input)
            )
        )

    @computed_field
    @property
    def has_external_trigger(self) -> bool:
        return self.webhook_input_node is not None

    @computed_field
    @property
    def has_human_in_the_loop(self) -> bool:
        return any(
            node.block_id
            for node in self.nodes
            if node.block.block_type == BlockType.HUMAN_IN_THE_LOOP
        )

    @computed_field
    @property
    def has_sensitive_action(self) -> bool:
        return any(
            node.block_id for node in self.nodes if node.block.is_sensitive_action
        )

    @property
    def webhook_input_node(self) -> Node | None:
        return next(
            (
                node
                for node in self.nodes
                if node.block.block_type
                in (BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
            ),
            None,
        )

    @computed_field
    @property
    def trigger_setup_info(self) -> "GraphTriggerInfo | None":
        if not (
            self.webhook_input_node
            and (trigger_block := self.webhook_input_node.block).webhook_config
        ):
            return None

        return GraphTriggerInfo(
            provider=trigger_block.webhook_config.provider,
            config_schema={
                **(json_schema := trigger_block.input_schema.jsonschema()),
                "properties": {
                    pn: sub_schema
                    for pn, sub_schema in json_schema["properties"].items()
                    if not is_credentials_field_name(pn)
                },
                "required": [
                    pn
                    for pn in json_schema.get("required", [])
                    if not is_credentials_field_name(pn)
                ],
            },
            credentials_input_name=next(
                iter(trigger_block.input_schema.get_credentials_fields()), None
            ),
        )

    @staticmethod
    def _generate_schema(
        *props: tuple[type[AgentInputBlock.Input] | type[AgentOutputBlock.Input], dict],
    ) -> dict[str, Any]:
        schema_fields: list[AgentInputBlock.Input | AgentOutputBlock.Input] = []
        for type_class, input_default in props:
            try:
                schema_fields.append(type_class.model_construct(**input_default))
            except Exception as e:
                logger.error(f"Invalid {type_class}: {input_default}, {e}")

        return {
            "type": "object",
            "properties": {
                p.name: {
                    **{
                        k: v
                        for k, v in p.generate_schema().items()
                        if k not in ["description", "default"]
                    },
                    "secret": p.secret,
                    # Default value has to be set for advanced fields.
                    "advanced": p.advanced and p.value is not None,
                    "title": p.title or p.name,
                    **({"description": p.description} if p.description else {}),
                    **({"default": p.value} if p.value is not None else {}),
                }
                for p in schema_fields
            },
            "required": [p.name for p in schema_fields if p.value is None],
        }


class GraphTriggerInfo(BaseModel):
    provider: ProviderName
    config_schema: dict[str, Any] = Field(
        description="Input schema for the trigger block"
    )
    credentials_input_name: Optional[str]


class Graph(BaseGraph):
    """Creatable graph model used in API create/update endpoints."""

    sub_graphs: list[BaseGraph] = Field(default_factory=list)  # Flattened sub-graphs


class GraphMeta(GraphBaseMeta):
    """
    Lightweight graph metadata model representing an existing graph from the database,
    for use in listings and summaries.

    Lacks `GraphModel`'s nodes, links, and expensive computed fields.
    Use for list endpoints where full graph data is not needed and performance matters.
    """

    id: str  # type: ignore
    version: int  # type: ignore
    user_id: str
    created_at: datetime

    @classmethod
    def from_db(cls, graph: "AgentGraph") -> Self:
        return cls(
            id=graph.id,
            version=graph.version,
            is_active=graph.isActive,
            name=graph.name or "",
            description=graph.description or "",
            instructions=graph.instructions,
            recommended_schedule_cron=graph.recommendedScheduleCron,
            forked_from_id=graph.forkedFromId,
            forked_from_version=graph.forkedFromVersion,
            user_id=graph.userId,
            created_at=graph.createdAt,
        )


class GraphModel(Graph, GraphMeta):
    """
    Full graph model representing an existing graph from the database.

    This is the primary model for working with persisted graphs. Includes all
    graph data (nodes, links, sub_graphs) plus user ownership and timestamps.
    Provides computed fields (input_schema, output_schema, etc.) used during
    set-up (frontend) and execution (backend).

    Inherits from:
    - `Graph`: provides structure (nodes, links, sub_graphs) and computed schemas
    - `GraphMeta`: provides user_id, created_at for database records
    """

    nodes: list[NodeModel] = Field(default_factory=list)  # type: ignore

    @property
    def starting_nodes(self) -> list[NodeModel]:
        outbound_nodes = {link.sink_id for link in self.links}
        input_nodes = {
            node.id for node in self.nodes if node.block.block_type == BlockType.INPUT
        }
        return [
            node
            for node in self.nodes
            if node.id not in outbound_nodes or node.id in input_nodes
        ]

    @property
    def webhook_input_node(self) -> NodeModel | None:  # type: ignore
        return cast(NodeModel, super().webhook_input_node)

    @computed_field
    @property
    def credentials_input_schema(self) -> dict[str, Any]:
        graph_credentials_inputs = self.aggregate_credentials_inputs()

        logger.debug(
            f"Combined credentials input fields for graph #{self.id} ({self.name}): "
            f"{graph_credentials_inputs}"
        )

        # Warn if same-provider credentials inputs can't be combined (= bad UX)
        graph_cred_fields = list(graph_credentials_inputs.values())
        for i, (field, keys, _) in enumerate(graph_cred_fields):
            for other_field, other_keys, _ in list(graph_cred_fields)[i + 1 :]:
                if field.provider != other_field.provider:
                    continue
                if ProviderName.HTTP in field.provider:
                    continue
                # MCP credentials are intentionally split by server URL
                if ProviderName.MCP in field.provider:
                    continue

                # If this happens, that means a block implementation probably needs
                # to be updated.
                logger.warning(
                    "Multiple combined credentials fields "
                    f"for provider {field.provider} "
                    f"on graph #{self.id} ({self.name}); "
                    f"fields: {field} <> {other_field};"
                    f"keys: {keys} <> {other_keys}."
                )

        # Build JSON schema directly to avoid expensive create_model + validation overhead
        properties = {}
        required_fields = []

        for agg_field_key, (
            field_info,
            _,
            is_required,
        ) in graph_credentials_inputs.items():
            providers = list(field_info.provider)
            cred_types = list(field_info.supported_types)

            field_schema: dict[str, Any] = {
                "credentials_provider": providers,
                "credentials_types": cred_types,
                "type": "object",
                "properties": {
                    "id": {"title": "Id", "type": "string"},
                    "title": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "title": "Title",
                    },
                    "provider": {
                        "title": "Provider",
                        "type": "string",
                        **(
                            {"enum": providers}
                            if len(providers) > 1
                            else {"const": providers[0]}
                        ),
                    },
                    "type": {
                        "title": "Type",
                        "type": "string",
                        **(
                            {"enum": cred_types}
                            if len(cred_types) > 1
                            else {"const": cred_types[0]}
                        ),
                    },
                },
                "required": ["id", "provider", "type"],
            }

            # Add a descriptive display title when URL-based discriminator values
            # are present (e.g. "mcp.sentry.dev" instead of just "Mcp")
            if (
                field_info.discriminator
                and not field_info.discriminator_mapping
                and field_info.discriminator_values
            ):
                hostnames = sorted(
                    parse_url(str(v)).netloc for v in field_info.discriminator_values
                )
                field_schema["display_name"] = ", ".join(hostnames)

            # Add other (optional) field info items
            field_schema.update(
                field_info.model_dump(
                    by_alias=True,
                    exclude_defaults=True,
                    exclude={"provider", "supported_types"},  # already included above
                )
            )

            # Ensure field schema is well-formed
            CredentialsMetaInput.validate_credentials_field_schema(
                field_schema, agg_field_key
            )

            properties[agg_field_key] = field_schema
            if is_required:
                required_fields.append(agg_field_key)

        return {
            "type": "object",
            "properties": properties,
            "required": required_fields,
        }

    def aggregate_credentials_inputs(
        self,
    ) -> dict[str, tuple[CredentialsFieldInfo, set[tuple[str, str]], bool]]:
        """
        Returns:
            dict[aggregated_field_key, tuple(
                CredentialsFieldInfo: A spec for one aggregated credentials field
                    (now includes discriminator_values from matching nodes)
                set[(node_id, field_name)]: Node credentials fields that are
                    compatible with this aggregated field spec
                bool: True if the field is required (any node has credentials_optional=False)
            )]
        """
        # First collect all credential field data with input defaults
        # Track (field_info, (node_id, field_name), is_required) for each credential field
        node_credential_data: list[tuple[CredentialsFieldInfo, tuple[str, str]]] = []
        node_required_map: dict[str, bool] = {}  # node_id -> is_required

        for graph in [self] + self.sub_graphs:
            for node in graph.nodes:
                # A node's credentials are optional if either:
                # 1. The node metadata says so (credentials_optional=True), or
                # 2. All credential fields on the block have defaults (not required by schema)
                block_required = node.block.input_schema.get_required_fields()
                creds_required_by_schema = any(
                    fname in block_required
                    for fname in node.block.input_schema.get_credentials_fields()
                )
                node_required_map[node.id] = (
                    not node.credentials_optional and creds_required_by_schema
                )

                for (
                    field_name,
                    field_info,
                ) in node.block.input_schema.get_credentials_fields_info().items():

                    discriminator = field_info.discriminator
                    if not discriminator:
                        node_credential_data.append((field_info, (node.id, field_name)))
                        continue

                    discriminator_value = node.input_default.get(discriminator)
                    if discriminator_value is None:
                        node_credential_data.append((field_info, (node.id, field_name)))
                        continue

                    discriminated_info = field_info.discriminate(discriminator_value)
                    discriminated_info.discriminator_values.add(discriminator_value)

                    node_credential_data.append(
                        (discriminated_info, (node.id, field_name))
                    )

        # Combine credential field info (this will merge discriminator_values automatically)
        combined = CredentialsFieldInfo.combine(*node_credential_data)

        # Add is_required flag to each aggregated field
        # A field is required if ANY node using it has credentials_optional=False
        return {
            key: (
                field_info,
                node_field_pairs,
                any(
                    node_required_map.get(node_id, True)
                    for node_id, _ in node_field_pairs
                ),
            )
            for key, (field_info, node_field_pairs) in combined.items()
        }

    def reassign_ids(self, user_id: str, reassign_graph_id: bool = False):
        """
        Reassigns all IDs in the graph to new UUIDs.
        This method can be used before storing a new graph to the database.
        """
        if reassign_graph_id:
            graph_id_map = {
                self.id: str(uuid.uuid4()),
                **{sub_graph.id: str(uuid.uuid4()) for sub_graph in self.sub_graphs},
            }
        else:
            graph_id_map = {}

        self._reassign_ids(self, user_id, graph_id_map)
        for sub_graph in self.sub_graphs:
            self._reassign_ids(sub_graph, user_id, graph_id_map)

    @staticmethod
    def _reassign_ids(
        graph: BaseGraph,
        user_id: str,
        graph_id_map: dict[str, str],
    ):
        # Reassign Graph ID
        if graph.id in graph_id_map:
            graph.id = graph_id_map[graph.id]

        # Reassign Node IDs
        id_map = {node.id: str(uuid.uuid4()) for node in graph.nodes}
        for node in graph.nodes:
            node.id = id_map[node.id]

        # Reassign Link IDs
        for link in graph.links:
            if link.source_id in id_map:
                link.source_id = id_map[link.source_id]
            if link.sink_id in id_map:
                link.sink_id = id_map[link.sink_id]

        # Reassign User IDs for agent blocks
        for node in graph.nodes:
            if node.block_id != AgentExecutorBlock().id:
                continue
            node.input_default["user_id"] = user_id
            node.input_default.setdefault("inputs", {})
            if (
                graph_id := node.input_default.get("graph_id")
            ) and graph_id in graph_id_map:
                node.input_default["graph_id"] = graph_id_map[graph_id]

    def validate_graph(
        self,
        for_run: bool = False,
        nodes_input_masks: Optional["NodesInputMasks"] = None,
    ):
        """
        Validate graph structure and raise `ValueError` on issues.
        For structured error reporting, use `validate_graph_get_errors`.
        """
        self._validate_graph(self, for_run, nodes_input_masks)
        for sub_graph in self.sub_graphs:
            self._validate_graph(sub_graph, for_run, nodes_input_masks)

    @staticmethod
    def _validate_graph(
        graph: BaseGraph,
        for_run: bool = False,
        nodes_input_masks: Optional["NodesInputMasks"] = None,
    ) -> None:
        errors = GraphModel._validate_graph_get_errors(
            graph, for_run, nodes_input_masks
        )
        if errors:
            # Just raise the first error for backward compatibility
            first_error = next(iter(errors.values()))
            first_field_error = next(iter(first_error.values()))
            raise ValueError(first_field_error)

    def validate_graph_get_errors(
        self,
        for_run: bool = False,
        nodes_input_masks: Optional["NodesInputMasks"] = None,
    ) -> dict[str, dict[str, str]]:
        """
        Validate graph and return structured errors per node.

        Returns: dict[node_id, dict[field_name, error_message]]
        """
        return {
            **self._validate_graph_get_errors(self, for_run, nodes_input_masks),
            **{
                node_id: error
                for sub_graph in self.sub_graphs
                for node_id, error in self._validate_graph_get_errors(
                    sub_graph, for_run, nodes_input_masks
                ).items()
            },
        }

    @staticmethod
    def _validate_graph_get_errors(
        graph: BaseGraph,
        for_run: bool = False,
        nodes_input_masks: Optional["NodesInputMasks"] = None,
    ) -> dict[str, dict[str, str]]:
        """
        Validate graph and return structured errors per node.

        Returns: dict[node_id, dict[field_name, error_message]]
        """
        # First, check for structural issues with the graph
        try:
            GraphModel._validate_graph_structure(graph)
        except ValueError:
            # If structural validation fails, we can't provide per-node errors
            # so we re-raise as is
            raise

        # Collect errors per node
        node_errors: dict[str, dict[str, str]] = defaultdict(dict)

        # Validate smart decision maker nodes
        nodes_block = {
            node.id: block
            for node in graph.nodes
            if (block := get_block(node.block_id)) is not None
        }

        input_links: dict[str, list[Link]] = defaultdict(list)

        for link in graph.links:
            input_links[link.sink_id].append(link)

        # Nodes: required fields are filled or connected and dependencies are satisfied
        for node in graph.nodes:
            if (block := nodes_block.get(node.id)) is None:
                # For invalid blocks, we still raise immediately as this is a structural issue
                raise ValueError(f"Invalid block {node.block_id} for node #{node.id}")

            if block.disabled:
                raise ValueError(
                    f"Block {node.block_id} is disabled and cannot be used in graphs"
                )

            node_input_mask = (
                nodes_input_masks.get(node.id, {}) if nodes_input_masks else {}
            )
            provided_inputs = set(
                [sanitize_pin_name(name) for name in node.input_default]
                + [
                    sanitize_pin_name(link.sink_name)
                    for link in input_links.get(node.id, [])
                ]
                + ([name for name in node_input_mask] if node_input_mask else [])
            )
            InputSchema = block.input_schema

            for name in (required_fields := InputSchema.get_required_fields()):
                if (
                    name not in provided_inputs
                    # Checking availability of credentials is done by ExecutionManager
                    and name not in InputSchema.get_credentials_fields()
                    # Validate only I/O nodes, or validate everything when executing
                    and (
                        for_run
                        or block.block_type
                        in [
                            BlockType.INPUT,
                            BlockType.OUTPUT,
                            BlockType.AGENT,
                        ]
                    )
                ):
                    node_errors[node.id][name] = "This field is required"

                if (
                    block.block_type == BlockType.INPUT
                    and (input_key := node.input_default.get("name"))
                    and is_credentials_field_name(input_key)
                ):
                    node_errors[node.id]["name"] = (
                        f"'{input_key}' is a reserved input name: "
                        "'credentials' and `*_credentials` are reserved"
                    )

            # Check custom block-level validation (e.g., MCP dynamic tool arguments).
            # Blocks can override get_missing_input to report additional missing fields
            # beyond the standard top-level required fields.
            if for_run:
                credential_fields = InputSchema.get_credentials_fields()
                custom_missing = InputSchema.get_missing_input(node.input_default)
                for field_name in custom_missing:
                    if (
                        field_name not in provided_inputs
                        and field_name not in credential_fields
                    ):
                        node_errors[node.id][field_name] = "This field is required"

            # Get input schema properties and check dependencies
            input_fields = InputSchema.model_fields

            def has_value(node: Node, name: str):
                return (
                    (
                        name in node.input_default
                        and node.input_default[name] is not None
                        and str(node.input_default[name]).strip() != ""
                    )
                    or (name in input_fields and input_fields[name].default is not None)
                    or (
                        name in node_input_mask
                        and node_input_mask[name] is not None
                        and str(node_input_mask[name]).strip() != ""
                    )
                )

            # Validate dependencies between fields
            for field_name in input_fields.keys():
                field_json_schema = InputSchema.get_field_schema(field_name)

                dependencies: list[str] = []

                # Check regular field dependencies (only pre graph execution)
                if for_run:
                    dependencies.extend(field_json_schema.get("depends_on", []))

                # Require presence of credentials discriminator (always).
                # The `discriminator` is either the name of a sibling field (str),
                # or an object that discriminates between possible types for this field:
                # {"propertyName": prop_name, "mapping": {prop_value: sub_schema}}
                if (
                    discriminator := field_json_schema.get("discriminator")
                ) and isinstance(discriminator, str):
                    dependencies.append(discriminator)

                if not dependencies:
                    continue

                # Check if dependent field has value in input_default
                field_has_value = has_value(node, field_name)
                field_is_required = field_name in required_fields

                # Check for missing dependencies when dependent field is present
                missing_deps = [dep for dep in dependencies if not has_value(node, dep)]
                if missing_deps and (field_has_value or field_is_required):
                    node_errors[node.id][
                        field_name
                    ] = f"Requires {', '.join(missing_deps)} to be set"

        return node_errors

    @staticmethod
    def _validate_graph_structure(graph: BaseGraph):
        """Validate graph structure (links, connections, etc.)"""
        node_map = {v.id: v for v in graph.nodes}

        def is_static_output_block(nid: str) -> bool:
            return node_map[nid].block.static_output

        # Links: links are connected and the connected pin data type are compatible.
        for link in graph.links:
            source = (link.source_id, link.source_name)
            sink = (link.sink_id, link.sink_name)
            prefix = f"Link {source} <-> {sink}"

            for i, (node_id, name) in enumerate([source, sink]):
                node = node_map.get(node_id)
                if not node:
                    raise ValueError(
                        f"{prefix}, {node_id} is invalid node id, available nodes: {node_map.keys()}"
                    )

                block = get_block(node.block_id)
                if not block:
                    blocks = {v().id: v().name for v in get_blocks().values()}
                    raise ValueError(
                        f"{prefix}, {node.block_id} is invalid block id, available blocks: {blocks}"
                    )

                sanitized_name = sanitize_pin_name(name)
                vals = node.input_default
                if i == 0:
                    fields = (
                        block.output_schema.get_fields()
                        if block.block_type not in [BlockType.AGENT]
                        else vals.get("output_schema", {}).get("properties", {}).keys()
                    )
                else:
                    fields = (
                        block.input_schema.get_fields()
                        if block.block_type not in [BlockType.AGENT]
                        else vals.get("input_schema", {}).get("properties", {}).keys()
                    )
                if sanitized_name not in fields and not is_tool_pin(name):
                    fields_msg = f"Allowed fields: {fields}"
                    raise ValueError(f"{prefix}, `{name}` invalid, {fields_msg}")

            if is_static_output_block(link.source_id):
                link.is_static = True  # Each value block output should be static.

    @classmethod
    def from_db(  # type: ignore[reportIncompatibleMethodOverride]
        cls,
        graph: AgentGraph,
        for_export: bool = False,
        sub_graphs: list[AgentGraph] | None = None,
    ) -> Self:
        return cls(
            id=graph.id,
            user_id=graph.userId if not for_export else "",
            version=graph.version,
            forked_from_id=graph.forkedFromId,
            forked_from_version=graph.forkedFromVersion,
            created_at=graph.createdAt,
            is_active=graph.isActive,
            name=graph.name or "",
            description=graph.description or "",
            instructions=graph.instructions,
            recommended_schedule_cron=graph.recommendedScheduleCron,
            nodes=[NodeModel.from_db(node, for_export) for node in graph.Nodes or []],
            links=list(
                {
                    Link.from_db(link)
                    for node in graph.Nodes or []
                    for link in (node.Input or []) + (node.Output or [])
                }
            ),
            sub_graphs=[
                GraphModel.from_db(sub_graph, for_export)
                for sub_graph in sub_graphs or []
            ],
        )

    def hide_nodes(self) -> "GraphModelWithoutNodes":
        """
        Returns a copy of the `GraphModel` with nodes, links, and sub-graphs hidden
        (excluded from serialization). They are still present in the model instance
        so all computed fields (e.g. `credentials_input_schema`) still work.
        """
        return GraphModelWithoutNodes.model_validate(self, from_attributes=True)


class GraphModelWithoutNodes(GraphModel):
    """
    GraphModel variant that excludes nodes, links, and sub-graphs from serialization.

    Used in contexts like the store where exposing internal graph structure
    is not desired. Inherits all computed fields from GraphModel but marks
    nodes and links as excluded from JSON output.
    """

    nodes: list[NodeModel] = Field(default_factory=list, exclude=True)
    links: list[Link] = Field(default_factory=list, exclude=True)

    sub_graphs: list[BaseGraph] = Field(default_factory=list, exclude=True)


class GraphsPaginated(BaseModel):
    """Response schema for paginated graphs."""

    graphs: list[GraphMeta]
    pagination: Pagination


# --------------------- CRUD functions --------------------- #


async def get_node(node_id: str) -> NodeModel:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    node = await AgentNode.prisma().find_unique_or_raise(
        where={"id": node_id},
        include=AGENT_NODE_INCLUDE,
    )
    return NodeModel.from_db(node)


async def set_node_webhook(node_id: str, webhook_id: str | None) -> NodeModel:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    node = await AgentNode.prisma().update(
        where={"id": node_id},
        data=(
            {"Webhook": {"connect": {"id": webhook_id}}}
            if webhook_id
            else {"Webhook": {"disconnect": True}}
        ),
        include=AGENT_NODE_INCLUDE,
    )
    if not node:
        raise ValueError(f"Node #{node_id} not found")
    return NodeModel.from_db(node)


async def list_graphs_paginated(
    user_id: str,
    page: int = 1,
    page_size: int = 25,
    filter_by: Literal["active"] | None = "active",
) -> GraphsPaginated:
    """
    Retrieves paginated graph metadata objects.

    Args:
        user_id: The ID of the user that owns the graphs.
        page: Page number (1-based).
        page_size: Number of graphs per page.
        filter_by: An optional filter to either select graphs.

    Returns:
        GraphsPaginated: Paginated list of graph metadata.
    """
    where_clause: AgentGraphWhereInput = {"userId": user_id}

    if filter_by == "active":
        where_clause["isActive"] = True

    # Get total count
    total_count = await AgentGraph.prisma().count(where=where_clause)
    total_pages = (total_count + page_size - 1) // page_size

    # Get paginated results
    offset = (page - 1) * page_size
    graphs = await AgentGraph.prisma().find_many(
        where=where_clause,
        distinct=["id"],
        order={"version": "desc"},
        skip=offset,
        take=page_size,
    )

    graph_models = [GraphMeta.from_db(graph) for graph in graphs]

    return GraphsPaginated(
        graphs=graph_models,
        pagination=Pagination(
            total_items=total_count,
            total_pages=total_pages,
            current_page=page,
            page_size=page_size,
        ),
    )


async def get_graph_metadata(graph_id: str, version: int | None = None) -> Graph | None:
    where_clause: AgentGraphWhereInput = {
        "id": graph_id,
    }

    if version is not None:
        where_clause["version"] = version

    graph = await AgentGraph.prisma().find_first(
        where=where_clause,
        order={"version": "desc"},
    )

    if not graph:
        return None

    return Graph(
        id=graph.id,
        name=graph.name or "",
        description=graph.description or "",
        version=graph.version,
        is_active=graph.isActive,
    )


async def get_graph(
    graph_id: str,
    version: int | None,
    user_id: str | None,
    *,
    for_export: bool = False,
    include_subgraphs: bool = False,
    skip_access_check: bool = False,
) -> GraphModel | None:
    """
    Retrieves a graph from the DB.
    Defaults to the version with `is_active` if `version` is not passed.

    Returns `None` if the record is not found.
    """
    graph = None

    # Only search graph directly on owned graph (or access check is skipped)
    if skip_access_check or user_id is not None:
        graph_where_clause: AgentGraphWhereInput = {
            "id": graph_id,
        }
        if version is not None:
            graph_where_clause["version"] = version
        if not skip_access_check and user_id is not None:
            graph_where_clause["userId"] = user_id

        graph = await AgentGraph.prisma().find_first(
            where=graph_where_clause,
            include=AGENT_GRAPH_INCLUDE,
            order={"version": "desc"},
        )

    # Use store listed graph to find not owned graph
    if graph is None:
        store_where_clause: StoreListingVersionWhereInput = {
            "agentGraphId": graph_id,
            "submissionStatus": SubmissionStatus.APPROVED,
            "isDeleted": False,
        }
        if version is not None:
            store_where_clause["agentGraphVersion"] = version

        if store_listing := await StoreListingVersion.prisma().find_first(
            where=store_where_clause,
            order={"agentGraphVersion": "desc"},
            include={"AgentGraph": {"include": AGENT_GRAPH_INCLUDE}},
        ):
            graph = store_listing.AgentGraph

    if graph is None:
        return None

    if include_subgraphs or for_export:
        sub_graphs = await get_sub_graphs(graph)
        return GraphModel.from_db(
            graph=graph,
            sub_graphs=sub_graphs,
            for_export=for_export,
        )

    return GraphModel.from_db(graph, for_export)


async def get_store_listed_graphs(graph_ids: list[str]) -> dict[str, GraphModel]:
    """Batch-fetch multiple store-listed graphs by their IDs.

    Only returns graphs that have approved store listings (publicly available).
    Does not require permission checks since store-listed graphs are public.

    Args:
        graph_ids: List of graph IDs to fetch

    Returns:
        Dict mapping graph_id to GraphModel for graphs with approved store listings
    """
    if not graph_ids:
        return {}

    store_listings = await StoreListingVersion.prisma().find_many(
        where={
            "agentGraphId": {"in": list(graph_ids)},
            "submissionStatus": SubmissionStatus.APPROVED,
            "isDeleted": False,
        },
        include={"AgentGraph": {"include": AGENT_GRAPH_INCLUDE}},
        distinct=["agentGraphId"],
        order={"agentGraphVersion": "desc"},
    )

    return {
        listing.agentGraphId: GraphModel.from_db(listing.AgentGraph)
        for listing in store_listings
        if listing.AgentGraph
    }


async def get_graph_as_admin(
    graph_id: str,
    version: int | None = None,
    user_id: str | None = None,
    for_export: bool = False,
) -> GraphModel | None:
    """
    Intentionally parallels the get_graph but should only be used for admin tasks, because can return any graph that's been submitted
    Retrieves a graph from the DB.
    Defaults to the version with `is_active` if `version` is not passed.

    Returns `None` if the record is not found.
    """
    logger.warning(f"Getting {graph_id=} {version=} as ADMIN {user_id=} {for_export=}")
    where_clause: AgentGraphWhereInput = {
        "id": graph_id,
    }

    if version is not None:
        where_clause["version"] = version

    graph = await AgentGraph.prisma().find_first(
        where=where_clause,
        include=AGENT_GRAPH_INCLUDE,
        order={"version": "desc"},
    )

    # For access, the graph must be owned by the user or listed in the store
    if graph is None or (
        graph.userId != user_id
        and not await is_graph_published_in_marketplace(
            graph_id, version or graph.version
        )
    ):
        return None

    if for_export:
        sub_graphs = await get_sub_graphs(graph)
        return GraphModel.from_db(
            graph=graph,
            sub_graphs=sub_graphs,
            for_export=for_export,
        )

    return GraphModel.from_db(graph, for_export)


async def get_sub_graphs(graph: AgentGraph) -> list[AgentGraph]:
    """
    Iteratively fetches all sub-graphs of a given graph, and flattens them into a list.
    This call involves a DB fetch in batch, breadth-first, per-level of graph depth.
    On each DB fetch we will only fetch the sub-graphs that are not already in the list.
    """
    sub_graphs = {graph.id: graph}
    search_graphs = [graph]
    agent_block_id = AgentExecutorBlock().id

    while search_graphs:
        sub_graph_ids = [
            (graph_id, graph_version)
            for graph in search_graphs
            for node in graph.Nodes or []
            if (
                node.AgentBlock
                and node.AgentBlock.id == agent_block_id
                and (graph_id := cast(str, dict(node.constantInput).get("graph_id")))
                and (
                    graph_version := cast(
                        int, dict(node.constantInput).get("graph_version")
                    )
                )
            )
        ]
        if not sub_graph_ids:
            break

        graphs = await AgentGraph.prisma().find_many(
            where={
                "OR": [
                    {
                        "id": graph_id,
                        "version": graph_version,
                        "userId": graph.userId,  # Ensure the sub-graph is owned by the same user
                    }
                    for graph_id, graph_version in sub_graph_ids
                ]
            },
            include=AGENT_GRAPH_INCLUDE,
        )

        search_graphs = [graph for graph in graphs if graph.id not in sub_graphs]
        sub_graphs.update({graph.id: graph for graph in search_graphs})

    return [g for g in sub_graphs.values() if g.id != graph.id]


async def get_connected_output_nodes(node_id: str) -> list[tuple[Link, Node]]:
    links = await AgentNodeLink.prisma().find_many(
        where={"agentNodeSourceId": node_id},
        include={"AgentNodeSink": {"include": AGENT_NODE_INCLUDE}},
    )
    return [
        (Link.from_db(link), NodeModel.from_db(link.AgentNodeSink))
        for link in links
        if link.AgentNodeSink
    ]


async def set_graph_active_version(graph_id: str, version: int, user_id: str) -> None:
    # Activate the requested version if it exists and is owned by the user.
    updated_count = await AgentGraph.prisma().update_many(
        data={"isActive": True},
        where={
            "id": graph_id,
            "version": version,
            "userId": user_id,
        },
    )
    if updated_count == 0:
        raise Exception(f"Graph #{graph_id} v{version} not found or not owned by user")

    # Deactivate all other versions.
    await AgentGraph.prisma().update_many(
        data={"isActive": False},
        where={
            "id": graph_id,
            "version": {"not": version},
            "userId": user_id,
            "isActive": True,
        },
    )


async def get_graph_all_versions(
    graph_id: str, user_id: str, limit: int = MAX_GRAPH_VERSIONS_FETCH
) -> list[GraphModel]:
    graph_versions = await AgentGraph.prisma().find_many(
        where={"id": graph_id, "userId": user_id},
        order={"version": "desc"},
        include=AGENT_GRAPH_INCLUDE,
        take=limit,
    )

    if not graph_versions:
        return []

    return [GraphModel.from_db(graph) for graph in graph_versions]


async def delete_graph(graph_id: str, user_id: str) -> int:
    entries_count = await AgentGraph.prisma().delete_many(
        where={"id": graph_id, "userId": user_id}
    )
    if entries_count:
        logger.info(f"Deleted {entries_count} graph entries for Graph #{graph_id}")
    return entries_count


async def get_graph_settings(user_id: str, graph_id: str) -> GraphSettings:
    lib = await LibraryAgent.prisma().find_first(
        where={
            "userId": user_id,
            "agentGraphId": graph_id,
            "isDeleted": False,
            "isArchived": False,
        },
        order={"agentGraphVersion": "desc"},
    )
    if not lib or not lib.settings:
        return GraphSettings()

    try:
        return GraphSettings.model_validate(lib.settings)
    except Exception:
        logger.warning(
            f"Malformed settings for LibraryAgent user={user_id} graph={graph_id}"
        )
        return GraphSettings()


async def validate_graph_execution_permissions(
    user_id: str, graph_id: str, graph_version: int, is_sub_graph: bool = False
) -> None:
    """
    Validate that a user has permission to execute a specific graph.

    This function performs comprehensive authorization checks and raises specific
    exceptions for different types of failures to enable appropriate error handling.

    ## Logic
    A user can execute a graph if any of these is true:
    1. They own the graph and some version of it is still listed in their library
    2. The graph is published in the marketplace and listed in their library
    3. The graph is published in the marketplace and is being executed as a sub-agent

    Args:
        graph_id: The ID of the graph to check
        user_id: The ID of the user
        graph_version: The version of the graph to check
        is_sub_graph: Whether this is being executed as a sub-graph.
            If `True`, the graph isn't required to be in the user's Library.

    Raises:
        GraphNotAccessibleError: If the graph is not accessible to the user.
        GraphNotInLibraryError: If the graph is not in the user's library (deleted/archived).
        NotAuthorizedError: If the user lacks execution permissions for other reasons
    """
    graph, library_agent = await asyncio.gather(
        AgentGraph.prisma().find_unique(
            where={"graphVersionId": {"id": graph_id, "version": graph_version}}
        ),
        LibraryAgent.prisma().find_first(
            where={
                "userId": user_id,
                "agentGraphId": graph_id,
                "isDeleted": False,
                "isArchived": False,
            }
        ),
    )

    # Step 1: Check if user owns this graph
    user_owns_graph = graph and graph.userId == user_id

    # Step 2: Check if agent is in the library *and not deleted*
    user_has_in_library = library_agent is not None

    # Step 3: Apply permission logic
    if not (
        user_owns_graph
        or await is_graph_published_in_marketplace(graph_id, graph_version)
    ):
        raise GraphNotAccessibleError(
            f"You do not have access to graph #{graph_id} v{graph_version}: "
            "it is not owned by you and not available in the Marketplace"
        )
    elif not (user_has_in_library or is_sub_graph):
        raise GraphNotInLibraryError(f"Graph #{graph_id} is not in your library")

    # Step 6: Check execution-specific permissions (raises generic NotAuthorizedError)
    # Additional authorization checks beyond the above:
    # 1. Check if user has execution credits (future)
    # 2. Check if graph is suspended/disabled (future)
    # 3. Check rate limiting rules (future)
    # 4. Check organization-level permissions (future)

    # For now, the above check logic is sufficient for execution permission.
    # Future enhancements can add more granular permission checks here.
    # When adding new checks, raise NotAuthorizedError for non-library issues.


async def is_graph_published_in_marketplace(graph_id: str, graph_version: int) -> bool:
    """
    Check if a graph is published in the marketplace.

    Params:
        graph_id: The ID of the graph to check
        graph_version: The version of the graph to check

    Returns:
        True if the graph is published and approved in the marketplace, False otherwise
    """
    marketplace_listing = await StoreListingVersion.prisma().find_first(
        where={
            "agentGraphId": graph_id,
            "agentGraphVersion": graph_version,
            "submissionStatus": SubmissionStatus.APPROVED,
            "isDeleted": False,
        }
    )
    return marketplace_listing is not None


async def create_graph(graph: Graph, user_id: str) -> GraphModel:
    async with transaction() as tx:
        await __create_graph(tx, graph, user_id)

    if created_graph := await get_graph(graph.id, graph.version, user_id=user_id):
        return created_graph

    raise ValueError(f"Created graph {graph.id} v{graph.version} is not in DB")


async def fork_graph(graph_id: str, graph_version: int, user_id: str) -> GraphModel:
    """
    Forks a graph by copying it and all its nodes and links to a new graph.
    """
    graph = await get_graph(graph_id, graph_version, user_id=user_id, for_export=True)
    if not graph:
        raise ValueError(f"Graph {graph_id} v{graph_version} not found")

    # Set forked from ID and version as itself as it's about ot be copied
    graph.forked_from_id = graph.id
    graph.forked_from_version = graph.version
    graph.name = f"{graph.name} (copy)"
    graph.reassign_ids(user_id=user_id, reassign_graph_id=True)
    graph.validate_graph(for_run=False)

    async with transaction() as tx:
        await __create_graph(tx, graph, user_id)

    return graph


async def __create_graph(tx, graph: Graph, user_id: str):
    graphs = [graph] + graph.sub_graphs

    await AgentGraph.prisma(tx).create_many(
        data=[
            AgentGraphCreateInput(
                id=graph.id,
                version=graph.version,
                name=graph.name,
                description=graph.description,
                recommendedScheduleCron=graph.recommended_schedule_cron,
                isActive=graph.is_active,
                userId=user_id,
                forkedFromId=graph.forked_from_id,
                forkedFromVersion=graph.forked_from_version,
            )
            for graph in graphs
        ]
    )

    await AgentNode.prisma(tx).create_many(
        data=[
            AgentNodeCreateInput(
                id=node.id,
                agentGraphId=graph.id,
                agentGraphVersion=graph.version,
                agentBlockId=node.block_id,
                constantInput=SafeJson(node.input_default),
                metadata=SafeJson(node.metadata),
            )
            for graph in graphs
            for node in graph.nodes
        ]
    )

    await AgentNodeLink.prisma(tx).create_many(
        data=[
            AgentNodeLinkCreateInput(
                id=str(uuid.uuid4()),
                sourceName=link.source_name,
                sinkName=link.sink_name,
                agentNodeSourceId=link.source_id,
                agentNodeSinkId=link.sink_id,
                isStatic=link.is_static,
            )
            for graph in graphs
            for link in graph.links
        ]
    )


# ------------------------ UTILITIES ------------------------ #


def make_graph_model(creatable_graph: Graph, user_id: str) -> GraphModel:
    """
    Convert a Graph to a GraphModel, setting graph_id and graph_version on all nodes.

    Args:
        creatable_graph (Graph): The creatable graph to convert.
        user_id (str): The ID of the user creating the graph.

    Returns:
        GraphModel: The converted Graph object.
    """
    # Create a new Graph object, inheriting properties from CreatableGraph
    return GraphModel(
        **creatable_graph.model_dump(exclude={"nodes"}),
        user_id=user_id,
        created_at=datetime.now(tz=timezone.utc),
        nodes=[
            NodeModel(
                **creatable_node.model_dump(),
                graph_id=creatable_graph.id,
                graph_version=creatable_graph.version,
            )
            for creatable_node in creatable_graph.nodes
        ],
    )


async def fix_llm_provider_credentials():
    """Fix node credentials with provider `llm`"""
    from backend.integrations.credentials_store import IntegrationCredentialsStore

    from .user import get_user_integrations

    store = IntegrationCredentialsStore()

    broken_nodes = []
    try:
        broken_nodes = await query_raw_with_schema(
            """
            SELECT    graph."userId"       user_id,
                  node.id              node_id,
                  node."constantInput" node_preset_input
            FROM      {schema_prefix}"AgentNode"  node
            LEFT JOIN {schema_prefix}"AgentGraph" graph
            ON        node."agentGraphId" = graph.id
            WHERE     node."constantInput"::jsonb->'credentials'->>'provider' = 'llm'
            ORDER BY  graph."userId";
            """
        )
        logger.info(f"Fixing LLM credential inputs on {len(broken_nodes)} nodes")
    except Exception as e:
        logger.error(f"Error fixing LLM credential inputs: {e}")

    user_id: str = ""
    user_integrations = None
    for node in broken_nodes:
        if node["user_id"] != user_id:
            # Save queries by only fetching once per user
            user_id = node["user_id"]
            user_integrations = await get_user_integrations(user_id)
        elif not user_integrations:
            raise RuntimeError(f"Impossible state while processing node {node}")

        node_id: str = node["node_id"]
        node_preset_input: dict = node["node_preset_input"]
        credentials_meta: dict = node_preset_input["credentials"]

        credentials = next(
            (
                c
                for c in user_integrations.credentials
                if c.id == credentials_meta["id"]
            ),
            None,
        )
        if not credentials:
            continue
        if credentials.type != "api_key":
            logger.warning(
                f"User {user_id} credentials {credentials.id} with provider 'llm' "
                f"has invalid type '{credentials.type}'"
            )
            continue

        api_key = credentials.api_key.get_secret_value()
        if api_key.startswith("sk-ant-api03-"):
            credentials.provider = credentials_meta["provider"] = "anthropic"
        elif api_key.startswith("sk-"):
            credentials.provider = credentials_meta["provider"] = "openai"
        elif api_key.startswith("gsk_"):
            credentials.provider = credentials_meta["provider"] = "groq"
        else:
            logger.warning(
                f"Could not identify provider from key prefix {api_key[:13]}*****"
            )
            continue

        await store.update_creds(user_id, credentials)
        await AgentNode.prisma().update(
            where={"id": node_id},
            data={"constantInput": SafeJson(node_preset_input)},
        )


async def migrate_llm_models(migrate_to: LlmModel):
    """
    Update all LLM models in all AI blocks that don't exist in the enum.
    Note: Only updates top level LlmModel SchemaFields of blocks (won't update nested fields).
    """
    logger.info("Migrating LLM models")
    # Scan all blocks and search for LlmModel fields
    llm_model_fields: dict[str, str] = {}  # {block_id: field_name}

    # Search for all LlmModel fields
    for block_type in get_blocks().values():
        block = block_type()
        from pydantic.fields import FieldInfo

        fields: dict[str, FieldInfo] = block.input_schema.model_fields

        # Collect top-level LlmModel fields
        for field_name, field in fields.items():
            if field.annotation == LlmModel:
                llm_model_fields[block.id] = field_name

    # Convert enum values to a list of strings for the SQL query
    enum_values = [v.value for v in LlmModel]
    escaped_enum_values = repr(tuple(enum_values))  # hack but works

    # Update each block
    for id, path in llm_model_fields.items():
        query = f"""
            UPDATE platform."AgentNode"
            SET "constantInput" = jsonb_set("constantInput", $1, to_jsonb($2), true)
            WHERE "agentBlockId" = $3
            AND "constantInput" ? ($4)::text
            AND "constantInput"->>($4)::text NOT IN {escaped_enum_values}
            """

        await db.execute_raw(
            query,  # type: ignore - is supposed to be LiteralString
            [path],
            migrate_to.value,
            id,
            path,
        )


# Simple placeholder class for deleted/missing blocks
class _UnknownBlockBase(Block):
    """
    Placeholder for deleted/missing blocks that inherits from Block
    but uses a name that doesn't end with 'Block' to avoid auto-discovery.
    """

    def __init__(self, block_id: str = "00000000-0000-0000-0000-000000000000"):
        # Initialize with minimal valid Block parameters
        super().__init__(
            id=block_id,
            description=f"Unknown or deleted block (original ID: {block_id})",
            disabled=True,
            input_schema=EmptySchema,
            output_schema=EmptySchema,
            categories=set(),
            contributors=[],
            static_output=False,
            block_type=BlockType.STANDARD,
            webhook_config=None,
        )

    @property
    def name(self):
        return "UnknownBlock"

    async def run(self, input_data, **kwargs):
        """Always yield an error for missing blocks."""
        yield "error", f"Block {self.id} no longer exists"
