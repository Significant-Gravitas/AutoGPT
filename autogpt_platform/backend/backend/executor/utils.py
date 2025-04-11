import logging
from typing import TYPE_CHECKING, Any, cast

from autogpt_libs.utils.cache import thread_cached
from pydantic import BaseModel

from backend.data.block import (
    Block,
    BlockData,
    BlockInput,
    BlockSchema,
    BlockType,
    get_block,
)
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.cost import BlockCostType
from backend.data.execution import GraphExecutionEntry, RedisExecutionEventBus
from backend.data.graph import GraphModel, Node
from backend.data.rabbitmq import (
    Exchange,
    ExchangeType,
    Queue,
    RabbitMQConfig,
    SyncRabbitMQ,
)
from backend.util.mock import MockObject
from backend.util.service import get_service_client
from backend.util.settings import Config
from backend.util.type import convert

if TYPE_CHECKING:
    from backend.executor import DatabaseManager
    from backend.integrations.credentials_store import IntegrationCredentialsStore

config = Config()
logger = logging.getLogger(__name__)

# ============ Resource Helpers ============ #


@thread_cached
def get_execution_event_bus() -> RedisExecutionEventBus:
    return RedisExecutionEventBus()


@thread_cached
def get_execution_queue() -> SyncRabbitMQ:
    client = SyncRabbitMQ(create_execution_queue_config())
    client.connect()
    return client


@thread_cached
def get_integration_credentials_store() -> "IntegrationCredentialsStore":
    from backend.integrations.credentials_store import IntegrationCredentialsStore

    return IntegrationCredentialsStore()


@thread_cached
def get_db_client() -> "DatabaseManager":
    from backend.executor import DatabaseManager

    return get_service_client(DatabaseManager)


# ============ Execution Cost Helpers ============ #


class UsageTransactionMetadata(BaseModel):
    graph_exec_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    node_exec_id: str | None = None
    block_id: str | None = None
    block: str | None = None
    input: BlockInput | None = None


def execution_usage_cost(execution_count: int) -> tuple[int, int]:
    """
    Calculate the cost of executing a graph based on the number of executions.

    Args:
        execution_count: Number of executions

    Returns:
        Tuple of cost amount and remaining execution count
    """
    return (
        execution_count
        // config.execution_cost_count_threshold
        * config.execution_cost_per_threshold,
        execution_count % config.execution_cost_count_threshold,
    )


def block_usage_cost(
    block: Block,
    input_data: BlockInput,
    data_size: float = 0,
    run_time: float = 0,
) -> tuple[int, BlockInput]:
    """
    Calculate the cost of using a block based on the input data and the block type.

    Args:
        block: Block object
        input_data: Input data for the block
        data_size: Size of the input data in bytes
        run_time: Execution time of the block in seconds

    Returns:
        Tuple of cost amount and cost filter
    """
    block_costs = BLOCK_COSTS.get(type(block))
    if not block_costs:
        return 0, {}

    for block_cost in block_costs:
        if not _is_cost_filter_match(block_cost.cost_filter, input_data):
            continue

        if block_cost.cost_type == BlockCostType.RUN:
            return block_cost.cost_amount, block_cost.cost_filter

        if block_cost.cost_type == BlockCostType.SECOND:
            return (
                int(run_time * block_cost.cost_amount),
                block_cost.cost_filter,
            )

        if block_cost.cost_type == BlockCostType.BYTE:
            return (
                int(data_size * block_cost.cost_amount),
                block_cost.cost_filter,
            )

    return 0, {}


def _is_cost_filter_match(cost_filter: BlockInput, input_data: BlockInput) -> bool:
    """
    Filter rules:
      - If cost_filter is an object, then check if cost_filter is the subset of input_data
      - Otherwise, check if cost_filter is equal to input_data.
      - Undefined, null, and empty string are considered as equal.
    """
    if not isinstance(cost_filter, dict) or not isinstance(input_data, dict):
        return cost_filter == input_data

    return all(
        (not input_data.get(k) and not v)
        or (input_data.get(k) and _is_cost_filter_match(v, input_data[k]))
        for k, v in cost_filter.items()
    )


# ============ Execution Input Helpers ============ #

LIST_SPLIT = "_$_"
DICT_SPLIT = "_#_"
OBJC_SPLIT = "_@_"


def parse_execution_output(output: BlockData, name: str) -> Any | None:
    """
    Extracts partial output data by name from a given BlockData.

    The function supports extracting data from lists, dictionaries, and objects
    using specific naming conventions:
    - For lists: <output_name>_$_<index>
    - For dictionaries: <output_name>_#_<key>
    - For objects: <output_name>_@_<attribute>

    Args:
        output (BlockData): A tuple containing the output name and data.
        name (str): The name used to extract specific data from the output.

    Returns:
        Any | None: The extracted data if found, otherwise None.

    Examples:
        >>> output = ("result", [10, 20, 30])
        >>> parse_execution_output(output, "result_$_1")
        20

        >>> output = ("config", {"key1": "value1", "key2": "value2"})
        >>> parse_execution_output(output, "config_#_key1")
        'value1'

        >>> class Sample:
        ...     attr1 = "value1"
        ...     attr2 = "value2"
        >>> output = ("object", Sample())
        >>> parse_execution_output(output, "object_@_attr1")
        'value1'
    """
    output_name, output_data = output

    if name == output_name:
        return output_data

    if name.startswith(f"{output_name}{LIST_SPLIT}"):
        index = int(name.split(LIST_SPLIT)[1])
        if not isinstance(output_data, list) or len(output_data) <= index:
            return None
        return output_data[int(name.split(LIST_SPLIT)[1])]

    if name.startswith(f"{output_name}{DICT_SPLIT}"):
        index = name.split(DICT_SPLIT)[1]
        if not isinstance(output_data, dict) or index not in output_data:
            return None
        return output_data[index]

    if name.startswith(f"{output_name}{OBJC_SPLIT}"):
        index = name.split(OBJC_SPLIT)[1]
        if isinstance(output_data, object) and hasattr(output_data, index):
            return getattr(output_data, index)
        return None

    return None


def validate_exec(
    node: Node,
    data: BlockInput,
    resolve_input: bool = True,
) -> tuple[BlockInput | None, str]:
    """
    Validate the input data for a node execution.

    Args:
        node: The node to execute.
        data: The input data for the node execution.
        resolve_input: Whether to resolve dynamic pins into dict/list/object.

    Returns:
        A tuple of the validated data and the block name.
        If the data is invalid, the first element will be None, and the second element
        will be an error message.
        If the data is valid, the first element will be the resolved input data, and
        the second element will be the block name.
    """
    node_block: Block | None = get_block(node.block_id)
    if not node_block:
        return None, f"Block for {node.block_id} not found."
    schema = node_block.input_schema

    # Convert non-matching data types to the expected input schema.
    for name, data_type in schema.__annotations__.items():
        if (value := data.get(name)) and (type(value) is not data_type):
            data[name] = convert(value, data_type)

    # Input data (without default values) should contain all required fields.
    error_prefix = f"Input data missing or mismatch for `{node_block.name}`:"
    if missing_links := schema.get_missing_links(data, node.input_links):
        return None, f"{error_prefix} unpopulated links {missing_links}"

    # Merge input data with default values and resolve dynamic dict/list/object pins.
    input_default = schema.get_input_defaults(node.input_default)
    data = {**input_default, **data}
    if resolve_input:
        data = merge_execution_input(data)

    # Input data post-merge should contain all required fields from the schema.
    if missing_input := schema.get_missing_input(data):
        return None, f"{error_prefix} missing input {missing_input}"

    # Last validation: Validate the input values against the schema.
    if error := schema.get_mismatch_error(data):
        error_message = f"{error_prefix} {error}"
        logger.error(error_message)
        return None, error_message

    return data, node_block.name


def merge_execution_input(data: BlockInput) -> BlockInput:
    """
    Merges dynamic input pins into a single list, dictionary, or object based on naming patterns.

    This function processes input keys that follow specific patterns to merge them into a unified structure:
    - `<input_name>_$_<index>` for list inputs.
    - `<input_name>_#_<index>` for dictionary inputs.
    - `<input_name>_@_<index>` for object inputs.

    Args:
        data (BlockInput): A dictionary containing input keys and their corresponding values.

    Returns:
        BlockInput: A dictionary with merged inputs.

    Raises:
        ValueError: If a list index is not an integer.

    Examples:
        >>> data = {
        ...     "list_$_0": "a",
        ...     "list_$_1": "b",
        ...     "dict_#_key1": "value1",
        ...     "dict_#_key2": "value2",
        ...     "object_@_attr1": "value1",
        ...     "object_@_attr2": "value2"
        ... }
        >>> merge_execution_input(data)
        {
            "list": ["a", "b"],
            "dict": {"key1": "value1", "key2": "value2"},
            "object": <MockObject attr1="value1" attr2="value2">
        }
    """

    # Merge all input with <input_name>_$_<index> into a single list.
    items = list(data.items())

    for key, value in items:
        if LIST_SPLIT not in key:
            continue
        name, index = key.split(LIST_SPLIT)
        if not index.isdigit():
            raise ValueError(f"Invalid key: {key}, #{index} index must be an integer.")

        data[name] = data.get(name, [])
        if int(index) >= len(data[name]):
            # Pad list with empty string on missing indices.
            data[name].extend([""] * (int(index) - len(data[name]) + 1))
        data[name][int(index)] = value

    # Merge all input with <input_name>_#_<index> into a single dict.
    for key, value in items:
        if DICT_SPLIT not in key:
            continue
        name, index = key.split(DICT_SPLIT)
        data[name] = data.get(name, {})
        data[name][index] = value

    # Merge all input with <input_name>_@_<index> into a single object.
    for key, value in items:
        if OBJC_SPLIT not in key:
            continue
        name, index = key.split(OBJC_SPLIT)
        if name not in data or not isinstance(data[name], object):
            data[name] = MockObject()
        setattr(data[name], index, value)

    return data


def _validate_node_input_credentials(graph: GraphModel, user_id: str):
    """Checks all credentials for all nodes of the graph"""

    for node in graph.nodes:
        block = node.block

        # Find any fields of type CredentialsMetaInput
        credentials_fields = cast(
            type[BlockSchema], block.input_schema
        ).get_credentials_fields()
        if not credentials_fields:
            continue

        for field_name, credentials_meta_type in credentials_fields.items():
            credentials_meta = credentials_meta_type.model_validate(
                node.input_default[field_name]
            )
            # Fetch the corresponding Credentials and perform sanity checks
            credentials = get_integration_credentials_store().get_creds_by_id(
                user_id, credentials_meta.id
            )
            if not credentials:
                raise ValueError(
                    f"Unknown credentials #{credentials_meta.id} "
                    f"for node #{node.id} input '{field_name}'"
                )
            if (
                credentials.provider != credentials_meta.provider
                or credentials.type != credentials_meta.type
            ):
                logger.warning(
                    f"Invalid credentials #{credentials.id} for node #{node.id}: "
                    "type/provider mismatch: "
                    f"{credentials_meta.type}<>{credentials.type};"
                    f"{credentials_meta.provider}<>{credentials.provider}"
                )
                raise ValueError(
                    f"Invalid credentials #{credentials.id} for node #{node.id}: "
                    "type/provider mismatch"
                )


def construct_node_execution_input(
    graph: GraphModel,
    user_id: str,
    data: BlockInput,
) -> list[tuple[str, BlockInput]]:
    """
    Validates and prepares the input data for executing a graph.
    This function checks the graph for starting nodes, validates the input data
    against the schema, and resolves dynamic input pins into a single list,
    dictionary, or object.

    Args:
        graph (GraphModel): The graph model to execute.
        user_id (str): The ID of the user executing the graph.
        data (BlockInput): The input data for the graph execution.

    Returns:
        list[tuple[str, BlockInput]]: A list of tuples, each containing the node ID and
            the corresponding input data for that node.
    """
    graph.validate_graph(for_run=True)
    _validate_node_input_credentials(graph, user_id)

    nodes_input = []
    for node in graph.starting_nodes:
        input_data = {}
        block = node.block

        # Note block should never be executed.
        if block.block_type == BlockType.NOTE:
            continue

        # Extract request input data, and assign it to the input pin.
        if block.block_type == BlockType.INPUT:
            input_name = node.input_default.get("name")
            if input_name and input_name in data:
                input_data = {"value": data[input_name]}

        # Extract webhook payload, and assign it to the input pin
        webhook_payload_key = f"webhook_{node.webhook_id}_payload"
        if (
            block.block_type in (BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
            and node.webhook_id
        ):
            if webhook_payload_key not in data:
                raise ValueError(
                    f"Node {block.name} #{node.id} webhook payload is missing"
                )
            input_data = {"payload": data[webhook_payload_key]}

        input_data, error = validate_exec(node, input_data)
        if input_data is None:
            raise ValueError(error)
        else:
            nodes_input.append((node.id, input_data))

    if not nodes_input:
        raise ValueError(
            "No starting nodes found for the graph, make sure an AgentInput or blocks with no inbound links are present as starting nodes."
        )

    return nodes_input


# ============ Execution Queue Helpers ============ #


class CancelExecutionEvent(BaseModel):
    graph_exec_id: str


GRAPH_EXECUTION_EXCHANGE = Exchange(
    name="graph_execution",
    type=ExchangeType.DIRECT,
    durable=True,
    auto_delete=False,
)
GRAPH_EXECUTION_QUEUE_NAME = "graph_execution_queue"
GRAPH_EXECUTION_ROUTING_KEY = "graph_execution.run"

GRAPH_EXECUTION_CANCEL_EXCHANGE = Exchange(
    name="graph_execution_cancel",
    type=ExchangeType.FANOUT,
    durable=True,
    auto_delete=True,
)
GRAPH_EXECUTION_CANCEL_QUEUE_NAME = "graph_execution_cancel_queue"


def create_execution_queue_config() -> RabbitMQConfig:
    """
    Define two exchanges and queues:
    - 'graph_execution' (DIRECT) for run tasks.
    - 'graph_execution_cancel' (FANOUT) for cancel requests.
    """
    run_queue = Queue(
        name=GRAPH_EXECUTION_QUEUE_NAME,
        exchange=GRAPH_EXECUTION_EXCHANGE,
        routing_key=GRAPH_EXECUTION_ROUTING_KEY,
        durable=True,
        auto_delete=False,
    )
    cancel_queue = Queue(
        name=GRAPH_EXECUTION_CANCEL_QUEUE_NAME,
        exchange=GRAPH_EXECUTION_CANCEL_EXCHANGE,
        routing_key="",  # not used for FANOUT
        durable=True,
        auto_delete=False,
    )
    return RabbitMQConfig(
        vhost="/",
        exchanges=[GRAPH_EXECUTION_EXCHANGE, GRAPH_EXECUTION_CANCEL_EXCHANGE],
        queues=[run_queue, cancel_queue],
    )


def add_graph_execution(
    graph_id: str,
    data: BlockInput,
    user_id: str,
    graph_version: int | None = None,
    preset_id: str | None = None,
) -> GraphExecutionEntry:
    """
    Adds a graph execution to the queue and returns the execution entry.

    Args:
        graph_id (str): The ID of the graph to execute.
        data (BlockInput): The input data for the graph execution.
        user_id (str): The ID of the user executing the graph.
        graph_version (int | None): The version of the graph to execute. Defaults to None.
        preset_id (str | None): The ID of the preset to use. Defaults to None.
    Returns:
        GraphExecutionEntry: The entry for the graph execution.
    Raises:
        ValueError: If the graph is not found or if there are validation errors.
    """
    graph: GraphModel | None = get_db_client().get_graph(
        graph_id=graph_id, user_id=user_id, version=graph_version
    )
    if not graph:
        raise ValueError(f"Graph #{graph_id} not found.")

    graph_exec = get_db_client().create_graph_execution(
        graph_id=graph_id,
        graph_version=graph.version,
        nodes_input=construct_node_execution_input(graph, user_id, data),
        user_id=user_id,
        preset_id=preset_id,
    )
    get_execution_event_bus().publish(graph_exec)

    graph_exec_entry = graph_exec.to_graph_execution_entry()
    get_execution_queue().publish_message(
        routing_key=GRAPH_EXECUTION_ROUTING_KEY,
        message=graph_exec_entry.model_dump_json(),
        exchange=GRAPH_EXECUTION_EXCHANGE,
    )

    return graph_exec_entry
