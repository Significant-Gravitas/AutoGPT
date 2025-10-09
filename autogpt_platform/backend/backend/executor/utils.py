import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from typing import Mapping, Optional, cast

from pydantic import BaseModel, JsonValue, ValidationError

from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.block import (
    Block,
    BlockCostType,
    BlockInput,
    BlockOutputEntry,
    BlockType,
    get_block,
)
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.db import prisma

# Import dynamic field utilities from centralized location
from backend.data.dynamic_fields import merge_execution_input
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionStats,
    GraphExecutionWithNodes,
    NodesInputMasks,
    UserContext,
)
from backend.data.graph import GraphModel, Node
from backend.data.model import CredentialsMetaInput
from backend.data.rabbitmq import Exchange, ExchangeType, Queue, RabbitMQConfig
from backend.data.user import get_user_by_id
from backend.util.clients import (
    get_async_execution_event_bus,
    get_async_execution_queue,
    get_database_manager_async_client,
    get_integration_credentials_store,
)
from backend.util.exceptions import GraphValidationError, NotFoundError
from backend.util.logging import TruncatedLogger
from backend.util.settings import Config
from backend.util.type import convert


async def get_user_context(user_id: str) -> UserContext:
    """
    Get UserContext for a user, always returns a valid context with timezone.
    Defaults to UTC if user has no timezone set.
    """
    user_context = UserContext(timezone="UTC")  # Default to UTC
    try:
        user = await get_user_by_id(user_id)
        if user and user.timezone and user.timezone != "not-set":
            user_context.timezone = user.timezone
            logger.debug(f"Retrieved user context: timezone={user.timezone}")
        else:
            logger.debug("User has no timezone set, using UTC")
    except Exception as e:
        logger.warning(f"Could not fetch user timezone: {e}")
        # Continue with UTC as default

    return user_context


config = Config()
logger = TruncatedLogger(logging.getLogger(__name__), prefix="[GraphExecutorUtil]")

# ============ Resource Helpers ============ #


class LogMetadata(TruncatedLogger):
    def __init__(
        self,
        logger: logging.Logger,
        user_id: str,
        graph_eid: str,
        graph_id: str,
        node_eid: str,
        node_id: str,
        block_name: str,
        max_length: int = 1000,
    ):
        metadata = {
            "component": "ExecutionManager",
            "user_id": user_id,
            "graph_eid": graph_eid,
            "graph_id": graph_id,
            "node_eid": node_eid,
            "node_id": node_id,
            "block_name": block_name,
        }
        prefix = f"[ExecutionManager|uid:{user_id}|gid:{graph_id}|nid:{node_id}]|geid:{graph_eid}|neid:{node_eid}|{block_name}]"
        super().__init__(
            logger,
            max_length=max_length,
            prefix=prefix,
            metadata=metadata,
        )


# ============ Execution Cost Helpers ============ #


def execution_usage_cost(execution_count: int) -> tuple[int, int]:
    """
    Calculate the cost of executing a graph based on the current number of node executions.

    Args:
        execution_count: Number of node executions

    Returns:
        Tuple of cost amount and the number of execution count that is included in the cost.
    """
    return (
        (
            config.execution_cost_per_threshold
            if execution_count % config.execution_cost_count_threshold == 0
            else 0
        ),
        config.execution_cost_count_threshold,
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

# Dynamic field utilities are now imported from backend.data.dynamic_fields


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
    node_block = get_block(node.block_id)
    if not node_block:
        return None, f"Block for {node.block_id} not found."
    schema = node_block.input_schema

    # Input data (without default values) should contain all required fields.
    error_prefix = f"Input data missing or mismatch for `{node_block.name}`:"
    if missing_links := schema.get_missing_links(data, node.input_links):
        return None, f"{error_prefix} unpopulated links {missing_links}"

    # Merge input data with default values and resolve dynamic dict/list/object pins.
    input_default = schema.get_input_defaults(node.input_default)
    data = {**input_default, **data}
    if resolve_input:
        data = merge_execution_input(data)

    # Convert non-matching data types to the expected input schema.
    for name, data_type in schema.__annotations__.items():
        value = data.get(name)
        if (value is not None) and (type(value) is not data_type):
            data[name] = convert(value, data_type)

    # Input data post-merge should contain all required fields from the schema.
    if missing_input := schema.get_missing_input(data):
        return None, f"{error_prefix} missing input {missing_input}"

    # Last validation: Validate the input values against the schema.
    if error := schema.get_mismatch_error(data):
        error_message = f"{error_prefix} {error}"
        logger.warning(error_message)
        return None, error_message

    return data, node_block.name


async def _validate_node_input_credentials(
    graph: GraphModel,
    user_id: str,
    nodes_input_masks: Optional[NodesInputMasks] = None,
) -> dict[str, dict[str, str]]:
    """
    Checks all credentials for all nodes of the graph and returns structured errors.

    Returns:
        dict[node_id, dict[field_name, error_message]]: Credential validation errors per node
    """
    credential_errors: dict[str, dict[str, str]] = defaultdict(dict)

    for node in graph.nodes:
        block = node.block

        # Find any fields of type CredentialsMetaInput
        credentials_fields = block.input_schema.get_credentials_fields()
        if not credentials_fields:
            continue

        for field_name, credentials_meta_type in credentials_fields.items():
            try:
                if (
                    nodes_input_masks
                    and (node_input_mask := nodes_input_masks.get(node.id))
                    and field_name in node_input_mask
                ):
                    credentials_meta = credentials_meta_type.model_validate(
                        node_input_mask[field_name]
                    )
                elif field_name in node.input_default:
                    credentials_meta = credentials_meta_type.model_validate(
                        node.input_default[field_name]
                    )
                else:
                    # Missing credentials
                    credential_errors[node.id][
                        field_name
                    ] = "These credentials are required"
                    continue
            except ValidationError as e:
                credential_errors[node.id][field_name] = f"Invalid credentials: {e}"
                continue

            try:
                # Fetch the corresponding Credentials and perform sanity checks
                credentials = await get_integration_credentials_store().get_creds_by_id(
                    user_id, credentials_meta.id
                )
            except Exception as e:
                # Handle any errors fetching credentials
                credential_errors[node.id][
                    field_name
                ] = f"Credentials not available: {e}"
                continue

            if not credentials:
                credential_errors[node.id][
                    field_name
                ] = f"Unknown credentials #{credentials_meta.id}"
                continue

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
                credential_errors[node.id][
                    field_name
                ] = "Invalid credentials: type/provider mismatch"
                continue

    return credential_errors


def make_node_credentials_input_map(
    graph: GraphModel,
    graph_credentials_input: Mapping[str, CredentialsMetaInput],
) -> NodesInputMasks:
    """
    Maps credentials for an execution to the correct nodes.

    Params:
        graph: The graph to be executed.
        graph_credentials_input: A (graph_input_name, credentials_meta) map.

    Returns:
        dict[node_id, dict[field_name, CredentialsMetaRaw]]: Node credentials input map.
    """
    result: dict[str, dict[str, JsonValue]] = {}

    # Get aggregated credentials fields for the graph
    graph_cred_inputs = graph.aggregate_credentials_inputs()

    for graph_input_name, (_, compatible_node_fields) in graph_cred_inputs.items():
        # Best-effort map: skip missing items
        if graph_input_name not in graph_credentials_input:
            continue

        # Use passed-in credentials for all compatible node input fields
        for node_id, node_field_name in compatible_node_fields:
            if node_id not in result:
                result[node_id] = {}
            result[node_id][node_field_name] = graph_credentials_input[
                graph_input_name
            ].model_dump(exclude_none=True)

    return result


async def validate_graph_with_credentials(
    graph: GraphModel,
    user_id: str,
    nodes_input_masks: Optional[NodesInputMasks] = None,
) -> Mapping[str, Mapping[str, str]]:
    """
    Validate graph including credentials and return structured errors per node.

    Returns:
        dict[node_id, dict[field_name, error_message]]: Validation errors per node
    """
    # Get input validation errors
    node_input_errors = GraphModel.validate_graph_get_errors(
        graph, for_run=True, nodes_input_masks=nodes_input_masks
    )

    # Get credential input/availability/validation errors
    node_credential_input_errors = await _validate_node_input_credentials(
        graph, user_id, nodes_input_masks
    )

    # Merge credential errors with structural errors
    for node_id, field_errors in node_credential_input_errors.items():
        if node_id not in node_input_errors:
            node_input_errors[node_id] = {}
        node_input_errors[node_id].update(field_errors)

    return node_input_errors


async def _construct_starting_node_execution_input(
    graph: GraphModel,
    user_id: str,
    graph_inputs: BlockInput,
    nodes_input_masks: Optional[NodesInputMasks] = None,
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
        node_credentials_map: `dict[node_id, dict[input_name, CredentialsMetaInput]]`

    Returns:
        list[tuple[str, BlockInput]]: A list of tuples, each containing the node ID and
            the corresponding input data for that node.
    """
    # Use new validation function that includes credentials
    validation_errors = await validate_graph_with_credentials(
        graph, user_id, nodes_input_masks
    )
    n_error_nodes = len(validation_errors)
    n_errors = sum(len(errors) for errors in validation_errors.values())
    if validation_errors:
        raise GraphValidationError(
            f"Graph validation failed: {n_errors} issues on {n_error_nodes} nodes",
            node_errors=validation_errors,
        )

    nodes_input = []
    for node in graph.starting_nodes:
        input_data = {}
        block = node.block

        # Note block should never be executed.
        if block.block_type == BlockType.NOTE:
            continue

        # Extract request input data, and assign it to the input pin.
        if block.block_type == BlockType.INPUT:
            input_name = cast(str | None, node.input_default.get("name"))
            if input_name and input_name in graph_inputs:
                input_data = {"value": graph_inputs[input_name]}

        # Apply node input overrides
        if nodes_input_masks and (node_input_mask := nodes_input_masks.get(node.id)):
            input_data.update(node_input_mask)

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


async def validate_and_construct_node_execution_input(
    graph_id: str,
    user_id: str,
    graph_inputs: BlockInput,
    graph_version: Optional[int] = None,
    graph_credentials_inputs: Optional[Mapping[str, CredentialsMetaInput]] = None,
    nodes_input_masks: Optional[NodesInputMasks] = None,
) -> tuple[GraphModel, list[tuple[str, BlockInput]], NodesInputMasks]:
    """
    Public wrapper that handles graph fetching, credential mapping, and validation+construction.
    This centralizes the logic used by both scheduler validation and actual execution.

    Args:
        graph_id: The ID of the graph to validate/construct.
        user_id: The ID of the user.
        graph_inputs: The input data for the graph execution.
        graph_version: The version of the graph to use.
        graph_credentials_inputs: Credentials inputs to use.
        nodes_input_masks: Node inputs to use.

    Returns:
        GraphModel: Full graph object for the given `graph_id`.
        list[tuple[node_id, BlockInput]]: Starting node IDs with corresponding inputs.
        dict[str, BlockInput]: Node input masks including all passed-in credentials.

    Raises:
        NotFoundError: If the graph is not found.
        GraphValidationError: If the graph has validation issues.
        ValueError: If there are other validation errors.
    """
    if prisma.is_connected():
        gdb = graph_db
    else:
        gdb = get_database_manager_async_client()

    graph: GraphModel | None = await gdb.get_graph(
        graph_id=graph_id,
        user_id=user_id,
        version=graph_version,
        include_subgraphs=True,
    )
    if not graph:
        raise NotFoundError(f"Graph #{graph_id} not found.")

    nodes_input_masks = _merge_nodes_input_masks(
        (
            make_node_credentials_input_map(graph, graph_credentials_inputs)
            if graph_credentials_inputs
            else {}
        ),
        nodes_input_masks or {},
    )

    starting_nodes_input = await _construct_starting_node_execution_input(
        graph=graph,
        user_id=user_id,
        graph_inputs=graph_inputs,
        nodes_input_masks=nodes_input_masks,
    )

    return graph, starting_nodes_input, nodes_input_masks


def _merge_nodes_input_masks(
    overrides_map_1: NodesInputMasks,
    overrides_map_2: NodesInputMasks,
) -> NodesInputMasks:
    """Perform a per-node merge of input overrides"""
    result = dict(overrides_map_1).copy()
    for node_id, overrides2 in overrides_map_2.items():
        if node_id in result:
            result[node_id] = {**result[node_id], **overrides2}
        else:
            result[node_id] = overrides2
    return result


# ============ Execution Queue Helpers ============ #

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

# Graceful shutdown timeout constants
# Agent executions can run for up to 1 day, so we need a graceful shutdown period
# that allows long-running executions to complete naturally
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 24 * 60 * 60  # 1 day to complete active executions


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
        arguments={
            # x-consumer-timeout (1 week)
            # Problem: Default 30-minute consumer timeout kills long-running graph executions
            # Original error: "Consumer acknowledgement timed out after 1800000 ms (30 minutes)"
            # Solution: Disable consumer timeout entirely - let graphs run indefinitely
            # Safety: Heartbeat mechanism now handles dead consumer detection instead
            # Use case: Graph executions that take hours to complete (AI model training, etc.)
            "x-consumer-timeout": GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS
            * 1000,
        },
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


class CancelExecutionEvent(BaseModel):
    graph_exec_id: str


async def stop_graph_execution(
    user_id: str,
    graph_exec_id: str,
    wait_timeout: float = 15.0,
):
    """
    Mechanism:
    1. Set the cancel event
    2. Graph executor's cancel handler thread detects the event, terminates workers,
       reinitializes worker pool, and returns.
    3. Update execution statuses in DB and set `error` outputs to `"TERMINATED"`.
    """
    queue_client = await get_async_execution_queue()
    db = execution_db if prisma.is_connected() else get_database_manager_async_client()
    await queue_client.publish_message(
        routing_key="",
        message=CancelExecutionEvent(graph_exec_id=graph_exec_id).model_dump_json(),
        exchange=GRAPH_EXECUTION_CANCEL_EXCHANGE,
    )

    if not wait_timeout:
        return

    start_time = time.time()
    while time.time() - start_time < wait_timeout:
        graph_exec = await db.get_graph_execution_meta(
            execution_id=graph_exec_id, user_id=user_id
        )

        if not graph_exec:
            raise NotFoundError(f"Graph execution #{graph_exec_id} not found.")

        if graph_exec.status in [
            ExecutionStatus.TERMINATED,
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
        ]:
            # If graph execution is terminated/completed/failed, cancellation is complete
            await get_async_execution_event_bus().publish(graph_exec)
            return

        if graph_exec.status in [
            ExecutionStatus.QUEUED,
            ExecutionStatus.INCOMPLETE,
        ]:
            # If the graph is still on the queue, we can prevent them from being executed
            # by setting the status to TERMINATED.
            graph_exec.status = ExecutionStatus.TERMINATED

            await asyncio.gather(
                # Update graph execution status
                db.update_graph_execution_stats(
                    graph_exec_id=graph_exec.id,
                    status=ExecutionStatus.TERMINATED,
                ),
                # Publish graph execution event
                get_async_execution_event_bus().publish(graph_exec),
            )
            return

        if graph_exec.status == ExecutionStatus.RUNNING:
            await asyncio.sleep(0.1)

    raise TimeoutError(
        f"Graph execution #{graph_exec_id} will need to take longer than {wait_timeout} seconds to stop. "
        f"You can check the status of the execution in the UI or try again later."
    )


async def add_graph_execution(
    graph_id: str,
    user_id: str,
    inputs: Optional[BlockInput] = None,
    preset_id: Optional[str] = None,
    graph_version: Optional[int] = None,
    graph_credentials_inputs: Optional[Mapping[str, CredentialsMetaInput]] = None,
    nodes_input_masks: Optional[NodesInputMasks] = None,
) -> GraphExecutionWithNodes:
    """
    Adds a graph execution to the queue and returns the execution entry.

    Args:
        graph_id: The ID of the graph to execute.
        user_id: The ID of the user executing the graph.
        inputs: The input data for the graph execution.
        preset_id: The ID of the preset to use.
        graph_version: The version of the graph to execute.
        graph_credentials_inputs: Credentials inputs to use in the execution.
            Keys should map to the keys generated by `GraphModel.aggregate_credentials_inputs`.
        nodes_input_masks: Node inputs to use in the execution.
    Returns:
        GraphExecutionEntry: The entry for the graph execution.
    Raises:
        ValueError: If the graph is not found or if there are validation errors.
    """
    if prisma.is_connected():
        edb = execution_db
    else:
        edb = get_database_manager_async_client()

    graph, starting_nodes_input, compiled_nodes_input_masks = (
        await validate_and_construct_node_execution_input(
            graph_id=graph_id,
            user_id=user_id,
            graph_inputs=inputs or {},
            graph_version=graph_version,
            graph_credentials_inputs=graph_credentials_inputs,
            nodes_input_masks=nodes_input_masks,
        )
    )
    graph_exec = None

    try:
        # Sanity check: running add_graph_execution with the properties of
        # the graph_exec created here should create the same execution again.
        graph_exec = await edb.create_graph_execution(
            user_id=user_id,
            graph_id=graph_id,
            graph_version=graph.version,
            inputs=inputs or {},
            credential_inputs=graph_credentials_inputs,
            nodes_input_masks=nodes_input_masks,
            starting_nodes_input=starting_nodes_input,
            preset_id=preset_id,
        )

        graph_exec_entry = graph_exec.to_graph_execution_entry(
            user_context=await get_user_context(user_id),
            compiled_nodes_input_masks=compiled_nodes_input_masks,
        )
        logger.info(
            f"Created graph execution #{graph_exec.id} for graph "
            f"#{graph_id} with {len(starting_nodes_input)} starting nodes. "
            f"Now publishing to execution queue."
        )

        exec_queue = await get_async_execution_queue()
        await exec_queue.publish_message(
            routing_key=GRAPH_EXECUTION_ROUTING_KEY,
            message=graph_exec_entry.model_dump_json(),
            exchange=GRAPH_EXECUTION_EXCHANGE,
        )
        logger.info(f"Published execution {graph_exec.id} to RabbitMQ queue")

        graph_exec.status = ExecutionStatus.QUEUED
        await edb.update_graph_execution_stats(
            graph_exec_id=graph_exec.id,
            status=graph_exec.status,
        )
        await get_async_execution_event_bus().publish(graph_exec)

        return graph_exec
    except BaseException as e:
        err = str(e) or type(e).__name__
        if not graph_exec:
            logger.error(f"Unable to execute graph #{graph_id} failed: {err}")
            raise

        logger.error(
            f"Unable to publish graph #{graph_id} exec #{graph_exec.id}: {err}"
        )
        await edb.update_node_execution_status_batch(
            [node_exec.node_exec_id for node_exec in graph_exec.node_executions],
            ExecutionStatus.FAILED,
        )
        await edb.update_graph_execution_stats(
            graph_exec_id=graph_exec.id,
            status=ExecutionStatus.FAILED,
            stats=GraphExecutionStats(error=err),
        )
        raise


# ============ Execution Output Helpers ============ #


class ExecutionOutputEntry(BaseModel):
    node: Node
    node_exec_id: str
    data: BlockOutputEntry


class NodeExecutionProgress:
    def __init__(self):
        self.output: dict[str, list[ExecutionOutputEntry]] = defaultdict(list)
        self.tasks: dict[str, Future] = {}
        self._lock = threading.Lock()

    def add_task(self, node_exec_id: str, task: Future):
        self.tasks[node_exec_id] = task

    def add_output(self, output: ExecutionOutputEntry):
        with self._lock:
            self.output[output.node_exec_id].append(output)

    def pop_output(self) -> ExecutionOutputEntry | None:
        exec_id = self._next_exec()
        if not exec_id:
            return None

        if self._pop_done_task(exec_id):
            return self.pop_output()

        with self._lock:
            if next_output := self.output[exec_id]:
                return next_output.pop(0)

        return None

    def is_done(self, wait_time: float = 0.0) -> bool:
        exec_id = self._next_exec()
        if not exec_id:
            return True

        if self._pop_done_task(exec_id):
            return self.is_done(wait_time)

        if wait_time <= 0:
            return False

        try:
            self.tasks[exec_id].result(wait_time)
        except TimeoutError:
            pass
        except Exception as e:
            logger.error(
                f"Task for exec ID {exec_id} failed with error: {e.__class__.__name__} {str(e)}"
            )
            pass
        return self.is_done(0)

    def stop(self) -> list[str]:
        """
        Stops all tasks and clears the output.
        This is useful for cleaning up when the execution is cancelled or terminated.
        Returns a list of execution IDs that were stopped.
        """
        cancelled_ids = []
        for task_id, task in self.tasks.items():
            if task.done():
                continue
            task.cancel()
            cancelled_ids.append(task_id)
        return cancelled_ids

    def wait_for_done(self, timeout: float = 5.0):
        """
        Wait for all cancelled tasks to complete cancellation.

        Args:
            timeout: Maximum time to wait for cancellation in seconds
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            while self.pop_output():
                pass

            if self.is_done():
                return

            time.sleep(0.1)  # Small delay to avoid busy waiting

        raise TimeoutError(
            f"Timeout waiting for cancellation of tasks: {list(self.tasks.keys())}"
        )

    def _pop_done_task(self, exec_id: str) -> bool:
        task = self.tasks.get(exec_id)
        if not task:
            return True

        if not task.done():
            return False

        with self._lock:
            if self.output[exec_id]:
                return False

        self.tasks.pop(exec_id)
        return True

    def _next_exec(self) -> str | None:
        if not self.tasks:
            return None
        return next(iter(self.tasks.keys()))
