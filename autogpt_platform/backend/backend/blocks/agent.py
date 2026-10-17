import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockInput,
    BlockOutput,
    BlockSchema,
    BlockSchemaInput,
    BlockType,
)
from backend.data.event_bus import EventStreamLostError
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    ExecutionTrigger,
    GraphExecutionEvent,
    NodesInputMasks,
)
from backend.data.model import NodeExecutionStats, SchemaField
from backend.util.json import validate_with_jsonschema
from backend.util.retry import func_retry

if TYPE_CHECKING:
    from backend.data.execution import (
        ExecutionEvent,
        GraphExecution,
        NodeExecutionResult,
    )
    from backend.executor.utils import LogMetadata

_logger = logging.getLogger(__name__)

TERMINAL_STATUSES = (
    ExecutionStatus.COMPLETED,
    ExecutionStatus.TERMINATED,
    ExecutionStatus.FAILED,
)
# How long the sub-agent's stream may stay quiet before the database is
# checked, bounding how long a missed completion can leave this block waiting.
_QUIET_STREAM_RECHECK_S = 30.0
_RESUBSCRIBE_DELAY_S = 1.0


class AgentExecutorBlock(Block):
    # Coordination block: waits on a child graph's full execution. The child
    # has its own per-node wall-clock caps, so applying the parent's leaf-
    # block cap here would false-positive on legitimately long sub-agent runs.
    execution_timeout_seconds: int | None = None

    class Input(BlockSchemaInput):
        user_id: str = SchemaField(description="User ID")
        graph_id: str = SchemaField(description="Graph ID")
        graph_version: int = SchemaField(description="Graph Version")
        agent_name: Optional[str] = SchemaField(
            default=None, description="Name to display in the Builder UI"
        )

        inputs: BlockInput = SchemaField(description="Input data for the graph")
        input_schema: dict = SchemaField(description="Input schema for the graph")
        output_schema: dict = SchemaField(description="Output schema for the graph")

        nodes_input_masks: Optional[NodesInputMasks] = SchemaField(
            default=None, hidden=True
        )

        @classmethod
        def get_input_schema(cls, data: BlockInput) -> dict[str, Any]:
            return data.get("input_schema", {})

        @classmethod
        def get_input_defaults(cls, data: BlockInput) -> BlockInput:
            return data.get("inputs", {})

        @classmethod
        def get_missing_input(cls, data: BlockInput) -> set[str]:
            required_fields = cls.get_input_schema(data).get("required", [])
            # Check against the nested `inputs` dict, not the top-level node
            # data — required fields like "topic" live inside data["inputs"],
            # not at data["topic"].
            provided = data.get("inputs", {})
            return set(required_fields) - set(provided)

        @classmethod
        def get_mismatch_error(cls, data: BlockInput) -> str | None:
            return validate_with_jsonschema(
                cls.get_input_schema(data), data.get("inputs", {})
            )

    class Output(BlockSchema):
        # Use BlockSchema to avoid automatic error field that could clash with graph outputs
        pass

    def __init__(self):
        super().__init__(
            id="e189baac-8c20-45a1-94a7-55177ea42565",
            description="Executes an existing agent inside your agent",
            input_schema=AgentExecutorBlock.Input,
            output_schema=AgentExecutorBlock.Output,
            block_type=BlockType.AGENT,
            categories={BlockCategory.AGENT},
        )

    async def run(
        self,
        input_data: Input,
        *,
        graph_exec_id: str,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        from backend.executor import utils as execution_utils

        graph_exec = await execution_utils.add_graph_execution(
            graph_id=input_data.graph_id,
            graph_version=input_data.graph_version,
            user_id=input_data.user_id,
            inputs=input_data.inputs,
            nodes_input_masks=input_data.nodes_input_masks,
            execution_context=execution_context.model_copy(
                update={"parent_execution_id": graph_exec_id},
            ),
            dry_run=execution_context.dry_run,
            organization_id=execution_context.organization_id,
            team_id=execution_context.team_id,
            trigger=ExecutionTrigger.SUBGRAPH,
            trigger_ref=graph_exec_id,
        )

        logger = execution_utils.LogMetadata(
            logger=_logger,
            user_id=input_data.user_id,
            graph_eid=graph_exec.id,
            graph_id=input_data.graph_id,
            node_eid="*",
            node_id="*",
            block_name=self.name,
        )

        try:
            async for name, data in self._run(
                graph_id=input_data.graph_id,
                graph_version=input_data.graph_version,
                graph_exec_id=graph_exec.id,
                user_id=input_data.user_id,
                logger=logger,
            ):
                yield name, data
        except BaseException as e:
            await self._stop(
                graph_exec_id=graph_exec.id,
                user_id=input_data.user_id,
                logger=logger,
            )
            logger.warning(
                f"Execution of graph {input_data.graph_id}v{input_data.graph_version} failed: {e.__class__.__name__} {str(e)}; execution is stopped."
            )
            raise

    async def _run(
        self,
        graph_id: str,
        graph_version: int,
        graph_exec_id: str,
        user_id: str,
        logger: "LogMetadata",
    ) -> BlockOutput:
        log_id = f"Graph #{graph_id}-V{graph_version}, exec-id: {graph_exec_id}"
        logger.info(f"Starting execution of {log_id}")
        yielded_node_exec_ids: set[str] = set()

        # This block has no execution timeout, so the sub-agent's completion
        # must not hinge on a single pub/sub message: whenever the stream is
        # lost or goes quiet, the database says where the sub-agent is.
        events: asyncio.Queue["ExecutionEvent | Exception | None"] = asyncio.Queue()
        reader = asyncio.create_task(
            self._read_events(events, user_id, graph_id, graph_exec_id, log_id, logger)
        )
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        events.get(), _QUIET_STREAM_RECHECK_S
                    )
                except asyncio.TimeoutError:
                    event = None

                if isinstance(event, Exception):
                    raise event

                if event is None:
                    outputs, finished = await self._catch_up(
                        user_id, graph_exec_id, yielded_node_exec_ids, log_id, logger
                    )
                    for output in outputs:
                        yield output
                    if finished is not None:
                        self._record_sub_graph_stats(finished.stats)
                        return
                    continue

                if event.status not in TERMINAL_STATUSES:
                    logger.info(
                        f"Execution {log_id} skipping event {event.event_type} status={event.status} "
                        f"node={getattr(event, 'node_exec_id', '?')}"
                    )
                    continue

                if isinstance(event, GraphExecutionEvent):
                    logger.info(
                        f"Execution {log_id} graph completed with status {event.status}, "
                        f"yielded {len(yielded_node_exec_ids)} outputs"
                    )
                    self._record_sub_graph_stats(event.stats)
                    return

                for output in self._outputs_of(
                    event, yielded_node_exec_ids, log_id, logger
                ):
                    yield output
        finally:
            reader.cancel()
            with suppress(asyncio.CancelledError):
                await reader

    async def _read_events(
        self,
        events: asyncio.Queue["ExecutionEvent | Exception | None"],
        user_id: str,
        graph_id: str,
        graph_exec_id: str,
        log_id: str,
        logger: "LogMetadata",
    ) -> None:
        """Feed the sub-agent's events into ``events``, resubscribing whenever
        the stream is lost; ``None`` marks a gap to catch up on. Any other
        error is handed to ``_run`` to raise rather than retried."""
        from backend.executor import utils as execution_utils

        event_bus = execution_utils.get_async_execution_event_bus()
        while True:
            try:
                # Once subscribed, a catch-up from the database can't miss
                # anything: earlier events are stored, later ones stream in.
                async for event in event_bus.listen(
                    user_id=user_id,
                    graph_id=graph_id,
                    graph_exec_id=graph_exec_id,
                    on_subscribed=lambda: events.put_nowait(None),
                ):
                    events.put_nowait(event)
            except EventStreamLostError as e:
                logger.warning(f"Execution {log_id} lost its event stream: {e}")
            except Exception as e:
                events.put_nowait(e)
                return
            await asyncio.sleep(_RESUBSCRIBE_DELAY_S)

    async def _catch_up(
        self,
        user_id: str,
        graph_exec_id: str,
        yielded_node_exec_ids: set[str],
        log_id: str,
        logger: "LogMetadata",
    ) -> tuple[list[tuple[str, Any]], "GraphExecution | None"]:
        """Outputs the stream may have missed, and the sub-agent's execution
        if it has finished."""
        from backend.util.clients import get_database_manager_async_client

        db = get_database_manager_async_client()
        graph_exec = await db.get_graph_execution(
            user_id=user_id, execution_id=graph_exec_id
        )
        if graph_exec is None:
            return [], None
        finished = graph_exec.status in TERMINAL_STATUSES
        # Ownership was checked above; node rows belong to that execution.
        nodes = await db.get_node_executions(
            graph_exec_id=graph_exec_id, statuses=[ExecutionStatus.COMPLETED]
        )
        outputs = [
            output
            for node in nodes
            for output in self._outputs_of(node, yielded_node_exec_ids, log_id, logger)
        ]
        if finished:
            logger.info(
                f"Execution {log_id} found finished ({graph_exec.status}) "
                "in the database"
            )
        return outputs, graph_exec if finished else None

    def _outputs_of(
        self,
        node: "NodeExecutionResult",
        yielded_node_exec_ids: set[str],
        log_id: str,
        logger: "LogMetadata",
    ) -> list[tuple[str, Any]]:
        from backend.blocks import get_block

        if node.node_exec_id in yielded_node_exec_ids:
            logger.debug(f"{log_id} already yielded node execution {node.node_exec_id}")
            return []
        yielded_node_exec_ids.add(node.node_exec_id)

        if not node.block_id:
            logger.warning(f"{log_id} received event without block_id {node}")
            return []

        block = get_block(node.block_id)
        if not block or block.block_type != BlockType.OUTPUT:
            return []

        output_name = node.input_data.get("name")
        if not output_name:
            logger.warning(f"{log_id} produced an output with no name {node}")
            return []

        logger.debug(f"Execution {log_id} produced {output_name}")
        return [(output_name, data) for data in node.output_data.get("output", [])]

    def _record_sub_graph_stats(self, stats: "GraphExecution.Stats | None") -> None:
        self.merge_stats(
            NodeExecutionStats(
                # Sub-graph already debited each of its own nodes; we
                # roll up its total so graph_stats.cost reflects the
                # full sub-graph spend.
                reconciled_cost_delta=(stats.cost if stats else 0),
                extra_steps=stats.node_exec_count if stats else 0,
            )
        )

    @func_retry
    async def _stop(
        self,
        graph_exec_id: str,
        user_id: str,
        logger: "LogMetadata",
    ) -> None:
        from backend.executor import utils as execution_utils

        log_id = f"Graph exec-id: {graph_exec_id}"
        logger.info(f"Stopping execution of {log_id}")

        try:
            await execution_utils.stop_graph_execution(
                graph_exec_id=graph_exec_id,
                user_id=user_id,
                wait_timeout=3600,
            )
            logger.info(f"Execution {log_id} stopped successfully.")
        except TimeoutError as e:
            logger.error(f"Execution {log_id} stop timed out: {e}")
