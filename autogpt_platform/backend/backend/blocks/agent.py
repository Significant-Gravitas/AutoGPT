import asyncio
import logging
from collections.abc import AsyncIterator
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
from backend.data.db_accessors import execution_db
from backend.data.execution import (
    ExecutionContext,
    ExecutionEvent,
    ExecutionEventType,
    ExecutionStatus,
    ExecutionTrigger,
    GraphExecutionMeta,
    NodeExecutionResult,
    NodesInputMasks,
    exec_channel,
)
from backend.data.model import NodeExecutionStats, SchemaField
from backend.util.retry import func_retry

if TYPE_CHECKING:
    from backend.executor.utils import LogMetadata

_logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = (
    ExecutionStatus.COMPLETED,
    ExecutionStatus.TERMINATED,
    ExecutionStatus.FAILED,
)
# A sub-agent's end can go unheard on a healthy stream too (its executor failed
# to publish it), so after this long without an event the database is asked.
# Bounds how long such a miss holds the parent run, which has no timeout.
_QUIET_RECHECK_S = 60.0

NO_WORKFLOW_TO_RUN = (
    "This sub-workflow has no workflow to run: remove it, or add the workflow you "
    "want in its place"
)


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
        def get_mismatch_error(
            cls, data: BlockInput, exclude_fields: set[str] | None = None
        ) -> str | None:
            # `data` holds the flattened `inputs` (see validate_exec), never the
            # node's `input_schema`; the child graph validates its own inputs.
            return None

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

        if not input_data.graph_id:
            # A copy of someone else's workflow keeps a node like this where
            # the original named one of its new owner's own workflows (see
            # `clear_sub_graph_refs_owned_by`).
            raise ValueError(NO_WORKFLOW_TO_RUN)

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
        from backend.executor import utils as execution_utils

        event_bus = execution_utils.get_async_execution_event_bus()
        log_id = f"Graph #{graph_id}-V{graph_version}, exec-id: {graph_exec_id}"
        logger.info(f"Starting execution of {log_id}")
        yielded: set[str] = set()
        # Pub/sub keeps no history, so the stream is only the fast path: when
        # it may have missed events, or stays quiet, the database is read.
        events: asyncio.Queue[ExecutionEvent | None] = asyncio.Queue()
        reader = asyncio.create_task(
            _read_into(
                events,
                event_bus.listen_with_resync(
                    exec_channel(user_id, graph_id, graph_exec_id)
                ),
                log_id,
            )
        )
        stale = False  # events may have been missed and not caught up yet
        try:
            while True:
                try:
                    async with asyncio.timeout(_QUIET_RECHECK_S):
                        event = await events.get()
                    stale = stale or event is None
                except TimeoutError:
                    event = None
                if event is not None:
                    if event.status not in _TERMINAL_STATUSES:
                        logger.info(
                            f"Execution {log_id} skipping event {event.event_type} "
                            f"status={event.status} "
                            f"node={getattr(event, 'node_exec_id', '?')}"
                        )
                        continue
                    if not stale:
                        if event.event_type == ExecutionEventType.NODE_EXEC_UPDATE:
                            for output in self._outputs_of(
                                event, yielded, log_id, logger
                            ):
                                yield output
                            continue
                        logger.info(
                            f"Execution {log_id} graph completed with status "
                            f"{event.status}, yielded {len(yielded)} outputs"
                        )
                        self._record_sub_graph_stats(event.stats)
                        return
                # A resync, a quiet spell, or an event while events may have
                # been missed (which must not overtake them): the database has
                # the rest, in order.
                try:
                    outputs, finished = await self._catch_up(
                        user_id, graph_exec_id, stale, yielded, log_id, logger
                    )
                except Exception:
                    # Like a missed event: the next event, resync or quiet
                    # spell retries, and nothing newer is yielded meanwhile.
                    logger.warning(f"Execution {log_id} catch-up failed", exc_info=True)
                    continue
                stale = False
                if event is not None and (
                    event.event_type == ExecutionEventType.NODE_EXEC_UPDATE
                ):
                    # In case the database hadn't caught up with it yet.
                    outputs += self._outputs_of(event, yielded, log_id, logger)
                for output in outputs:
                    yield output
                if finished is not None:
                    logger.info(
                        f"Execution {log_id} found {finished.status} in the database"
                    )
                    self._record_sub_graph_stats(finished.stats)
                    return
                if event is not None and (
                    event.event_type == ExecutionEventType.GRAPH_EXEC_UPDATE
                ):
                    # The live end, which the database hadn't recorded yet.
                    logger.info(f"Execution {log_id} graph completed: {event.status}")
                    self._record_sub_graph_stats(event.stats)
                    return
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def _catch_up(
        self,
        user_id: str,
        graph_exec_id: str,
        stale: bool,
        yielded: set[str],
        log_id: str,
        logger: "LogMetadata",
    ) -> tuple[list[tuple[str, Any]], GraphExecutionMeta | None]:
        """Outputs the stream may have missed, oldest first, and the
        sub-agent's execution if it has finished."""
        db = execution_db()
        meta = await db.get_graph_execution_meta(
            user_id=user_id, execution_id=graph_exec_id
        )
        if meta is None:
            # Not read, so nothing is caught up: the caller retries.
            raise LookupError(f"{log_id}: execution not found")
        finished = meta if meta.status in _TERMINAL_STATUSES else None
        if not (stale or finished):
            return [], None
        from backend.blocks import get_output_block_ids

        # Read with the user's id just above, so these nodes are the user's.
        nodes = await db.get_node_executions(
            graph_exec_id=graph_exec_id,
            statuses=list(_TERMINAL_STATUSES),
            block_ids=list(get_output_block_ids()),
        )
        # Reversed first: equal times then keep the order they were queued in.
        nodes = sorted(reversed(nodes), key=lambda node: node.end_time or node.add_time)
        outputs = [
            output
            for node in nodes
            for output in self._outputs_of(node, yielded, log_id, logger)
        ]
        return outputs, finished

    def _outputs_of(
        self,
        node: NodeExecutionResult,
        yielded: set[str],
        log_id: str,
        logger: "LogMetadata",
    ) -> list[tuple[str, Any]]:
        from backend.blocks import get_block

        if node.node_exec_id in yielded:
            logger.debug(f"{log_id} already yielded node execution {node.node_exec_id}")
            return []
        yielded.add(node.node_exec_id)

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

    def _record_sub_graph_stats(self, stats: GraphExecutionMeta.Stats | None) -> None:
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


async def _read_into(
    queue: asyncio.Queue[ExecutionEvent | None],
    events: AsyncIterator[ExecutionEvent | None],
    log_id: str,
) -> None:
    try:
        async for event in events:
            queue.put_nowait(event)
    except Exception:
        _logger.exception(
            f"{log_id}: event stream failed; checking the database every "
            f"{_QUIET_RECHECK_S:.0f}s instead"
        )
