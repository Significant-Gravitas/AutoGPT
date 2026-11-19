import logging
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
from backend.blocks._sub_agent_wait import SubAgentWait
from backend.data.execution import (
    ExecutionContext,
    ExecutionTrigger,
    GraphExecutionMeta,
    NodesInputMasks,
    exec_channel,
)
from backend.data.model import NodeExecutionStats, SchemaField
from backend.util.retry import func_retry

if TYPE_CHECKING:
    from backend.executor.utils import LogMetadata

_logger = logging.getLogger(__name__)

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
        wait = SubAgentWait(
            event_bus.listen_with_resync(
                exec_channel(user_id, graph_id, graph_exec_id)
            ),
            user_id=user_id,
            graph_exec_id=graph_exec_id,
            log_id=log_id,
            logger=logger,
        )
        async for output in wait.outputs():
            yield output
        self._record_sub_graph_stats(wait.stats)

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
