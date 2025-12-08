import logging
from typing import Any

from prisma.enums import ReviewStatus

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockType,
)
from backend.data.execution import ExecutionContext, ExecutionStatus
from backend.data.human_review import ReviewResult
from backend.data.model import SchemaField
from backend.executor.manager import async_update_node_execution_status
from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)


class HumanInTheLoopBlock(Block):
    """
    This block pauses execution and waits for human approval or modification of the data.

    When executed, it creates a pending review entry and sets the node execution status
    to REVIEW. The execution will remain paused until a human user either:
    - Approves the data (with or without modifications)
    - Rejects the data

    This is useful for workflows that require human validation or intervention before
    proceeding to the next steps.
    """

    class Input(BlockSchemaInput):
        data: Any = SchemaField(description="The data to be reviewed by a human user")
        name: str = SchemaField(
            description="A descriptive name for what this data represents",
        )
        editable: bool = SchemaField(
            description="Whether the human reviewer can edit the data",
            default=True,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        approved_data: Any = SchemaField(
            description="The data when approved (may be modified by reviewer)"
        )
        rejected_data: Any = SchemaField(
            description="The data when rejected (may be modified by reviewer)"
        )
        review_message: str = SchemaField(
            description="Any message provided by the reviewer", default=""
        )

    def __init__(self):
        super().__init__(
            id="8b2a7b3c-6e9d-4a5f-8c1b-2e3f4a5b6c7d",
            description="Pause execution and wait for human approval or modification of data",
            categories={BlockCategory.BASIC},
            input_schema=HumanInTheLoopBlock.Input,
            output_schema=HumanInTheLoopBlock.Output,
            block_type=BlockType.HUMAN_IN_THE_LOOP,
            test_input={
                "data": {"name": "John Doe", "age": 30},
                "name": "User profile data",
                "editable": True,
            },
            test_output=[
                ("approved_data", {"name": "John Doe", "age": 30}),
            ],
            test_mock={
                "get_or_create_human_review": lambda *_args, **_kwargs: ReviewResult(
                    data={"name": "John Doe", "age": 30},
                    status=ReviewStatus.APPROVED,
                    message="",
                    processed=False,
                    node_exec_id="test-node-exec-id",
                ),
                "update_node_execution_status": lambda *_args, **_kwargs: None,
                "update_review_processed_status": lambda *_args, **_kwargs: None,
            },
        )

    async def get_or_create_human_review(self, **kwargs):
        return await get_database_manager_async_client().get_or_create_human_review(
            **kwargs
        )

    async def update_node_execution_status(self, **kwargs):
        return await async_update_node_execution_status(
            db_client=get_database_manager_async_client(), **kwargs
        )

    async def update_review_processed_status(self, node_exec_id: str, processed: bool):
        return await get_database_manager_async_client().update_review_processed_status(
            node_exec_id, processed
        )

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        node_exec_id: str,
        graph_exec_id: str,
        graph_id: str,
        graph_version: int,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        if not execution_context.safe_mode:
            logger.info(
                f"HITL block skipping review for node {node_exec_id} - safe mode disabled"
            )
            yield "approved_data", input_data.data
            yield "review_message", "Auto-approved (safe mode disabled)"
            return

        try:
            result = await self.get_or_create_human_review(
                user_id=user_id,
                node_exec_id=node_exec_id,
                graph_exec_id=graph_exec_id,
                graph_id=graph_id,
                graph_version=graph_version,
                input_data=input_data.data,
                message=input_data.name,
                editable=input_data.editable,
            )
        except Exception as e:
            logger.error(f"Error in HITL block for node {node_exec_id}: {str(e)}")
            raise

        if result is None:
            logger.info(
                f"HITL block pausing execution for node {node_exec_id} - awaiting human review"
            )
            try:
                await self.update_node_execution_status(
                    exec_id=node_exec_id,
                    status=ExecutionStatus.REVIEW,
                )
                return
            except Exception as e:
                logger.error(
                    f"Failed to update node status for HITL block {node_exec_id}: {str(e)}"
                )
                raise

        if not result.processed:
            await self.update_review_processed_status(
                node_exec_id=node_exec_id, processed=True
            )

            if result.status == ReviewStatus.APPROVED:
                yield "approved_data", result.data
                if result.message:
                    yield "review_message", result.message

            elif result.status == ReviewStatus.REJECTED:
                yield "rejected_data", result.data
                if result.message:
                    yield "review_message", result.message
