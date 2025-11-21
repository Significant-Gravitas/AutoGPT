import logging
from typing import Any, Literal

from prisma.enums import ReviewStatus

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionStatus
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
        reviewed_data: Any = SchemaField(
            description="The data after human review (may be modified)"
        )
        status: Literal["approved", "rejected"] = SchemaField(
            description="Status of the review: 'approved' or 'rejected'"
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
            test_input={
                "data": {"name": "John Doe", "age": 30},
                "name": "User profile data",
                "editable": True,
            },
            test_output=[
                ("reviewed_data", {"name": "John Doe", "age": 30}),
                ("status", "approved"),
                ("review_message", ""),
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
            },
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
        **kwargs,
    ) -> BlockOutput:
        """
        Execute the Human In The Loop block.

        This method uses one function to handle the complete workflow - checking existing reviews
        and creating pending ones as needed.
        """
        try:
            logger.debug(f"HITL block executing for node {node_exec_id}")

            # Use the data layer to handle the complete workflow
            db_client = get_database_manager_async_client()
            result = await db_client.get_or_create_human_review(
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

        # Check if we're waiting for human input
        if result is None:
            logger.info(
                f"HITL block pausing execution for node {node_exec_id} - awaiting human review"
            )
            try:
                # Set node status to REVIEW so execution manager can't mark it as COMPLETED
                # The VALID_STATUS_TRANSITIONS will then prevent any unwanted status changes
                # Use the proper wrapper function to ensure websocket events are published
                await async_update_node_execution_status(
                    db_client=db_client,
                    exec_id=node_exec_id,
                    status=ExecutionStatus.REVIEW,
                )
                # Execution pauses here until API routes process the review
                return
            except Exception as e:
                logger.error(
                    f"Failed to update node status for HITL block {node_exec_id}: {str(e)}"
                )
                raise

        # Review is complete (approved or rejected) - check if unprocessed
        if not result.processed:
            # Mark as processed before yielding
            await db_client.update_review_processed_status(
                node_exec_id=node_exec_id, processed=True
            )

            if result.status == ReviewStatus.APPROVED:
                yield "status", "approved"
                yield "reviewed_data", result.data
                if result.message:
                    yield "review_message", result.message

            elif result.status == ReviewStatus.REJECTED:
                yield "status", "rejected"
                if result.message:
                    yield "review_message", result.message
