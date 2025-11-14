from typing import Any

from backend.blocks.human_in_the_loop_service import (
    HITLValidationError,
    HumanInTheLoopService,
)
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionStatus, update_graph_execution_stats
from backend.data.model import SchemaField


class HumanInTheLoopBlock(Block):
    """
    This block pauses execution and waits for human approval or modification of the data.

    When executed, it creates a pending review entry and sets the node execution status
    to WAITING_FOR_REVIEW. The execution will remain paused until a human user either:
    - Approves the data (with or without modifications)
    - Rejects the data

    This is useful for workflows that require human validation or intervention before
    proceeding to the next steps.
    """

    class Input(BlockSchemaInput):
        data: Any = SchemaField(description="The data to be reviewed by a human user")
        message: str = SchemaField(
            description="Instructions or message for the human reviewer",
            default="Please review and approve or modify the following data:",
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
        status: str = SchemaField(
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
                "message": "Please verify this user data",
                "editable": True,
            },
            test_output=[
                ("reviewed_data", {"name": "John Doe", "age": 30}),
                ("status", "approved"),
                ("review_message", ""),
            ],
            test_mock={
                "handle_review_workflow": lambda *args, **kwargs: HumanInTheLoopBlock._create_test_result(),
                "update_graph_execution_stats": lambda *args, **kwargs: None,
            },
        )

    @staticmethod
    def _create_test_result():
        """Create test result for mocking"""
        from backend.blocks.human_in_the_loop_service import ReviewResult

        return ReviewResult(
            data={"name": "John Doe", "age": 30}, status="approved", message=""
        )

    async def handle_review_workflow(self, *args, **kwargs):
        """Wrapper method for HumanInTheLoopService.handle_review_workflow that can be mocked"""
        return await HumanInTheLoopService.handle_review_workflow(*args, **kwargs)

    async def update_graph_execution_stats(self, *args, **kwargs):
        """Wrapper method for update_graph_execution_stats that can be mocked"""
        return await update_graph_execution_stats(*args, **kwargs)

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

        This method uses the HumanInTheLoopService to handle all business logic,
        keeping the block implementation clean and focused on data flow.
        """
        # Use the service to handle the complete workflow
        result = await self.handle_review_workflow(
            user_id=user_id,
            node_exec_id=node_exec_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            graph_version=graph_version,
            input_data=input_data.data,
            message=input_data.message,
            editable=input_data.editable,
            expected_data_type=type(input_data.data),
        )

        if result is not None:
            # Review is complete (approved or rejected)
            if result.status == "approved":
                yield "reviewed_data", result.data
                yield "status", "approved"
                yield "review_message", result.message
            elif result.status == "rejected":
                yield "status", "rejected"
                yield "review_message", result.message
                # Raise an exception for rejected reviews to stop execution
                raise HITLValidationError(
                    f"Human review rejected: {result.message}", result.message
                )
            return

        # No result means we're waiting for human input
        # Update the graph execution status to indicate waiting state
        await self.update_graph_execution_stats(
            graph_exec_id=graph_exec_id, status=ExecutionStatus.WAITING_FOR_REVIEW
        )

        # This will pause the execution here
        # The execution will be resumed when the review is approved via the API
        return
