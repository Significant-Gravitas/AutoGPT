import logging
from typing import Any

from prisma.enums import ReviewStatus

from backend.blocks.helpers.review import HITLReviewHelper
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockType,
)
from backend.data.execution import ExecutionContext
from backend.data.human_review import ReviewResult
from backend.data.model import SchemaField

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
                "handle_review_decision": lambda **kwargs: type(
                    "ReviewDecision",
                    (),
                    {
                        "should_proceed": True,
                        "message": "Test approval message",
                        "review_result": ReviewResult(
                            data={"name": "John Doe", "age": 30},
                            status=ReviewStatus.APPROVED,
                            message="",
                            processed=False,
                            node_exec_id="test-node-exec-id",
                        ),
                    },
                )(),
            },
        )

    async def handle_review_decision(self, **kwargs):
        return await HITLReviewHelper.handle_review_decision(**kwargs)

    async def run(
        self,
        input_data: Input,
        *,
        user_id: str,
        node_id: str,
        node_exec_id: str,
        graph_exec_id: str,
        graph_id: str,
        graph_version: int,
        execution_context: ExecutionContext,
        **_kwargs,
    ) -> BlockOutput:
        if not execution_context.human_in_the_loop_safe_mode:
            logger.info(
                f"HITL block skipping review for node {node_exec_id} - safe mode disabled"
            )
            yield "approved_data", input_data.data
            yield "review_message", "Auto-approved (safe mode disabled)"
            return

        decision = await self.handle_review_decision(
            input_data=input_data.data,
            user_id=user_id,
            node_id=node_id,
            node_exec_id=node_exec_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            graph_version=graph_version,
            block_name=input_data.name,  # Use user-provided name instead of block type
            editable=input_data.editable,
        )

        if decision is None:
            return

        status = decision.review_result.status
        if status == ReviewStatus.APPROVED:
            yield "approved_data", decision.review_result.data
        elif status == ReviewStatus.REJECTED:
            yield "rejected_data", decision.review_result.data
        else:
            raise RuntimeError(f"Unexpected review status: {status}")

        if decision.message:
            yield "review_message", decision.message
