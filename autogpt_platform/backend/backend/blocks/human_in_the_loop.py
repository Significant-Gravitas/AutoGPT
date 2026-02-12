import logging
from typing import Any

from prisma.enums import ReviewStatus

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockType,
)
from backend.blocks.helpers.review import HITLReviewHelper
from backend.data.execution import ExecutionContext
from backend.data.human_review import ReviewResult
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)


class HumanInTheLoopBlock(Block):
    """
    Pauses execution and waits for human approval or rejection of the data.

    When executed, this block creates a pending review entry and sets the node execution
    status to REVIEW. The execution remains paused until a human user either approves
    or rejects the data.

    **How it works:**
    - The input data is presented to a human reviewer
    - The reviewer can approve or reject (and optionally modify the data if editable)
    - On approval: the data flows out through the `approved_data` output pin
    - On rejection: the data flows out through the `rejected_data` output pin

    **Important:** The output pins yield the actual data itself, NOT status strings.
    The approval/rejection decision determines WHICH output pin fires, not the value.
    You do NOT need to compare the output to "APPROVED" or "REJECTED" - simply connect
    downstream blocks to the appropriate output pin for each case.

    **Example usage:**
    - Connect `approved_data` → next step in your workflow (data was approved)
    - Connect `rejected_data` → error handling or notification (data was rejected)
    """

    class Input(BlockSchemaInput):
        data: Any = SchemaField(
            description="The data to be reviewed by a human user. "
            "This exact data will be passed through to either approved_data or "
            "rejected_data output based on the reviewer's decision."
        )
        name: str = SchemaField(
            description="A descriptive name for what this data represents. "
            "This helps the reviewer understand what they are reviewing.",
        )
        editable: bool = SchemaField(
            description="Whether the human reviewer can edit the data before "
            "approving or rejecting it",
            default=True,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        approved_data: Any = SchemaField(
            description="Outputs the input data when the reviewer APPROVES it. "
            "The value is the actual data itself (not a status string like 'APPROVED'). "
            "If the reviewer edited the data, this contains the modified version. "
            "Connect downstream blocks here for the 'approved' workflow path."
        )
        rejected_data: Any = SchemaField(
            description="Outputs the input data when the reviewer REJECTS it. "
            "The value is the actual data itself (not a status string like 'REJECTED'). "
            "If the reviewer edited the data, this contains the modified version. "
            "Connect downstream blocks here for the 'rejected' workflow path."
        )
        review_message: str = SchemaField(
            description="Optional message provided by the reviewer explaining their "
            "decision. Only outputs when the reviewer provides a message; "
            "this pin does not fire if no message was given.",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="8b2a7b3c-6e9d-4a5f-8c1b-2e3f4a5b6c7d",
            description="Pause execution for human review. Data flows through "
            "approved_data or rejected_data output based on the reviewer's decision. "
            "Outputs contain the actual data, not status strings.",
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
