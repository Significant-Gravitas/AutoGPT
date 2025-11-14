from typing import Any

from prisma.models import PendingHumanReview

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.json import SafeJson
from backend.util.type import convert


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
        **kwargs
    ) -> BlockOutput:
        # Check if there's already an approved review for this node execution
        existing_review = await PendingHumanReview.prisma().find_unique(
            where={"nodeExecId": node_exec_id}
        )

        if existing_review and existing_review.status == "APPROVED":
            # Return the approved data (which may have been modified by the reviewer)
            # The data field now contains the approved/modified data from the review
            if (
                isinstance(existing_review.data, dict)
                and "data" in existing_review.data
            ):
                # Extract the actual data from the review data structure
                approved_data = existing_review.data["data"]
            else:
                # Fallback to the stored data directly
                approved_data = existing_review.data

            approved_data = convert(approved_data, type(input_data.data))
            yield "reviewed_data", approved_data
            yield "status", "approved"
            yield "review_message", existing_review.reviewMessage or ""

            # Clean up the review record as it's been processed
            await PendingHumanReview.prisma().delete(where={"id": existing_review.id})
            return

        elif existing_review and existing_review.status == "REJECTED":
            # Return rejection status without data
            yield "status", "rejected"
            yield "review_message", existing_review.reviewMessage or ""

            # Clean up the review record
            await PendingHumanReview.prisma().delete(where={"id": existing_review.id})
            return

        # No existing approved review, create a pending review
        review_data = {
            "data": input_data.data,
            "message": input_data.message,
            "editable": input_data.editable,
        }

        await PendingHumanReview.prisma().upsert(
            where={"nodeExecId": node_exec_id},
            data={
                "create": {
                    "userId": user_id,
                    "nodeExecId": node_exec_id,
                    "graphExecId": graph_exec_id,
                    "graphId": graph_id,
                    "graphVersion": graph_version,
                    "data": SafeJson(review_data),
                    "status": "WAITING",
                },
                "update": {"data": SafeJson(review_data), "status": "WAITING"},
            },
        )

        # This will effectively pause the execution here
        # The execution will be resumed when the review is approved
        # The manager will detect the pending review and set the status to WAITING_FOR_REVIEW
        return
