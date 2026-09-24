from datetime import datetime, timezone

import pytest
from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview

from .model import PendingHumanReviewModel


def _row(instructions: str) -> PendingHumanReview:
    return PendingHumanReview.model_construct(
        nodeExecId="node-exec",
        userId="user",
        graphExecId="graph-exec",
        graphId="graph",
        graphVersion=1,
        payload={"message_content": "hi"},
        instructions=instructions,
        editable=True,
        status=ReviewStatus.WAITING,
        reviewMessage=None,
        wasEdited=None,
        processed=False,
        createdAt=datetime(2026, 9, 23, tzinfo=timezone.utc),
        updatedAt=None,
        reviewedAt=None,
    )


@pytest.mark.parametrize(
    "instructions, block_id, action",
    [
        (
            "SendDiscordMessageBlock",
            "d0822ab5-9f8a-44a3-8971-531dd0178b6b",
            "Send Discord Message",
        ),
        ("Please check the refund amount", None, None),
    ],
)
def test_a_block_review_carries_its_block_and_action(instructions, block_id, action):
    review = PendingHumanReviewModel.from_db(_row(instructions), node_id="node")

    assert review.block_id == block_id
    assert review.action == action
