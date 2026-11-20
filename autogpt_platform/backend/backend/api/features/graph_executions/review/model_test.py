from datetime import datetime, timezone

import pytest
from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview
from pydantic import ValidationError

from .model import PendingHumanReviewModel, ReviewItem, ReviewRequest


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


@pytest.mark.parametrize("rule", ["ask", "bogus"])
def test_an_answer_can_only_allow_or_judge_for_the_chat(rule):
    """Reject is the ask; nothing else may ride on an approval."""
    with pytest.raises(ValidationError):
        ReviewItem(node_exec_id="n", approved=True, chat_rule=rule)


class _CountedId(str):
    """A review id that counts how often it is compared with another."""

    comparisons = 0

    def __eq__(self, other: object) -> bool:
        _CountedId.comparisons += 1
        return str.__eq__(self, other)

    __hash__ = str.__hash__


def test_duplicate_review_ids_are_found_in_one_pass():
    """The repeated id is found with each review id looked at once."""
    count = 999
    ids = [_CountedId(f"exec-{i}") for i in range(count)] + [_CountedId("exec-0")]
    reviews = [ReviewItem.model_construct(node_exec_id=i, approved=True) for i in ids]
    _CountedId.comparisons = 0

    with pytest.raises(ValidationError, match="Duplicate review IDs found: exec-0"):
        ReviewRequest(reviews=reviews)

    assert _CountedId.comparisons <= count


def test_distinct_review_ids_pass():
    reviews = [ReviewItem(node_exec_id=f"exec-{i}", approved=True) for i in range(3)]

    assert len(ReviewRequest(reviews=reviews).reviews) == 3


def test_more_reviews_than_the_cap_are_refused_before_any_is_validated():
    """Items past the cap are never looked at: the one error is the count,
    though every item after the first thousand is invalid."""
    reviews = [{"node_exec_id": f"exec-{i}", "approved": True} for i in range(1000)]
    reviews += [{"node_exec_id": i, "approved": "maybe"} for i in range(1000)]

    with pytest.raises(ValidationError) as caught:
        ReviewRequest.model_validate({"reviews": reviews})

    errors = caught.value.errors()
    assert len(errors) == 1
    assert "At most 1000 reviews" in errors[0]["msg"]


def test_reviews_at_the_cap_pass():
    reviews = [{"node_exec_id": f"exec-{i}", "approved": True} for i in range(1000)]

    assert len(ReviewRequest.model_validate({"reviews": reviews}).reviews) == 1000
