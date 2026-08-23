from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview


async def consume_approved_review(node_exec_id: str, user_id: str) -> bool:
    """Atomically claim one approved review before funding a task."""
    updated = await PendingHumanReview.prisma().update_many(
        where={
            "nodeExecId": node_exec_id,
            "userId": user_id,
            "status": ReviewStatus.APPROVED,
            "processed": False,
        },
        data={"processed": True},
    )
    return updated == 1
