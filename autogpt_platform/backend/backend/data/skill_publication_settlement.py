"""Atomically settle a pending version and its associated review."""

from datetime import datetime, timezone

import prisma.types

from backend.data.db import transaction
from backend.data.skill_versions import PENDING_WRITE_STATE


async def settle_publication(
    user_id: str,
    *,
    version_id: str,
    review_id: str | None,
    state: str,
    disposition: str,
    reason: str,
) -> None:
    where: prisma.types.SkillVersionWhereInput = {
        "id": version_id,
        "userId": user_id,
        "state": PENDING_WRITE_STATE,
    }
    if review_id is not None:
        where["reviewId"] = review_id
    async with transaction() as tx:
        changed = await tx.skillversion.update_many(
            where=where, data={"state": state, "stateReason": reason[:1000]}
        )
        if not changed or review_id is None:
            return
        await tx.skilllearningreview.update_many(
            where={"id": review_id, "userId": user_id, "appliedVersionId": version_id},
            data={
                "disposition": disposition,
                "reason": reason[:2000],
                "completedAt": datetime.now(timezone.utc),
            },
        )
