"""Atomically settle a pending version and its associated review."""

from datetime import datetime, timezone

import prisma.types
from prisma import Prisma

from backend.data.db import transaction
from backend.data.skill_versions import PENDING_WRITE_STATE
from backend.util.json import SafeJson


async def settle_publication(
    user_id: str,
    *,
    version_id: str,
    review_id: str | None,
    state: str,
    disposition: str,
    reason: str,
) -> None:
    async with transaction() as tx:
        await settle_publication_in(
            tx,
            user_id,
            version_id=version_id,
            review_id=review_id,
            state=state,
            disposition=disposition,
            reason=reason,
        )


async def settle_publication_in(
    tx: Prisma,
    user_id: str,
    *,
    version_id: str,
    review_id: str | None,
    state: str,
    disposition: str,
    reason: str,
) -> bool:
    """:func:`settle_publication` inside the caller's transaction; False when
    the version was no longer pending."""
    where: prisma.types.SkillVersionWhereInput = {
        "id": version_id,
        "userId": user_id,
        "state": PENDING_WRITE_STATE,
    }
    if review_id is not None:
        where["reviewId"] = review_id
    changed = await tx.skillversion.update_many(
        where=where, data={"state": state, "stateReason": reason[:1000]}
    )
    if not changed or review_id is None:
        return bool(changed)
    await tx.skilllearningreview.update_many(
        where={"id": review_id, "userId": user_id, "appliedVersionId": version_id},
        data={
            "disposition": disposition,
            "reviewReplay": SafeJson(None),
            "reason": reason[:2000],
            "completedAt": datetime.now(timezone.utc),
        },
    )
    return True
