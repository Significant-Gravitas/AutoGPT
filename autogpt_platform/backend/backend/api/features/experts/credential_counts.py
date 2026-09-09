import asyncio
from collections import Counter

import prisma.models

from backend.api.features.experts.credentials import _seed_if_needed, _user_credentials


async def count_expert_credentials(
    user_id: str, experts: list[prisma.models.Expert]
) -> dict[str, int]:
    owned = [expert for expert in experts if expert.ownerUserId == user_id]
    if not owned:
        return {}
    semaphore = asyncio.Semaphore(8)

    async def seed(expert: prisma.models.Expert) -> None:
        async with semaphore:
            await _seed_if_needed(user_id, expert)

    await asyncio.gather(*(seed(expert) for expert in owned))
    grants, credentials = await asyncio.gather(
        prisma.models.ExpertCredential.prisma().find_many(
            where={"expertId": {"in": [expert.id for expert in owned]}}
        ),
        _user_credentials(user_id),
    )
    live_ids = {credential.id for credential in credentials}
    return dict(
        Counter(grant.expertId for grant in grants if grant.credentialId in live_ids)
    )
